from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Protocol

import cv2
import numpy as np

from .config import Settings
from .domain import Detection
from .errors import ModelUnavailableError


class PersonDetector(Protocol):
    provider_name: str

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


@dataclass(frozen=True, slots=True)
class _DetectorRuntime:
    session: Any
    input_name: str
    provider_name: str


class YoloXOnnxDetector:
    def __init__(self, settings: Settings, confidence: float = 0.25, nms: float = 0.45):
        self.settings = settings
        self.confidence = confidence
        self.nms = nms
        self._runtime: _DetectorRuntime | None = None
        self._initialization_lock = threading.Lock()

    @property
    def provider_name(self) -> str:
        runtime = self._runtime
        return "uninitialized" if runtime is None else runtime.provider_name

    def ensure_ready(self) -> None:
        if self._runtime is not None:
            return
        with self._initialization_lock:
            if self._runtime is not None:
                return
            if not self.settings.yolox_model.is_file():
                raise ModelUnavailableError(
                    f"YOLOX model not found at {self.settings.yolox_model}; "
                    "set PERSON_SEARCH_YOLOX_MODEL to an exported YOLOX-Tiny ONNX file."
                )
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ModelUnavailableError(
                    "ONNX Runtime is missing; run `uv sync --extra inference-cpu --extra test`."
                ) from exc
            available = ort.get_available_providers()
            providers: list[str | tuple[str, dict[str, int]]] = []
            if self.settings.prefer_cuda and "CUDAExecutionProvider" in available:
                cuda_device_id = _runtime_int(
                    self.settings,
                    ("ort_cuda_device_id", "cuda_device_id"),
                    "PERSON_SEARCH_ORT_CUDA_DEVICE_ID",
                    minimum=0,
                )
                if cuda_device_id is None:
                    providers.append("CUDAExecutionProvider")
                else:
                    providers.append(
                        ("CUDAExecutionProvider", {"device_id": cuda_device_id})
                    )
            providers.append("CPUExecutionProvider")
            # Resolve these before constructing the session so malformed deployment
            # configuration fails with an actionable error even with a lightweight
            # ONNX Runtime shim (or an older runtime without SessionOptions).
            intra_threads = _runtime_int(
                self.settings,
                ("ort_intra_op_num_threads", "ort_intra_op_threads"),
                "PERSON_SEARCH_ORT_INTRA_OP_NUM_THREADS",
                minimum=0,
            )
            inter_threads = _runtime_int(
                self.settings,
                ("ort_inter_op_num_threads", "ort_inter_op_threads"),
                "PERSON_SEARCH_ORT_INTER_OP_NUM_THREADS",
                minimum=0,
            )
            try:
                session_kwargs: dict[str, object] = {"providers": providers}
                session_options_factory = getattr(ort, "SessionOptions", None)
                if session_options_factory is not None:
                    session_options = session_options_factory()
                    if intra_threads is not None:
                        session_options.intra_op_num_threads = intra_threads
                    if inter_threads is not None:
                        session_options.inter_op_num_threads = inter_threads
                    session_kwargs["sess_options"] = session_options
                session = ort.InferenceSession(
                    str(self.settings.yolox_model), **session_kwargs
                )
                inputs = session.get_inputs()
                if not inputs or not isinstance(getattr(inputs[0], "name", None), str):
                    raise ValueError("model has no named input")
                input_name = inputs[0].name
                runtime_providers = session.get_providers()
                if not runtime_providers or not isinstance(runtime_providers[0], str):
                    raise ValueError("model session has no execution provider")
                provider_name = runtime_providers[0]
            except Exception as exc:
                raise ModelUnavailableError(f"failed to load YOLOX model: {exc}") from exc
            self._runtime = _DetectorRuntime(
                session=session,
                input_name=input_name,
                provider_name=provider_name,
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.ensure_ready()
        runtime = self._runtime
        if runtime is None:  # Defensive guard for alternate ensure_ready implementations.
            raise ModelUnavailableError("YOLOX detector initialization did not complete")
        image, ratio = _preprocess(
            frame, (self.settings.person_input_height, self.settings.person_input_width)
        )
        output = runtime.session.run(None, {runtime.input_name: image[None]})[0]
        predictions = _decode_yolox(
            output[0], (self.settings.person_input_height, self.settings.person_input_width)
        )
        boxes = predictions[:, :4]
        boxes_xyxy = np.empty_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        # COCO class 0 is person.
        scores = predictions[:, 4] * predictions[:, 5]
        keep = np.where(scores >= self.confidence)[0]
        if keep.size == 0:
            return []
        selected = _nms(boxes_xyxy[keep], scores[keep], self.nms)
        return [
            Detection(bbox=boxes_xyxy[keep[index]].astype(np.float32), score=float(scores[keep[index]]))
            for index in selected
        ]


def _preprocess(frame: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    target_h, target_w = input_size
    ratio = min(target_h / frame.shape[0], target_w / frame.shape[1])
    resized = cv2.resize(
        frame,
        (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    # Write directly into the model's final CHW float32 layout. The previous
    # implementation allocated a uint8 HWC canvas, transposed it, and then copied
    # the whole frame again while converting to float32. The resized crop is still
    # uint8 (matching OpenCV's existing interpolation semantics), but the padded
    # full-size canvas is now allocated only once.
    padded = np.full((3, target_h, target_w), 114.0, dtype=np.float32)
    padded[:, : resized.shape[0], : resized.shape[1]] = resized.transpose(2, 0, 1)
    return padded, ratio


def _decode_yolox(output: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    # Accept list-like sizes as the old implementation did, while keeping the
    # cache key hashable and canonical.
    input_size = (int(input_size[0]), int(input_size[1]))
    grid, strides = _yolox_grid(input_size)
    if output.shape[0] != grid.shape[0]:
        raise ModelUnavailableError(
            f"unexpected YOLOX output shape {output.shape}; expected {grid.shape[0]} predictions"
        )
    decoded = output.copy()
    decoded[:, :2] = (decoded[:, :2] + grid) * strides
    decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * strides
    return decoded


@lru_cache(maxsize=8)
def _yolox_grid(input_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Build immutable YOLOX decode tensors once for each configured input size."""

    grids: list[np.ndarray] = []
    expanded_strides: list[np.ndarray] = []
    for stride in (8, 16, 32):
        hsize, wsize = input_size[0] // stride, input_size[1] // stride
        yv, xv = np.meshgrid(
            np.arange(hsize, dtype=np.float32),
            np.arange(wsize, dtype=np.float32),
            indexing="ij",
        )
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), stride, dtype=np.float32))
    grid = np.concatenate(grids, axis=1).reshape(-1, 2)
    strides = np.concatenate(expanded_strides, axis=1).reshape(-1, 1)
    grid.setflags(write=False)
    strides.setflags(write=False)
    return grid, strides


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    if order.size == 0:
        return keep

    # Areas are invariant across suppression rounds. Precomputing them removes two
    # allocations and four vector operations from every iteration while retaining
    # exactly the previous greedy, descending-score semantics.
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    areas = widths * heights
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[rest, 3])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = intersection / np.maximum(areas[current] + areas[rest] - intersection, 1e-6)
        order = rest[iou <= threshold]
    return keep


def _runtime_int(
    settings: Settings,
    attributes: tuple[str, ...],
    environment: str,
    *,
    minimum: int,
) -> int | None:
    """Resolve an optional runtime integer without requiring a Settings migration.

    Newer Settings versions can expose the named attributes directly. Until then,
    deployments can tune ONNX Runtime through the matching environment variables.
    An explicit Settings value wins, mirroring the rest of the configuration model.
    """

    value: object | None = None
    for attribute in attributes:
        candidate = getattr(settings, attribute, None)
        if candidate is not None:
            value = candidate
            break
    if value is None:
        raw = os.getenv(environment)
        if raw is None or not raw.strip():
            return None
        value = raw.strip()
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelUnavailableError(f"{environment} must be an integer") from exc
    if parsed < minimum:
        raise ModelUnavailableError(f"{environment} must be >= {minimum}")
    return parsed
