from __future__ import annotations

import threading
from functools import lru_cache
from typing import Protocol

import cv2
import numpy as np

from .config import Settings
from .domain import Detection
from .errors import ModelUnavailableError


class PersonDetector(Protocol):
    provider_name: str

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class YoloXOnnxDetector:
    def __init__(self, settings: Settings, confidence: float = 0.25, nms: float = 0.45):
        self.settings = settings
        self.confidence = confidence
        self.nms = nms
        self.provider_name = "uninitialized"
        self._session = None
        self._input_name = ""
        self._lock = threading.Lock()

    def ensure_ready(self) -> None:
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
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
            providers: list[str] = []
            if self.settings.prefer_cuda and "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            if not providers:
                raise ModelUnavailableError(
                    f"ONNX Runtime has no supported execution provider; available={available}"
                )
            try:
                session_options = ort.SessionOptions()
                if self.settings.onnx_intra_op_threads:
                    session_options.intra_op_num_threads = self.settings.onnx_intra_op_threads
                if self.settings.onnx_inter_op_threads:
                    session_options.inter_op_num_threads = self.settings.onnx_inter_op_threads
                self._session = ort.InferenceSession(
                    str(self.settings.yolox_model),
                    sess_options=session_options,
                    providers=providers,
                )
                self._input_name = self._session.get_inputs()[0].name
            except Exception as exc:
                raise ModelUnavailableError(f"failed to load YOLOX model: {exc}") from exc
            self.provider_name = self._session.get_providers()[0]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.ensure_ready()
        image, ratio = _preprocess(
            frame, (self.settings.person_input_height, self.settings.person_input_width)
        )
        # ``_preprocess`` already returns a contiguous float32 tensor.  Using a
        # view for the batch dimension avoids an otherwise unconditional copy on
        # every frame (which is particularly noticeable on ARM devices).
        output = self._session.run(None, {self._input_name: image[None]})[0]
        predictions = _decode_yolox(
            output[0],
            (self.settings.person_input_height, self.settings.person_input_width),
            copy=False,
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
            # ``boxes_xyxy`` is already float32.  A view is sufficient here:
            # the backing array remains alive through the returned detections,
            # and callers treat detection boxes as read-only.
            Detection(bbox=boxes_xyxy[keep[index]], score=float(scores[keep[index]]))
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
    # Allocate the tensor in its final CHW/float32 layout.  The previous
    # implementation allocated an HWC uint8 canvas, transposed it, and then
    # copied/conformed it to float32.  Writing the resized image directly into
    # the final tensor removes both the large transpose copy and one temporary
    # frame-sized allocation.
    tensor = np.full((3, target_h, target_w), 114.0, dtype=np.float32)
    tensor[:, : resized.shape[0], : resized.shape[1]] = resized.transpose(2, 0, 1)
    return tensor, ratio


@lru_cache(maxsize=8)
def _yolox_grid(input_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Build and cache the YOLOX decode grid for an input resolution.

    The grid depends only on the model input size, so rebuilding three mesh
    grids for every frame is needless work.  Cached arrays are marked
    read-only to prevent accidental mutation by decode callers.
    """

    grids: list[np.ndarray] = []
    expanded_strides: list[np.ndarray] = []
    for stride in (8, 16, 32):
        hsize, wsize = input_size[0] // stride, input_size[1] // stride
        yv, xv = np.meshgrid(
            np.arange(hsize, dtype=np.float32),
            np.arange(wsize, dtype=np.float32),
            indexing="ij",
        )
        grid = np.stack((xv, yv), axis=2).reshape(-1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((grid.shape[0], 1), stride, dtype=np.float32))
    grid = np.concatenate(grids, axis=0)
    strides = np.concatenate(expanded_strides, axis=0)
    grid.setflags(write=False)
    strides.setflags(write=False)
    return grid, strides


def _decode_yolox(
    output: np.ndarray,
    input_size: tuple[int, int],
    *,
    copy: bool = True,
) -> np.ndarray:
    """Decode raw YOLOX predictions.

    ``copy=True`` retains the historical non-mutating behavior for direct
    callers.  The detector passes ``copy=False`` because ONNX Runtime returns
    a disposable, writable float32 output for each inference; decoding in
    place avoids another ``N x 85`` allocation.  Read-only or non-float32
    inputs are copied regardless, so the fast path is always safe.
    """

    raw = np.asarray(output)
    if raw.ndim != 2:
        raise ModelUnavailableError(
            f"unexpected YOLOX output shape {raw.shape}; expected a 2-D prediction tensor"
        )
    grid, strides = _yolox_grid(tuple(input_size))
    if output.shape[0] != grid.shape[0]:
        raise ModelUnavailableError(
            f"unexpected YOLOX output shape {output.shape}; expected {grid.shape[0]} predictions"
        )
    if (
        copy
        or raw.dtype != np.float32
        or not raw.flags.writeable
        or not raw.flags.c_contiguous
    ):
        decoded = np.array(raw, dtype=np.float32, copy=True, order="C")
    else:
        decoded = raw

    # Keep all operations in-place on the fast path.  Besides avoiding a copy,
    # this avoids temporary arrays for the two coordinate transforms.
    decoded[:, :2] += grid
    decoded[:, :2] *= strides
    np.exp(decoded[:, 2:4], out=decoded[:, 2:4])
    decoded[:, 2:4] *= strides
    return decoded


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
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
        area_current = max(0.0, boxes[current, 2] - boxes[current, 0]) * max(
            0.0, boxes[current, 3] - boxes[current, 1]
        )
        area_rest = np.maximum(0.0, boxes[rest, 2] - boxes[rest, 0]) * np.maximum(
            0.0, boxes[rest, 3] - boxes[rest, 1]
        )
        iou = intersection / np.maximum(area_current + area_rest - intersection, 1e-6)
        order = rest[iou <= threshold]
    return keep
