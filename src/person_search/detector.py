from __future__ import annotations

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

    def ensure_ready(self) -> None:
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
        providers.append("CPUExecutionProvider")
        try:
            self._session = ort.InferenceSession(str(self.settings.yolox_model), providers=providers)
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:
            raise ModelUnavailableError(f"failed to load YOLOX model: {exc}") from exc
        self.provider_name = self._session.get_providers()[0]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.ensure_ready()
        image, ratio = _preprocess(
            frame, (self.settings.person_input_height, self.settings.person_input_width)
        )
        output = self._session.run(None, {self._input_name: image[None].astype(np.float32)})[0]
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
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), ratio


def _decode_yolox(output: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    grids: list[np.ndarray] = []
    expanded_strides: list[np.ndarray] = []
    for stride in (8, 16, 32):
        hsize, wsize = input_size[0] // stride, input_size[1] // stride
        yv, xv = np.meshgrid(np.arange(hsize), np.arange(wsize), indexing="ij")
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        expanded_strides.append(np.full((*grid.shape[:2], 1), stride))
    grid = np.concatenate(grids, axis=1).reshape(-1, 2)
    strides = np.concatenate(expanded_strides, axis=1).reshape(-1, 1)
    if output.shape[0] != grid.shape[0]:
        raise ModelUnavailableError(
            f"unexpected YOLOX output shape {output.shape}; expected {grid.shape[0]} predictions"
        )
    decoded = output.copy()
    decoded[:, :2] = (decoded[:, :2] + grid) * strides
    decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * strides
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
