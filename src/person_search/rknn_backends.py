"""RK3588 NPU backends for person and face search.

The exported models are intentionally ordinary RKNN tensors.  This module owns
the small amount of pre/post-processing that InsightFace and YOLOX previously
delegated to ONNX Runtime, while preserving the existing ``FaceBackend`` and
``PersonDetector`` protocols used by the service.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .config import Settings
from .detector import _decode_yolox, _nms
from .domain import Detection, EmbeddingContract, FaceObservation
from .errors import ModelUnavailableError
from .model_assets import sha256
from .quality import assess_face, normalize_embedding
from .rknn_runtime import (
    RKNN_TENSOR_NCHW,
    RKNN_TENSOR_NHWC,
    RknnModel,
    TensorInfo,
)

logger = logging.getLogger(__name__)

_SCRFD_STRIDES = (8, 16, 32)
_YOLOX_STRIDES = (8, 16, 32)
_SCRFD_OUTPUTS_PER_STRIDE = 3
_ARC_FACE_DST = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

ModelFactory = Callable[..., RknnModel]


class RknnYoloXDetector:
    """Run the existing YOLOX-Tiny export through RKNN on the NPU."""

    def __init__(
        self,
        settings: Settings,
        confidence: float = 0.25,
        nms: float = 0.45,
        *,
        model_factory: ModelFactory = RknnModel,
    ):
        self.settings = settings
        self.confidence = confidence
        self.nms = nms
        self._model_factory = model_factory
        self._model: RknnModel | None = None
        self._initialization_lock = threading.Lock()
        self._input_hw: tuple[int, int] | None = None

    @property
    def provider_name(self) -> str:
        return "uninitialized" if self._model is None else self._model.provider_name

    @property
    def static_input_hw(self) -> tuple[int, int] | None:
        """The converted model's fixed (height, width); ``None`` until loaded."""
        return self._input_hw

    @property
    def core_mask_applied(self) -> bool | None:
        return None if self._model is None else bool(getattr(self._model, "core_mask_applied", False))

    def ensure_ready(self) -> None:
        if self._model is not None and self._model.is_ready:
            return
        with self._initialization_lock:
            if self._model is not None and self._model.is_ready:
                return
            model_path = self.settings.rknn_yolox_model
            if model_path is None:
                raise ModelUnavailableError(
                    "PERSON_SEARCH_RKNN_YOLOX_MODEL is required when "
                    "PERSON_SEARCH_INFERENCE_BACKEND=rknn"
                )
            model = self._model_factory(
                model_path,
                self.settings.rknn_runtime_library,
                core_mask=self.settings.rknn_core_mask,
            )
            try:
                model.ensure_ready()
                input_hw = _image_input_hw(model.input_attrs[0])
                expected_hw = (
                    self.settings.person_input_height,
                    self.settings.person_input_width,
                )
                if input_hw != expected_hw:
                    raise ValueError(
                        f"YOLOX RKNN input is {input_hw}, expected configured {expected_hw}"
                    )
                if len(model.output_attrs) != 1:
                    raise ValueError(
                        f"YOLOX RKNN model must have one output, got {len(model.output_attrs)}"
                    )
                expected_predictions = _prediction_count(input_hw)
                output_shape = model.output_attrs[0].shape
                if int(np.prod(output_shape, dtype=np.int64)) != expected_predictions * 85:
                    raise ValueError(
                        f"YOLOX RKNN output {output_shape} is not a YOLOX-Tiny "
                        f"{expected_predictions}x85 head"
                    )
            except ModelUnavailableError:
                model.close()
                raise
            except Exception as exc:
                model.close()
                raise ModelUnavailableError(f"invalid RKNN YOLOX model: {exc}") from exc
            self._model = model
            self._input_hw = input_hw

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.ensure_ready()
        model = self._model
        input_hw = self._input_hw
        if model is None or input_hw is None:
            raise ModelUnavailableError("RKNN YOLOX initialization did not complete")
        image, ratio = _preprocess_yolox(frame, input_hw)
        output = model.run(image)
        raw = np.asarray(output[0], dtype=np.float32)
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim != 2 or raw.shape != (_prediction_count(input_hw), 85):
            raise ModelUnavailableError(
                f"unexpected RKNN YOLOX output shape {raw.shape}; "
                f"expected ({_prediction_count(input_hw)}, 85)"
            )
        if not np.isfinite(raw).all():
            raise ModelUnavailableError("RKNN YOLOX output contains non-finite values")
        predictions = _decode_yolox(raw, input_hw)
        boxes = predictions[:, :4]
        boxes_xyxy = np.empty_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
        boxes_xyxy /= ratio

        # COCO class 0 is person, matching the existing ONNX detector.
        scores = predictions[:, 4] * predictions[:, 5]
        keep = np.where(scores >= self.confidence)[0]
        if keep.size == 0:
            return []
        selected = _nms(boxes_xyxy[keep], scores[keep], self.nms)
        return [
            Detection(
                bbox=boxes_xyxy[keep[index]].astype(np.float32),
                score=float(scores[keep[index]]),
            )
            for index in selected
        ]

    def close(self) -> None:
        if self._model is not None:
            self._model.close()
        self._model = None
        self._input_hw = None


class RknnFaceBackend:
    """SCRFD + ArcFace backend backed by two RKNN NPU models."""

    def __init__(
        self,
        settings: Settings,
        *,
        model_factory: ModelFactory = RknnModel,
    ):
        self.settings = settings
        self.model_name = f"{settings.insightface_model}-rk3588"
        self.provider_name = "uninitialized"
        self.detection_provider_name = "uninitialized"
        self.recognition_provider_name = "uninitialized"
        self.embedding_contract: EmbeddingContract | None = None
        self._model_factory = model_factory
        self._detector: RknnModel | None = None
        self._enrollment_detector: RknnModel | None = None
        self._recognizer: RknnModel | None = None
        self._detector_hw: tuple[int, int] | None = None
        self._enrollment_detector_hw: tuple[int, int] | None = None
        self._recognizer_hw: tuple[int, int] | None = None
        self._initialization_lock = threading.Lock()

    def ensure_ready(self) -> None:
        if self._ready:
            return
        with self._initialization_lock:
            if self._ready:
                return
            detection_path = self.settings.rknn_face_detection_model
            recognition_path = self.settings.rknn_face_recognition_model
            if detection_path is None or recognition_path is None:
                raise ModelUnavailableError(
                    "PERSON_SEARCH_RKNN_FACE_DETECTION_MODEL and "
                    "PERSON_SEARCH_RKNN_FACE_RECOGNITION_MODEL are required when "
                    "PERSON_SEARCH_INFERENCE_BACKEND=rknn"
                )
            detector = self._model_factory(
                detection_path,
                self.settings.rknn_runtime_library,
                core_mask=self.settings.rknn_core_mask,
            )
            recognizer = self._model_factory(
                recognition_path,
                self.settings.rknn_runtime_library,
                core_mask=self.settings.rknn_core_mask,
            )
            enrollment_path = self.settings.rknn_enrollment_face_detection_model
            enrollment_detector: RknnModel | None = None
            enrollment_hw: tuple[int, int] | None = None
            if enrollment_path is not None and Path(enrollment_path) != Path(detection_path):
                enrollment_detector = self._model_factory(
                    enrollment_path,
                    self.settings.rknn_runtime_library,
                    core_mask=self.settings.rknn_core_mask,
                )
            try:
                detector.ensure_ready()
                recognizer.ensure_ready()
                detector_hw = _image_input_hw(detector.input_attrs[0])
                recognizer_hw = _image_input_hw(recognizer.input_attrs[0])
                if enrollment_detector is not None:
                    enrollment_detector.ensure_ready()
                    enrollment_hw = _image_input_hw(enrollment_detector.input_attrs[0])
                    if len(enrollment_detector.output_attrs) != (
                        len(_SCRFD_STRIDES) * _SCRFD_OUTPUTS_PER_STRIDE
                    ):
                        raise ValueError("enrollment SCRFD RKNN model must expose 9 outputs")
                    self._validate_scrfd_outputs(enrollment_detector, enrollment_hw)
                if len(detector.output_attrs) != len(_SCRFD_STRIDES) * _SCRFD_OUTPUTS_PER_STRIDE:
                    raise ValueError(
                        "SCRFD RKNN model must expose 9 outputs in "
                        "score/bbox/landmark stride order"
                    )
                if recognizer_hw != (112, 112):
                    raise ValueError(
                        f"ArcFace RKNN input is {recognizer_hw}, expected (112, 112)"
                    )
                recognition_elements = sum(
                    int(np.prod(info.shape, dtype=np.int64))
                    for info in recognizer.output_attrs
                )
                if recognition_elements != 512:
                    raise ValueError(
                        f"ArcFace RKNN output has {recognition_elements} values, expected 512"
                    )
                self._validate_scrfd_outputs(detector, detector_hw)
            except ModelUnavailableError:
                detector.close()
                recognizer.close()
                if enrollment_detector is not None:
                    enrollment_detector.close()
                raise
            except Exception as exc:
                detector.close()
                recognizer.close()
                if enrollment_detector is not None:
                    enrollment_detector.close()
                raise ModelUnavailableError(f"invalid RKNN face models: {exc}") from exc

            self._detector = detector
            self._enrollment_detector = enrollment_detector
            self._recognizer = recognizer
            self._detector_hw = detector_hw
            self._enrollment_detector_hw = enrollment_hw
            self._recognizer_hw = recognizer_hw
            self.detection_provider_name = detector.provider_name
            self.recognition_provider_name = recognizer.provider_name
            self.provider_name = (
                self.detection_provider_name
                if self.detection_provider_name == self.recognition_provider_name
                else f"detection={self.detection_provider_name},"
                f"recognition={self.recognition_provider_name}"
            )
            self.embedding_contract = EmbeddingContract(
                schema_version="arcface-v1-rknn",
                model_name=self.model_name,
                model_sha256=sha256(Path(recognition_path)),
                embedding_dimension=512,
                input_size=(112, 112),
                flip_tta=self.settings.embedding_flip_tta,
            )

    @property
    def _ready(self) -> bool:
        return (
            self._detector is not None
            and self._recognizer is not None
            and self.embedding_contract is not None
        )

    @property
    def static_input_hw(self) -> tuple[int, int] | None:
        """The converted detector's fixed (height, width); ``None`` until loaded.

        The RKNN graph is static, so ``detection_size`` arguments are ignored;
        this is the shape the full-frame pass actually runs at.
        """
        return self._detector_hw

    @property
    def enrollment_static_input_hw(self) -> tuple[int, int] | None:
        """Shape of the enrollment detector when it is a separate graph."""
        return self._enrollment_detector_hw

    @property
    def core_mask_applied(self) -> bool | None:
        if self._detector is None or self._recognizer is None:
            return None
        return bool(getattr(self._detector, "core_mask_applied", False)) and bool(
            getattr(self._recognizer, "core_mask_applied", False)
        )

    def detect_faces(
        self,
        frame: np.ndarray,
        *,
        enrollment: bool = False,
        detection_size: int | Sequence[int] | None = None,
    ) -> list[FaceObservation]:
        # RKNN models are statically converted at 640x640.  The argument remains
        # part of the protocol, but a size override cannot silently change the
        # model shape; deployment documentation calls this out explicitly.
        del detection_size
        self.ensure_ready()
        detector = self._detector
        detector_hw = self._detector_hw
        if enrollment and self._enrollment_detector is not None:
            # Close-up photos get the graph validated for close-ups; the far-scene
            # search graph is never asked to judge an enrollment portrait.
            detector = self._enrollment_detector
            detector_hw = self._enrollment_detector_hw
        if detector is None or detector_hw is None:
            raise ModelUnavailableError("RKNN face detector initialization did not complete")
        boxes, landmarks, scores = self._detect_candidates(frame, detector, detector_hw)
        observations: list[FaceObservation] = []
        for bbox, kps, score in zip(boxes, landmarks, scores, strict=True):
            quality = assess_face(
                frame, bbox, kps, float(score), self.settings, enrollment=enrollment
            )
            observations.append(
                FaceObservation(
                    bbox=np.asarray(bbox, dtype=np.float32),
                    detection_score=float(score),
                    embedding=None,
                    quality=quality.score,
                    landmarks=np.asarray(kps, dtype=np.float32),
                    accepted=quality.accepted,
                    rejection_reasons=quality.reasons,
                    blur_variance=quality.blur_variance,
                )
            )
        return observations

    def detect_faces_batch(
        self,
        frames: Sequence[np.ndarray],
        *,
        enrollment: bool = False,
        detection_size: int | Sequence[int] | None = None,
    ) -> list[list[FaceObservation]]:
        return [
            self.detect_faces(
                frame, enrollment=enrollment, detection_size=detection_size
            )
            for frame in frames
        ]

    def embed_faces(
        self, frame: np.ndarray, faces: list[FaceObservation]
    ) -> list[FaceObservation]:
        if not faces:
            return []
        self.ensure_ready()
        recognizer = self._recognizer
        recognizer_hw = self._recognizer_hw
        if recognizer is None or recognizer_hw is None:
            raise ModelUnavailableError("RKNN face recognizer initialization did not complete")
        embedded: list[FaceObservation] = []
        for face in faces:
            if face.landmarks is None:
                continue
            try:
                crop = _norm_crop(frame, face.landmarks, recognizer_hw)
                embedding = self._recognition_embedding(recognizer, crop)
                if self.settings.embedding_flip_tta:
                    flipped = self._recognition_embedding(recognizer, cv2.flip(crop, 1))
                    embedding = normalize_embedding(embedding + flipped)
            except (TypeError, ValueError, OverflowError, FloatingPointError):
                logger.debug("discarding face crop with invalid RKNN embedding", exc_info=True)
                continue
            embedded.append(replace(face, embedding=embedding))
        return embedded

    def analyze(self, frame: np.ndarray, *, enrollment: bool = False) -> list[FaceObservation]:
        return self.embed_faces(frame, self.detect_faces(frame, enrollment=enrollment))

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
        if self._enrollment_detector is not None:
            self._enrollment_detector.close()
        if self._recognizer is not None:
            self._recognizer.close()
        self._detector = None
        self._enrollment_detector = None
        self._recognizer = None
        self._detector_hw = None
        self._enrollment_detector_hw = None
        self._recognizer_hw = None
        self.embedding_contract = None

    @staticmethod
    def _validate_scrfd_outputs(model: RknnModel, input_hw: tuple[int, int]) -> None:
        height, width = input_hw
        for level, stride in enumerate(_SCRFD_STRIDES):
            grid_size = (height // stride) * (width // stride)
            score_info = model.output_attrs[level]
            bbox_info = model.output_attrs[3 + level]
            kps_info = model.output_attrs[6 + level]
            if int(np.prod(score_info.shape, dtype=np.int64)) % grid_size != 0:
                raise ValueError(f"SCRFD score output {score_info.shape} has invalid grid size")
            anchors = int(np.prod(score_info.shape, dtype=np.int64)) // grid_size
            if anchors <= 0:
                raise ValueError("SCRFD score output has no anchors")
            if int(np.prod(bbox_info.shape, dtype=np.int64)) != grid_size * anchors * 4:
                raise ValueError(f"SCRFD bbox output {bbox_info.shape} does not match score output")
            if int(np.prod(kps_info.shape, dtype=np.int64)) != grid_size * anchors * 10:
                raise ValueError(f"SCRFD landmark output {kps_info.shape} does not match score output")

    def _detect_candidates(
        self,
        frame: np.ndarray,
        detector: RknnModel,
        input_hw: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_image, ratio = _preprocess_scrfd_rgb(frame, input_hw)
        outputs = detector.run(input_image)
        self._validate_scrfd_outputs(detector, input_hw)
        boxes: list[np.ndarray] = []
        landmarks: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        height, width = input_hw
        threshold = self.settings.face_detection_threshold
        for level, stride in enumerate(_SCRFD_STRIDES):
            score = np.asarray(outputs[level], dtype=np.float32).reshape(-1)
            bbox = np.asarray(outputs[3 + level], dtype=np.float32).reshape(-1, 4)
            kps = np.asarray(outputs[6 + level], dtype=np.float32).reshape(-1, 10)
            grid_h, grid_w = height // stride, width // stride
            grid_count = grid_h * grid_w
            if score.size % grid_count != 0:
                raise ModelUnavailableError("SCRFD output grid is incompatible with input size")
            anchors = score.size // grid_count
            if bbox.shape[0] != score.size or kps.shape[0] != score.size:
                raise ModelUnavailableError("SCRFD output tensors have mismatched element counts")
            if not np.isfinite(score).all() or not np.isfinite(bbox).all() or not np.isfinite(kps).all():
                raise ModelUnavailableError("SCRFD output contains non-finite values")
            yv, xv = np.meshgrid(
                np.arange(grid_h, dtype=np.float32),
                np.arange(grid_w, dtype=np.float32),
                indexing="ij",
            )
            centers = np.stack((xv, yv), axis=-1).reshape(-1, 2) * stride
            centers = np.repeat(centers, anchors, axis=0)
            keep = np.where(score >= threshold)[0]
            if keep.size == 0:
                continue
            selected_centers = centers[keep]
            selected_bbox = bbox[keep] * stride
            selected_kps = kps[keep].reshape(-1, 5, 2) * stride
            decoded = np.empty((keep.size, 4), dtype=np.float32)
            decoded[:, 0] = selected_centers[:, 0] - selected_bbox[:, 0]
            decoded[:, 1] = selected_centers[:, 1] - selected_bbox[:, 1]
            decoded[:, 2] = selected_centers[:, 0] + selected_bbox[:, 2]
            decoded[:, 3] = selected_centers[:, 1] + selected_bbox[:, 3]
            selected_kps += selected_centers[:, None, :]
            boxes.append(decoded)
            landmarks.append(selected_kps)
            scores.append(score[keep])
        if not boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 5, 2), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            )
        all_boxes = np.concatenate(boxes, axis=0)
        all_landmarks = np.concatenate(landmarks, axis=0)
        all_scores = np.concatenate(scores, axis=0)
        all_boxes /= ratio
        all_landmarks /= ratio
        frame_height, frame_width = frame.shape[:2]
        all_boxes[:, [0, 2]] = np.clip(all_boxes[:, [0, 2]], 0, frame_width - 1)
        all_boxes[:, [1, 3]] = np.clip(all_boxes[:, [1, 3]], 0, frame_height - 1)
        all_landmarks[:, :, 0] = np.clip(all_landmarks[:, :, 0], 0, frame_width - 1)
        all_landmarks[:, :, 1] = np.clip(all_landmarks[:, :, 1], 0, frame_height - 1)
        keep = _nms(all_boxes, all_scores, 0.4)
        return all_boxes[keep], all_landmarks[keep], all_scores[keep]

    @staticmethod
    def _recognition_embedding(model: RknnModel, crop: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        output = model.run(np.ascontiguousarray(rgb, dtype=np.uint8))
        feature = np.asarray(output[0], dtype=np.float32).reshape(-1)
        if feature.size != 512 or not np.isfinite(feature).all():
            raise ValueError("ArcFace RKNN output is not a finite 512-dimensional vector")
        return normalize_embedding(feature)


def _image_input_hw(info: TensorInfo) -> tuple[int, int]:
    shape = info.shape
    if len(shape) != 4:
        raise ValueError(f"image RKNN input must be rank 4, got {shape}")
    if info.tensor_format == RKNN_TENSOR_NHWC:
        _, height, width, channels = shape
    elif info.tensor_format == RKNN_TENSOR_NCHW:
        _, channels, height, width = shape
    else:
        raise ValueError(f"unsupported RKNN image input format {info.tensor_format}")
    if channels != 3 or height <= 0 or width <= 0:
        raise ValueError(f"RKNN image input must be 3-channel, got {shape}")
    return int(height), int(width)


def _prediction_count(input_hw: tuple[int, int]) -> int:
    height, width = input_hw
    return sum((height // stride) * (width // stride) for stride in _YOLOX_STRIDES)


def _preprocess_yolox(
    frame: np.ndarray, input_hw: tuple[int, int]
) -> tuple[np.ndarray, float]:
    _validate_frame(frame)
    target_h, target_w = input_hw
    ratio = min(target_h / frame.shape[0], target_w / frame.shape[1])
    resized_width = max(1, int(frame.shape[1] * ratio))
    resized_height = max(1, int(frame.shape[0] * ratio))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    padded[:resized_height, :resized_width] = resized
    return padded, ratio


def _preprocess_scrfd_rgb(
    frame: np.ndarray, input_hw: tuple[int, int]
) -> tuple[np.ndarray, float]:
    _validate_frame(frame)
    target_h, target_w = input_hw
    ratio = min(target_h / frame.shape[0], target_w / frame.shape[1])
    resized_width = max(1, min(target_w, int(frame.shape[1] * ratio)))
    resized_height = max(1, min(target_h, int(frame.shape[0] * ratio)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded[:resized_height, :resized_width] = resized
    return padded, ratio


def _norm_crop(
    frame: np.ndarray, landmarks: np.ndarray, input_hw: tuple[int, int]
) -> np.ndarray:
    _validate_frame(frame)
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError("face landmarks must be a finite 5x2 array")
    height, width = input_hw
    if (height, width) != (112, 112):
        raise ValueError(f"ArcFace alignment only supports 112x112, got {input_hw}")
    transform = _similarity_transform(points, _ARC_FACE_DST)
    if transform is None or not np.isfinite(transform).all():
        raise ValueError("could not estimate ArcFace similarity transform")
    crop = cv2.warpAffine(frame, transform, (width, height), borderValue=0)
    if crop.size == 0 or crop.shape[:2] != (height, width):
        raise ValueError("ArcFace alignment returned an invalid crop")
    return crop


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Least-squares 2-D similarity from ``src`` to ``dst`` (Umeyama, PAMI 1991).

    This is what ``skimage.transform.SimilarityTransform.estimate`` computes and
    therefore what InsightFace's ``norm_crop`` -- and the ONNX backend -- align
    ArcFace crops with; the network was trained on crops produced this way.  The
    previous ``cv2.estimateAffinePartial2D(method=LMEDS)`` minimises the *median*
    residual, so with five landmarks it may fit three and ignore two.  On sharp,
    well-detected faces the two agree; on the noisy landmarks of a far face they
    do not, and that is exactly where every point of similarity is needed.

    Returns the 2x3 affine matrix for ``cv2.warpAffine``, or ``None`` when the
    points are degenerate.
    """
    source = np.asarray(src, dtype=np.float64)
    target = np.asarray(dst, dtype=np.float64)
    count, dim = source.shape
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_demean = source - source_mean
    target_demean = target - target_mean
    covariance = target_demean.T @ source_demean / count
    reflection = np.ones(dim)
    if np.linalg.det(covariance) < 0:
        reflection[dim - 1] = -1
    u, singular, vt = np.linalg.svd(covariance)
    tolerance = singular.max() * max(covariance.shape) * np.finfo(float).eps
    rank = int(np.count_nonzero(singular > tolerance))
    if rank == 0:
        return None
    rotation = np.eye(dim)
    if rank == dim - 1:
        if np.linalg.det(u) * np.linalg.det(vt) > 0:
            rotation = u @ vt
        else:
            flipped = reflection.copy()
            flipped[dim - 1] = -1
            rotation = u @ np.diag(flipped) @ vt
    else:
        rotation = u @ np.diag(reflection) @ vt
    variance = source_demean.var(axis=0).sum()
    if variance <= 0:
        return None
    scale = float(singular @ reflection) / variance
    transform = np.empty((dim, dim + 1), dtype=np.float64)
    transform[:, :dim] = scale * rotation
    transform[:, dim] = target_mean - scale * (rotation @ source_mean)
    return transform.astype(np.float32)


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ModelUnavailableError("RKNN backends require a BGR HWC frame")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ModelUnavailableError("RKNN backends received an empty frame")


__all__ = ["RknnFaceBackend", "RknnYoloXDetector"]
