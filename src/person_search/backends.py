from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

import cv2
import numpy as np

from .config import Settings
from .domain import FaceObservation
from .errors import ModelUnavailableError
from .quality import assess_face, normalize_embedding

logger = logging.getLogger(__name__)


class FaceBackend(Protocol):
    model_name: str
    provider_name: str
    detection_provider_name: str
    recognition_provider_name: str

    def detect_faces(
        self,
        frame: np.ndarray,
        *,
        enrollment: bool = False,
        detection_size: int | Sequence[int] | None = None,
    ) -> list[FaceObservation]: ...

    def detect_faces_batch(
        self,
        frames: Sequence[np.ndarray],
        *,
        enrollment: bool = False,
        detection_size: int | Sequence[int] | None = None,
    ) -> list[list[FaceObservation]]: ...

    def embed_faces(
        self, frame: np.ndarray, faces: list[FaceObservation]
    ) -> list[FaceObservation]: ...

    def analyze(self, frame: np.ndarray, *, enrollment: bool = False) -> list[FaceObservation]: ...


class InsightFaceBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.insightface_model
        self.provider_name = "uninitialized"
        self.detection_provider_name = "uninitialized"
        self.recognition_provider_name = "uninitialized"
        self._app = None
        self._lock = threading.Lock()

    def ensure_ready(self) -> None:
        if self._app is not None:
            return
        with self._lock:
            if self._app is not None:
                return
            try:
                import onnxruntime as ort
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise ModelUnavailableError(
                    "InsightFace inference dependencies are missing; run "
                    "`uv sync --extra inference-cpu --extra test`."
                ) from exc

            available = ort.get_available_providers()
            providers: list[str] = []
            if self.settings.prefer_cuda and "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            try:
                app = FaceAnalysis(
                    name=self.settings.insightface_model,
                    root=str(self.settings.insightface_root),
                    allowed_modules=["detection", "recognition"],
                    providers=providers,
                )
                detection_size = self.settings.face_detection_size
                app.prepare(
                    ctx_id=0 if providers[0] == "CUDAExecutionProvider" else -1,
                    det_thresh=self.settings.face_detection_threshold,
                    det_size=None if detection_size <= 0 else (detection_size, detection_size),
                )
            except Exception as exc:
                raise ModelUnavailableError(f"failed to load InsightFace model: {exc}") from exc
            self._app = app
            self.detection_provider_name = _model_provider_name(app, "detection")
            self.recognition_provider_name = _model_provider_name(app, "recognition")
            if self.detection_provider_name == self.recognition_provider_name:
                self.provider_name = self.detection_provider_name
            else:
                self.provider_name = (
                    f"detection={self.detection_provider_name},"
                    f"recognition={self.recognition_provider_name}"
                )

    def detect_faces(
        self,
        frame: np.ndarray,
        *,
        enrollment: bool = False,
        detection_size: int | Sequence[int] | None = None,
    ) -> list[FaceObservation]:
        """Run detection and quality only. The embedding is deferred to embed_faces().

        An enrollment pass resolves its own scales instead of letting ``input_size``
        fall through to whatever ``prepare()`` configured. That fallback is how the
        search-side ``face_detection_size`` silently became the enrollment scale, and
        a single large scale misses the close-up a portrait always is -- see
        ``Settings.enrollment_detection_scales``. An explicit argument still wins, so
        callers that know their crop keep full control.
        """
        self.ensure_ready()
        app = self._app
        assert app is not None
        if detection_size is None and enrollment:
            detection_size = list(self.settings.enrollment_detection_scales())
        input_size = _resolve_input_size(detection_size)
        try:
            with self._lock:
                bboxes, kpss = app.det_model.detect(
                    frame, input_size=input_size, max_num=0
                )
            bboxes, kpss = _coerce_detection_result(bboxes, kpss)
        except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
            raise ModelUnavailableError("InsightFace detector returned malformed output") from exc
        return self._observations_from_detection(
            frame, bboxes, kpss, enrollment=enrollment
        )

    def detect_faces_batch(
        self,
        frames: Sequence[np.ndarray],
        *,
        enrollment: bool = False,
        detection_size: int | Sequence[int] | None = None,
    ) -> list[list[FaceObservation]]:
        """Detect faces for a group of same-sized ROI crops.

        SCRFD models shipped by InsightFace 1.x expose a batch dimension of one,
        so they cannot consume a true ``N``-image tensor without a separately
        converted model.  Keeping the batch operation here still matters: the
        service groups crops by the two fixed ROI scales, and this method holds
        the backend lock once for the whole group.  Backends with a native
        ``detect_batch`` implementation can override this method and obtain a
        real batched ONNX call without changing the service contract.
        """
        if not frames:
            return []
        self.ensure_ready()
        app = self._app
        assert app is not None
        if detection_size is None and enrollment:
            detection_size = list(self.settings.enrollment_detection_scales())
        input_size = _resolve_input_size(detection_size)
        detections: list[tuple[np.ndarray, np.ndarray | None]] | None = None
        # Converted SCRFD providers may expose a native batch entry point.  The
        # stock InsightFace detector does not, so retain the per-image fallback
        # below; this hook lets an NPU/TensorRT backend opt in without changing
        # the service-facing protocol.
        native_batch = getattr(app.det_model, "detect_batch", None)
        if callable(native_batch):
            try:
                with self._lock:
                    candidate = native_batch(
                        frames, input_size=input_size, max_num=0
                    )
                if isinstance(candidate, (list, tuple)) and len(candidate) == len(frames):
                    parsed: list[tuple[np.ndarray, np.ndarray | None]] = []
                    for item in candidate:
                        if not isinstance(item, (list, tuple)) or len(item) != 2:
                            raise TypeError("detect_batch must return (bboxes, kpss) pairs")
                        parsed.append(_coerce_detection_result(item[0], item[1]))
                    detections = parsed
            except Exception:  # noqa: BLE001 - optional fast path must be fail-open
                # ``detect_batch`` is an optional extension point.  A converted
                # model/provider can expose the method but still reject a
                # particular input shape (or fail while being initialised).  Do
                # not turn that optimisation failure into a frame-wide outage;
                # the stock, locked per-crop path remains the compatibility
                # fallback.  Exceptions from the fallback itself are allowed to
                # propagate so a genuine detector failure is still observable.
                detections = None
        if detections is None:
            detections = []
            with self._lock:
                for frame in frames:
                    try:
                        bboxes, kpss = app.det_model.detect(
                            frame, input_size=input_size, max_num=0
                        )
                        detections.append(_coerce_detection_result(bboxes, kpss))
                    except (
                        TypeError,
                        ValueError,
                        OverflowError,
                        FloatingPointError,
                    ) as exc:
                        raise ModelUnavailableError(
                            "InsightFace detector returned malformed output"
                        ) from exc
        return [
            self._observations_from_detection(
                frame, bboxes, kpss, enrollment=enrollment
            )
            for frame, (bboxes, kpss) in zip(frames, detections, strict=True)
        ]

    def _observations_from_detection(
        self,
        frame: np.ndarray,
        bboxes: np.ndarray,
        kpss: np.ndarray | None,
        *,
        enrollment: bool,
    ) -> list[FaceObservation]:
        observations: list[FaceObservation] = []
        for index in range(bboxes.shape[0]):
            bbox = bboxes[index, 0:4]
            det_score = float(bboxes[index, 4])
            kps = None if kpss is None else kpss[index]
            quality = assess_face(
                frame, bbox, kps, det_score, self.settings, enrollment=enrollment
            )
            observations.append(
                FaceObservation(
                    bbox=np.asarray(bbox, dtype=np.float32),
                    detection_score=det_score,
                    embedding=None,
                    quality=quality.score,
                    landmarks=None if kps is None else np.asarray(kps, dtype=np.float32),
                    accepted=quality.accepted,
                    rejection_reasons=quality.reasons,
                    blur_variance=quality.blur_variance,
                )
            )
        return observations

    def embed_faces(
        self, frame: np.ndarray, faces: list[FaceObservation]
    ) -> list[FaceObservation]:
        """Fill in ArcFace embeddings, dropping faces the recogniser cannot use.

        Crops are aligned first and pushed through a single batched session run,
        so N faces cost one inference rather than N. With ``embedding_flip_tta``
        each crop's mirror rides along in the same batch and the two features are
        summed before normalising -- the standard ArcFace flip average, applied
        identically on the enrollment side so both halves of the cosine match.
        """
        if not faces:
            return []
        self.ensure_ready()
        app = self._app
        assert app is not None
        recogniser = app.models.get("recognition")
        if recogniser is None:
            raise ModelUnavailableError("InsightFace recognition model is unavailable")

        from insightface.utils import face_align

        image_size = recogniser.input_size[0]
        crops: list[np.ndarray] = []
        pending: list[FaceObservation] = []
        for face in faces:
            if face.landmarks is None:
                continue
            try:
                crop = face_align.norm_crop(
                    frame, landmark=face.landmarks, image_size=image_size
                )
            except Exception:  # noqa: BLE001 - malformed crop is local to one face
                # A detector can occasionally return incomplete landmarks near
                # the image edge.  Drop that crop and keep the other faces in the
                # same micro-batch usable; the quality/detection counters still
                # retain the original observation.
                logger.debug("discarding face crop with invalid landmarks")
                continue
            if (
                not isinstance(crop, np.ndarray)
                or crop.ndim < 2
                or crop.size == 0
                or crop.shape[0] == 0
                or crop.shape[1] == 0
            ):
                logger.debug("discarding empty face crop")
                continue
            crops.append(crop)
            pending.append(face)
        if not crops:
            return []

        embedded: list[FaceObservation] = []
        flip_tta = self.settings.embedding_flip_tta
        # The setting is expressed in faces, not recogniser rows.  TTA doubles
        # the rows sent to ArcFace but keeps the memory bound predictable to an
        # operator looking at the number of detections in a frame.
        micro_batch_size = max(
            1, int(getattr(self.settings, "arcface_micro_batch_size", len(crops)))
        )
        for offset in range(0, len(crops), micro_batch_size):
            crop_chunk = crops[offset : offset + micro_batch_size]
            face_chunk = pending[offset : offset + micro_batch_size]
            batch = (
                crop_chunk + [cv2.flip(crop, 1) for crop in crop_chunk]
                if flip_tta
                else crop_chunk
            )
            with self._lock:
                try:
                    features = np.asarray(recogniser.get_feat(batch))
                except (TypeError, ValueError, OverflowError, FloatingPointError):
                    # A malformed provider response is a miss for this chunk,
                    # not a reason to pair arbitrary rows with faces.
                    continue
            # ``np.asarray(None)`` and scalar provider responses are 0-D.  They
            # have no row dimension and must be rejected before ``shape[0]``
            # below; otherwise one bad inference aborts the whole search.
            if features.ndim == 0:
                continue
            if features.ndim == 1:
                features = features.reshape(1, -1)
            if flip_tta:
                if features.shape[0] != 2 * len(face_chunk):
                    # A malformed provider result must not silently pair one
                    # person's embedding with another person's mirror.
                    continue
                try:
                    features = features[: len(face_chunk)] + features[len(face_chunk) :]
                except (TypeError, ValueError, OverflowError, FloatingPointError):
                    continue
            if features.shape[0] != len(face_chunk):
                continue
            for face, feature in zip(face_chunk, features, strict=True):
                try:
                    embedding = normalize_embedding(feature.flatten())
                except (TypeError, ValueError, OverflowError, FloatingPointError):
                    continue
                embedded.append(replace(face, embedding=embedding))
        return embedded

    def analyze(self, frame: np.ndarray, *, enrollment: bool = False) -> list[FaceObservation]:
        return self.embed_faces(frame, self.detect_faces(frame, enrollment=enrollment))


def _resolve_input_size(
    detection_size: int | Sequence[int] | None,
) -> tuple[int, int] | list[tuple[int, int]] | None:
    """Map one scale, or a list of scales, onto SCRFD's ``input_size`` argument.

    SCRFD runs every scale it is given and merges the candidates through NMS, so a
    list is a genuine multi-scale pass rather than N separate calls. ``None`` falls
    back to whatever ``prepare()`` configured.
    """
    if detection_size is None:
        return None
    if isinstance(detection_size, int):
        return None if detection_size <= 0 else (detection_size, detection_size)
    sizes = [(int(value), int(value)) for value in detection_size if int(value) > 0]
    return sizes or None


def _coerce_detection_result(
    bboxes: object, kpss: object
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate one SCRFD detector result before quality scoring.

    InsightFace normally returns ``(N, 5)`` boxes and optional ``(N, 5, 2)``
    landmarks.  A provider shim that returns a scalar, ragged list, or a
    mismatched landmark count must not reach ``_observations_from_detection``;
    otherwise malformed values can crash on an image edge or become a NaN score.
    """

    boxes = np.asarray(bboxes, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] < 5 or not np.isfinite(boxes).all():
        raise ValueError("detector boxes must be a finite (N, >=5) array")
    if kpss is None:
        landmarks = None
    else:
        landmarks = np.asarray(kpss, dtype=np.float32)
        if (
            landmarks.ndim < 3
            or landmarks.shape[0] != boxes.shape[0]
            or landmarks.shape[-1] < 2
            or not np.isfinite(landmarks).all()
        ):
            raise ValueError("detector landmarks must match the box count")
    return boxes, landmarks


def _model_provider_name(app: object, task_name: str) -> str:
    """Return the provider selected by the model's real ONNX Runtime session."""
    models = getattr(app, "models", {})
    model = models.get(task_name) if isinstance(models, dict) else None
    if model is None and isinstance(models, dict):
        model = next(
            (
                candidate
                for candidate in models.values()
                if getattr(candidate, "taskname", None) == task_name
            ),
            None,
        )
    if model is None and task_name == "detection":
        model = getattr(app, "det_model", None)

    session = getattr(model, "session", None)
    get_providers = getattr(session, "get_providers", None)
    if not callable(get_providers):
        return "unknown"
    actual_providers = get_providers()
    return str(actual_providers[0]) if actual_providers else "unknown"
