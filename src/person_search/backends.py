from __future__ import annotations

import threading
from dataclasses import replace
from typing import Protocol

import numpy as np

from .config import Settings
from .domain import FaceObservation
from .errors import ModelUnavailableError
from .quality import assess_face, normalize_embedding


class FaceBackend(Protocol):
    model_name: str
    provider_name: str
    detection_provider_name: str
    recognition_provider_name: str

    def detect_faces(
        self, frame: np.ndarray, *, enrollment: bool = False, detection_size: int | None = None
    ) -> list[FaceObservation]: ...

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
        self, frame: np.ndarray, *, enrollment: bool = False, detection_size: int | None = None
    ) -> list[FaceObservation]:
        """Run detection and quality only. The embedding is deferred to embed_faces()."""
        self.ensure_ready()
        app = self._app
        assert app is not None
        input_size = (
            None if detection_size is None or detection_size <= 0
            else (detection_size, detection_size)
        )
        with self._lock:
            bboxes, kpss = app.det_model.detect(frame, input_size=input_size, max_num=0)
        observations: list[FaceObservation] = []
        for index in range(bboxes.shape[0]):
            bbox = bboxes[index, 0:4]
            det_score = float(bboxes[index, 4])
            kps = None if kpss is None else kpss[index]
            quality = assess_face(frame, bbox, kps, det_score, self.settings, enrollment=enrollment)
            observations.append(
                FaceObservation(
                    bbox=np.asarray(bbox, dtype=np.float32),
                    detection_score=det_score,
                    embedding=None,
                    quality=quality.score,
                    landmarks=None if kps is None else np.asarray(kps, dtype=np.float32),
                    accepted=quality.accepted,
                    rejection_reasons=quality.reasons,
                )
            )
        return observations

    def embed_faces(
        self, frame: np.ndarray, faces: list[FaceObservation]
    ) -> list[FaceObservation]:
        """Fill in ArcFace embeddings, dropping faces the recogniser cannot use.

        Crops are aligned first and pushed through a single batched session run,
        so N faces cost one inference rather than N.
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
            crops.append(face_align.norm_crop(frame, landmark=face.landmarks, image_size=image_size))
            pending.append(face)
        if not crops:
            return []

        with self._lock:
            features = recogniser.get_feat(crops)

        embedded: list[FaceObservation] = []
        for face, feature in zip(pending, np.asarray(features), strict=True):
            try:
                embedding = normalize_embedding(feature.flatten())
            except (TypeError, ValueError):
                continue
            embedded.append(replace(face, embedding=embedding))
        return embedded

    def analyze(self, frame: np.ndarray, *, enrollment: bool = False) -> list[FaceObservation]:
        return self.embed_faces(frame, self.detect_faces(frame, enrollment=enrollment))


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
