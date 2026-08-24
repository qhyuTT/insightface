from __future__ import annotations

import threading
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

    def analyze(self, frame: np.ndarray, *, enrollment: bool = False) -> list[FaceObservation]:
        self.ensure_ready()
        with self._lock:
            faces = self._app.get(frame)  # type: ignore[union-attr]
        observations: list[FaceObservation] = []
        for face in faces:
            quality = assess_face(
                frame,
                face.bbox,
                face.kps,
                float(face.det_score),
                self.settings,
                enrollment=enrollment,
            )
            try:
                embedding = normalize_embedding(face.embedding)
            except (TypeError, ValueError):
                continue
            observations.append(
                FaceObservation(
                    bbox=np.asarray(face.bbox, dtype=np.float32),
                    detection_score=float(face.det_score),
                    embedding=embedding,
                    quality=quality.score,
                    landmarks=None if face.kps is None else np.asarray(face.kps, dtype=np.float32),
                    accepted=quality.accepted,
                    rejection_reasons=quality.reasons,
                )
            )
        return observations


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
