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

    def analyze(self, frame: np.ndarray, *, enrollment: bool = False) -> list[FaceObservation]: ...


class InsightFaceBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = settings.insightface_model
        self.provider_name = "uninitialized"
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
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            if not providers:
                raise ModelUnavailableError(
                    f"ONNX Runtime has no supported execution provider; available={available}"
                )
            if not self.settings.insightface_allow_download:
                model_dir = self.settings.insightface_root / "models" / self.settings.insightface_model
                if not model_dir.is_dir():
                    raise ModelUnavailableError(
                        f"InsightFace model is not present at {model_dir}; "
                        "offline mode forbids downloading it"
                    )
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
                    det_size=None
                    if detection_size <= 0
                    else (detection_size, detection_size),
                )
            except Exception as exc:
                raise ModelUnavailableError(f"failed to load InsightFace model: {exc}") from exc
            self._app = app
            self.provider_name = providers[0]

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
