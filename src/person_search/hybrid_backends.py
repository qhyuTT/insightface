"""Performance-oriented RPP + RKNN face backend for RK3588."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import numpy as np

from .backends import FaceBackend
from .config import Settings
from .domain import EmbeddingContract, FaceObservation
from .errors import ModelUnavailableError
from .inference_timing import record_inference_timing
from .rknn_backends import RknnFaceBackend
from .rpp_backends import (
    EngineFactory,
    _decode_scrfd,
    _InferenceEngine,
    _load_rpp_engine,
    _preprocess_scrfd,
    _scrfd_output_shapes,
    _scrfd_static_model_path,
    _validate_static_scrfd_model,
)


class RppScrfdRknnFaceBackend:
    """Use RPP for face detection and RKNN NPU for ArcFace embeddings."""

    def __init__(
        self,
        settings: Settings,
        *,
        recognition_backend: FaceBackend | None = None,
        engine_factory: EngineFactory = _load_rpp_engine,
    ):
        self.settings = settings
        self.model_name = f"{settings.insightface_model}-rpp-scrfd-rknn-arcface"
        self.provider_name = "uninitialized"
        self.detection_provider_name = "uninitialized"
        self.recognition_provider_name = "uninitialized"
        self.embedding_contract: EmbeddingContract | None = None
        self._recognition_backend = recognition_backend or RknnFaceBackend(settings)
        self._engine_factory = engine_factory
        self._search_detector: _InferenceEngine | None = None
        self._enrollment_detector: _InferenceEngine | None = None
        self._detection_output_names: tuple[str, ...] = ()
        self._initialization_lock = threading.Lock()

    @property
    def detection_execution_wait_mode(self) -> str:
        return str(getattr(self._search_detector, "execution_wait_mode", "uninitialized"))

    @property
    def static_input_hw(self) -> tuple[int, int]:
        return (self.settings.rpp_face_detection_height, self.settings.rpp_face_detection_width)

    @property
    def enrollment_static_input_hw(self) -> tuple[int, int]:
        return (
            self.settings.rpp_enrollment_face_detection_height,
            self.settings.rpp_enrollment_face_detection_width,
        )

    @property
    def _ready(self) -> bool:
        return (
            self._search_detector is not None
            and self._enrollment_detector is not None
            and self.embedding_contract is not None
        )

    def ensure_ready(self) -> None:
        if self._ready:
            return
        with self._initialization_lock:
            if self._ready:
                return
            search_detector: _InferenceEngine | None = None
            enrollment_detector: _InferenceEngine | None = None
            try:
                self._recognition_backend.ensure_ready()
                contract = self._recognition_backend.embedding_contract
                recognition_provider = self._recognition_backend.recognition_provider_name
                if recognition_provider != "RKNN_NPU":
                    raise ModelUnavailableError(
                        "hybrid face recognition must run on RKNN_NPU; got "
                        f"{recognition_provider}"
                    )
                if contract is None or contract.schema_version != "arcface-v1-rknn":
                    raise ModelUnavailableError(
                        "hybrid face recognition must preserve the arcface-v1-rknn "
                        "embedding contract"
                    )

                detector_path = (
                    self.settings.insightface_root
                    / "models"
                    / self.settings.insightface_model
                    / "det_10g.onnx"
                )
                if not detector_path.is_file():
                    raise ModelUnavailableError(
                        "RPP SCRFD ONNX weight is missing from the configured model root"
                    )
                detector_bytes = detector_path.read_bytes()
                search_hw = self.static_input_hw
                enrollment_hw = self.enrollment_static_input_hw
                search_path = _scrfd_static_model_path(
                    detector_bytes,
                    input_hw=search_hw,
                    model_dir=self.settings.rpp_static_model_dir,
                )
                enrollment_path = _scrfd_static_model_path(
                    detector_bytes,
                    input_hw=enrollment_hw,
                    model_dir=self.settings.rpp_static_model_dir,
                )
                missing_graphs = [
                    path for path in {search_path, enrollment_path} if not path.is_file()
                ]
                if missing_graphs:
                    paths = ", ".join(str(path) for path in missing_graphs)
                    raise ModelUnavailableError(
                        "pre-specialized SCRFD graph(s) are missing: "
                        f"{paths}; prepare them before deployment with "
                        "scripts/prepare_rpp_scrfd.py"
                    )

                search_graph = search_path.read_bytes()
                output_names = _validate_static_scrfd_model(
                    search_graph, input_hw=search_hw
                )
                enrollment_graph = search_graph
                enrollment_names = output_names
                if enrollment_path != search_path:
                    enrollment_graph = enrollment_path.read_bytes()
                    enrollment_names = _validate_static_scrfd_model(
                        enrollment_graph, input_hw=enrollment_hw
                    )
                if enrollment_names != output_names:
                    raise ModelUnavailableError(
                        "SCRFD output names changed between search and enrollment graphs"
                    )

                self.settings.rpp_engine_cache_dir.mkdir(parents=True, exist_ok=True)
                search_detector = self._engine_factory(
                    search_graph,
                    cache_dir=self.settings.rpp_engine_cache_dir,
                    model_key="scrfd-search",
                    engine_cache_enabled=self.settings.rpp_engine_cache_enabled,
                    release_gil=self.settings.rpp_release_gil,
                    input_shape=(3, *search_hw),
                    output_shapes=dict(
                        zip(output_names, _scrfd_output_shapes(search_hw), strict=True)
                    ),
                )
                if enrollment_hw == search_hw:
                    enrollment_detector = search_detector
                else:
                    enrollment_detector = self._engine_factory(
                        enrollment_graph,
                        cache_dir=self.settings.rpp_engine_cache_dir,
                        model_key="scrfd-enrollment",
                        engine_cache_enabled=self.settings.rpp_engine_cache_enabled,
                        release_gil=self.settings.rpp_release_gil,
                        input_shape=(3, *enrollment_hw),
                        output_shapes=dict(
                            zip(
                                enrollment_names,
                                _scrfd_output_shapes(enrollment_hw),
                                strict=True,
                            )
                        ),
                    )

                if search_detector.input_shape != (3, *search_hw):
                    raise ModelUnavailableError("RPP search SCRFD input shape is incorrect")
                if enrollment_detector.input_shape != (3, *enrollment_hw):
                    raise ModelUnavailableError("RPP enrollment SCRFD input shape is incorrect")

                # These calls execute the configured SCRFD graphs on the real RPP device.
                search_detector.run(np.zeros(search_detector.input_shape, dtype=np.float32))
                if enrollment_detector is not search_detector:
                    enrollment_detector.run(
                        np.zeros(enrollment_detector.input_shape, dtype=np.float32)
                    )
            except ModelUnavailableError:
                self._close_engines(search_detector, enrollment_detector)
                self._close_recognition_backend()
                raise
            except Exception as exc:
                self._close_engines(search_detector, enrollment_detector)
                self._close_recognition_backend()
                raise ModelUnavailableError(
                    f"RPP/RKNN hybrid face model initialization failed: {exc}"
                ) from exc

            self._search_detector = search_detector
            self._enrollment_detector = enrollment_detector
            self._detection_output_names = output_names
            self.embedding_contract = contract
            self.detection_provider_name = "RPP_BF16"
            self.recognition_provider_name = recognition_provider
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
        del detection_size
        self.ensure_ready()
        detector = self._enrollment_detector if enrollment else self._search_detector
        input_hw = self.enrollment_static_input_hw if enrollment else self.static_input_hw
        if detector is None:
            raise ModelUnavailableError("RPP SCRFD engine did not initialize")
        preprocess_started = time.monotonic()
        input_tensor, ratio = _preprocess_scrfd(frame, input_hw)
        record_inference_timing("preprocess", time.monotonic() - preprocess_started)
        outputs = detector.run(input_tensor)
        postprocess_started = time.monotonic()
        boxes, landmarks, scores = _decode_scrfd(
            frame,
            outputs,
            self._detection_output_names,
            input_hw,
            ratio,
            threshold=self.settings.face_detection_threshold,
        )
        from .quality import assess_face

        observations: list[FaceObservation] = []
        for bbox, keypoints, score in zip(boxes, landmarks, scores, strict=True):
            quality = assess_face(
                frame, bbox, keypoints, float(score), self.settings, enrollment=enrollment
            )
            observations.append(
                FaceObservation(
                    bbox=np.asarray(bbox, dtype=np.float32),
                    detection_score=float(score),
                    embedding=None,
                    quality=quality.score,
                    landmarks=np.asarray(keypoints, dtype=np.float32),
                    accepted=quality.accepted,
                    rejection_reasons=quality.reasons,
                    blur_variance=quality.blur_variance,
                )
            )
        record_inference_timing("postprocess", time.monotonic() - postprocess_started)
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
        return self._recognition_backend.embed_faces(frame, faces)

    def analyze(self, frame: np.ndarray, *, enrollment: bool = False) -> list[FaceObservation]:
        return self.embed_faces(frame, self.detect_faces(frame, enrollment=enrollment))

    def close(self) -> None:
        self._close_engines(self._search_detector, self._enrollment_detector)
        self._search_detector = None
        self._enrollment_detector = None
        self._detection_output_names = ()
        self._close_recognition_backend()
        self.provider_name = "uninitialized"
        self.detection_provider_name = "uninitialized"
        self.recognition_provider_name = "uninitialized"
        self.embedding_contract = None

    @staticmethod
    def _close_engines(*engines: _InferenceEngine | None) -> None:
        unique_engines = {id(engine): engine for engine in engines if engine is not None}
        for engine in unique_engines.values():
            engine.close()

    def _close_recognition_backend(self) -> None:
        close = getattr(self._recognition_backend, "close", None)
        if callable(close):
            close()
