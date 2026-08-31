from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from person_search.backends import InsightFaceBackend, _embedding_contract, _resolve_input_size
from person_search.config import Settings
from person_search.domain import EmbeddingContract, FaceObservation
from person_search.errors import ModelUnavailableError
from person_search.model_assets import BUFFALO_L_EMBEDDING_MANIFEST

FAKE_CONTRACT = EmbeddingContract(
    schema_version="arcface-v1",
    model_name="buffalo_l",
    model_sha256="0" * 64,
    embedding_dimension=512,
    input_size=(112, 112),
    flip_tta=False,
)


def test_face_detection_uses_insightface_auto_mode_by_default() -> None:
    assert Settings().face_detection_size == 0


def test_insightface_reports_actual_model_session_providers(monkeypatch) -> None:
    monkeypatch.setattr("person_search.backends._embedding_contract", lambda *args, **kwargs: FAKE_CONTRACT)
    class FakeSession:
        def __init__(self, provider: str):
            self.provider = provider

        def get_providers(self) -> list[str]:
            return [self.provider]

    class FakeFaceAnalysis:
        requested_providers: ClassVar[list[str]] = []

        def __init__(self, **kwargs):
            type(self).requested_providers = kwargs["providers"]
            self.models = {
                "detector-model": SimpleNamespace(
                    taskname="detection", session=FakeSession("CPUExecutionProvider")
                ),
                "recognition": SimpleNamespace(
                    taskname="recognition", session=FakeSession("CPUExecutionProvider")
                ),
            }

        def prepare(self, **kwargs) -> None:
            pass

    ort_module = ModuleType("onnxruntime")
    ort_module.get_available_providers = lambda: [  # type: ignore[attr-defined]
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    insightface_module = ModuleType("insightface")
    app_module = ModuleType("insightface.app")
    app_module.FaceAnalysis = FakeFaceAnalysis  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)
    monkeypatch.setitem(sys.modules, "insightface", insightface_module)
    monkeypatch.setitem(sys.modules, "insightface.app", app_module)

    backend = InsightFaceBackend(Settings(prefer_cuda=True))
    backend.ensure_ready()

    assert FakeFaceAnalysis.requested_providers[0] == "CUDAExecutionProvider"
    assert backend.detection_provider_name == "CPUExecutionProvider"
    assert backend.recognition_provider_name == "CPUExecutionProvider"
    assert backend.provider_name == "CPUExecutionProvider"


def test_insightface_summarizes_mixed_actual_providers(monkeypatch) -> None:
    monkeypatch.setattr("person_search.backends._embedding_contract", lambda *args, **kwargs: FAKE_CONTRACT)
    class FakeSession:
        def __init__(self, provider: str):
            self.provider = provider

        def get_providers(self) -> list[str]:
            return [self.provider]

    class FakeFaceAnalysis:
        def __init__(self, **kwargs):
            self.models = {
                "detection": SimpleNamespace(session=FakeSession("CUDAExecutionProvider")),
                "recognition": SimpleNamespace(session=FakeSession("CPUExecutionProvider")),
            }

        def prepare(self, **kwargs) -> None:
            pass

    ort_module = ModuleType("onnxruntime")
    ort_module.get_available_providers = lambda: [  # type: ignore[attr-defined]
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    insightface_module = ModuleType("insightface")
    app_module = ModuleType("insightface.app")
    app_module.FaceAnalysis = FakeFaceAnalysis  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)
    monkeypatch.setitem(sys.modules, "insightface", insightface_module)
    monkeypatch.setitem(sys.modules, "insightface.app", app_module)

    backend = InsightFaceBackend(Settings(prefer_cuda=True))
    backend.ensure_ready()

    assert backend.provider_name == (
        "detection=CUDAExecutionProvider,recognition=CPUExecutionProvider"
    )


def test_insightface_failed_contract_validation_does_not_publish_and_can_retry(
    monkeypatch,
) -> None:
    class FakeSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

    class FakeFaceAnalysis:
        def __init__(self, **kwargs):
            self.models = {
                "detection": SimpleNamespace(session=FakeSession()),
                "recognition": SimpleNamespace(session=FakeSession()),
            }

        def prepare(self, **kwargs):
            pass

    attempts = 0

    def contract_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("contract metadata unavailable")
        return FAKE_CONTRACT

    ort_module = ModuleType("onnxruntime")
    ort_module.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]
    insightface_module = ModuleType("insightface")
    app_module = ModuleType("insightface.app")
    app_module.FaceAnalysis = FakeFaceAnalysis  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)
    monkeypatch.setitem(sys.modules, "insightface", insightface_module)
    monkeypatch.setitem(sys.modules, "insightface.app", app_module)
    monkeypatch.setattr("person_search.backends._embedding_contract", contract_once)
    backend = InsightFaceBackend(Settings(prefer_cuda=False))

    with pytest.raises(ModelUnavailableError, match="contract metadata unavailable"):
        backend.ensure_ready()
    assert backend._app is None
    assert backend.embedding_contract is None
    assert backend.provider_name == "uninitialized"

    backend.ensure_ready()

    assert backend.embedding_contract == FAKE_CONTRACT
    assert backend.provider_name == "CPUExecutionProvider"


def test_embedding_contract_validates_fixed_hash_and_shapes(monkeypatch, tmp_path) -> None:
    model_file = tmp_path / BUFFALO_L_EMBEDDING_MANIFEST.recognition_filename
    model_file.write_bytes(b"recognition-model")
    recogniser = SimpleNamespace(
        model_file=str(model_file),
        input_size=BUFFALO_L_EMBEDDING_MANIFEST.input_size,
        output_shape=(1, BUFFALO_L_EMBEDDING_MANIFEST.embedding_dimension),
    )
    app = SimpleNamespace(models={"recognition": recogniser})
    monkeypatch.setattr(
        "person_search.backends.sha256",
        lambda path: BUFFALO_L_EMBEDDING_MANIFEST.recognition_sha256,
    )

    contract = _embedding_contract(app, model_name="buffalo_l", flip_tta=True)

    assert contract.model_sha256 == BUFFALO_L_EMBEDDING_MANIFEST.recognition_sha256
    assert contract.embedding_dimension == 512
    assert contract.input_size == (112, 112)
    assert contract.flip_tta is True

    monkeypatch.setattr("person_search.backends.sha256", lambda path: "f" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _embedding_contract(app, model_name="buffalo_l", flip_tta=True)


def test_resolve_input_size_maps_one_scale_or_many_onto_scrfd() -> None:
    assert _resolve_input_size(None) is None
    assert _resolve_input_size(0) is None
    assert _resolve_input_size(320) == (320, 320)
    # SCRFD runs every scale it is handed and merges the candidates with NMS, so a
    # list is one multi-scale pass rather than N calls.
    assert _resolve_input_size([128, 640, 1280]) == [(128, 128), (640, 640), (1280, 1280)]
    assert _resolve_input_size([0]) is None


class _RecordingDetector:
    """Records the input_size SCRFD would actually be called with."""

    def __init__(self) -> None:
        self.input_sizes: list[object] = []

    def detect(self, frame, input_size=None, max_num=0):
        self.input_sizes.append(input_size)
        return np.empty((0, 5), dtype=np.float32), None


class _NativeBatchFailureDetector(_RecordingDetector):
    """Advertises the optional batch hook but rejects it at runtime."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_calls = 0

    def detect_batch(self, frames, input_size=None, max_num=0):
        self.batch_calls += 1
        raise RuntimeError("native batch is unsupported for this input")


def _detect_scales(settings: Settings, *, enrollment: bool, detection_size=None) -> object:
    backend = InsightFaceBackend(settings)
    detector = _RecordingDetector()
    backend._app = SimpleNamespace(det_model=detector)
    backend.embedding_contract = FAKE_CONTRACT
    backend.detect_faces(
        np.zeros((1280, 960, 3), dtype=np.uint8),
        enrollment=enrollment,
        detection_size=detection_size,
    )
    return detector.input_sizes[0]


def test_enrollment_keeps_auto_scales_when_search_pins_a_large_one() -> None:
    """A 1280 search scale must not follow the enrollment photo into the detector.

    SCRFD upscales to fill its input with no ceiling, so on a single 1280 pass a
    frame-filling portrait overshoots the stride-32 anchors and is missed outright
    (measured: MISS from 0.45 frame-height up, against 0.865-0.905 on Auto). The
    detector used to inherit this scale through prepare(), which broke enrollment.
    """
    resolved = _detect_scales(Settings(face_detection_size=1280), enrollment=True)

    assert resolved == [(128, 128), (640, 640)]


def test_search_pass_is_unaffected_by_the_enrollment_scale() -> None:
    # A search pass passes its scales explicitly and must keep doing so.
    assert _detect_scales(Settings(face_detection_size=1280), enrollment=False) is None
    assert _detect_scales(
        Settings(enrollment_detection_size=320), enrollment=False, detection_size=[1280]
    ) == [(1280, 1280)]


def test_enrollment_scale_is_overridable_and_explicit_arguments_win() -> None:
    assert _detect_scales(Settings(enrollment_detection_size=640), enrollment=True) == [(640, 640)]
    # An explicit argument still wins, so an ROI caller keeps control of its crop.
    assert _detect_scales(Settings(), enrollment=True, detection_size=320) == (320, 320)


def _landmarked_face() -> FaceObservation:
    return FaceObservation(
        bbox=np.asarray([100, 100, 160, 160], dtype=np.float32),
        detection_score=0.99,
        embedding=None,
        quality=0.9,
        landmarks=np.asarray(
            [[115, 120], [145, 120], [130, 135], [118, 148], [142, 148]], dtype=np.float32
        ),
    )


class _StubRecogniser:
    """Returns a distinct feature for the originals and for the mirrored half."""

    input_size = (112, 112)

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def get_feat(self, crops):
        self.batch_sizes.append(len(crops))
        half = len(crops) // 2 or len(crops)
        rows = [[1.0, 0.0]] * half
        if len(crops) > half:
            rows += [[0.0, 1.0]] * (len(crops) - half)
        return np.asarray(rows, dtype=np.float32)


def test_flip_tta_sends_one_batch_and_averages_the_mirror() -> None:
    pytest.importorskip("insightface", reason="flip TTA needs face_align, not a model")
    recogniser = _StubRecogniser()
    backend = InsightFaceBackend(Settings(embedding_flip_tta=True))
    backend._app = SimpleNamespace(models={"recognition": recogniser})
    backend.embedding_contract = FAKE_CONTRACT
    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    embedded = backend.embed_faces(frame, [_landmarked_face()])

    # One inference for both the crop and its mirror, not two.
    assert recogniser.batch_sizes == [2]
    assert embedded[0].embedding == pytest.approx([0.70710678, 0.70710678])


def test_flip_tta_can_be_switched_off() -> None:
    pytest.importorskip("insightface", reason="flip TTA needs face_align, not a model")
    recogniser = _StubRecogniser()
    backend = InsightFaceBackend(Settings(embedding_flip_tta=False))
    backend._app = SimpleNamespace(models={"recognition": recogniser})
    backend.embedding_contract = FAKE_CONTRACT
    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    embedded = backend.embed_faces(frame, [_landmarked_face()])

    assert recogniser.batch_sizes == [1]
    assert embedded[0].embedding == pytest.approx([1.0, 0.0])


def test_detect_faces_batch_preserves_one_result_list_per_crop() -> None:
    backend = InsightFaceBackend(Settings())
    detector = _RecordingDetector()
    backend._app = SimpleNamespace(det_model=detector)
    backend.embedding_contract = FAKE_CONTRACT
    frames = [
        np.zeros((120, 160, 3), dtype=np.uint8),
        np.zeros((90, 140, 3), dtype=np.uint8),
    ]

    result = backend.detect_faces_batch(frames, detection_size=320)

    assert result == [[], []]
    assert detector.input_sizes == [(320, 320), (320, 320)]


def test_detect_faces_batch_falls_back_when_optional_native_hook_fails() -> None:
    backend = InsightFaceBackend(Settings())
    detector = _NativeBatchFailureDetector()
    backend._app = SimpleNamespace(det_model=detector)
    backend.embedding_contract = FAKE_CONTRACT
    frames = [
        np.zeros((120, 160, 3), dtype=np.uint8),
        np.zeros((90, 140, 3), dtype=np.uint8),
    ]

    result = backend.detect_faces_batch(frames, detection_size=320)

    assert result == [[], []]
    assert detector.batch_calls == 1
    # The optional fast path failed, but both crops still received the ordinary
    # detector call under the backend lock.
    assert detector.input_sizes == [(320, 320), (320, 320)]


def test_detect_faces_rejects_malformed_detector_result() -> None:
    backend = InsightFaceBackend(Settings())

    class MalformedDetector:
        def detect(self, frame, input_size=None, max_num=0):
            return np.asarray(0.0, dtype=np.float32), None

    backend._app = SimpleNamespace(det_model=MalformedDetector())
    backend.embedding_contract = FAKE_CONTRACT

    with pytest.raises(ModelUnavailableError, match="malformed output"):
        backend.detect_faces(np.zeros((120, 160, 3), dtype=np.uint8))


def test_detect_faces_rejects_landmarks_with_wrong_rank() -> None:
    backend = InsightFaceBackend(Settings())

    class MalformedDetector:
        def detect(self, frame, input_size=None, max_num=0):
            return np.zeros((1, 5), dtype=np.float32), np.zeros((1,), dtype=np.float32)

    backend._app = SimpleNamespace(det_model=MalformedDetector())
    backend.embedding_contract = FAKE_CONTRACT

    with pytest.raises(ModelUnavailableError, match="malformed output"):
        backend.detect_faces(np.zeros((120, 160, 3), dtype=np.uint8))


def test_arcface_micro_batch_bounds_recogniser_rows() -> None:
    pytest.importorskip("insightface", reason="micro-batch test needs face_align")
    recogniser = _StubRecogniser()
    backend = InsightFaceBackend(
        Settings(embedding_flip_tta=False, arcface_micro_batch_size=2)
    )
    backend._app = SimpleNamespace(models={"recognition": recogniser})
    backend.embedding_contract = FAKE_CONTRACT
    frame = np.zeros((600, 600, 3), dtype=np.uint8)
    faces = []
    for index in range(5):
        offset = float(index * 70)
        face = _landmarked_face()
        face.bbox = face.bbox + np.asarray([offset, 0, offset, 0], dtype=np.float32)
        face.landmarks = face.landmarks + np.asarray([offset, 0], dtype=np.float32)
        faces.append(face)

    embedded = backend.embed_faces(frame, faces)

    assert len(embedded) == 5
    assert recogniser.batch_sizes == [2, 2, 1]


def test_embed_faces_discards_scalar_provider_response() -> None:
    pytest.importorskip("insightface", reason="malformed output test needs face_align")

    class ScalarRecogniser(_StubRecogniser):
        def get_feat(self, crops):
            self.batch_sizes.append(len(crops))
            return np.asarray(0.0, dtype=np.float32)

    recogniser = ScalarRecogniser()
    backend = InsightFaceBackend(Settings(embedding_flip_tta=False))
    backend._app = SimpleNamespace(models={"recognition": recogniser})
    backend.embedding_contract = FAKE_CONTRACT

    embedded = backend.embed_faces(
        np.zeros((400, 400, 3), dtype=np.uint8), [_landmarked_face()]
    )

    assert embedded == []


def test_embed_faces_discards_three_dimensional_provider_response() -> None:
    pytest.importorskip("insightface", reason="malformed output test needs face_align")

    class ThreeDimensionalRecogniser(_StubRecogniser):
        def get_feat(self, crops):
            self.batch_sizes.append(len(crops))
            return np.ones((1, 1, 2), dtype=np.float32)

    recogniser = ThreeDimensionalRecogniser()
    backend = InsightFaceBackend(Settings(embedding_flip_tta=False))
    backend._app = SimpleNamespace(models={"recognition": recogniser})
    backend.embedding_contract = FAKE_CONTRACT

    embedded = backend.embed_faces(
        np.zeros((400, 400, 3), dtype=np.uint8), [_landmarked_face()]
    )

    assert embedded == []
    assert recogniser.batch_sizes == [1]
