from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from person_search.backends import InsightFaceBackend, _resolve_input_size
from person_search.config import Settings
from person_search.domain import FaceObservation


def test_face_detection_uses_insightface_auto_mode_by_default() -> None:
    assert Settings().face_detection_size == 0


def test_insightface_reports_actual_model_session_providers(monkeypatch) -> None:
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


def test_resolve_input_size_maps_one_scale_or_many_onto_scrfd() -> None:
    assert _resolve_input_size(None) is None
    assert _resolve_input_size(0) is None
    assert _resolve_input_size(320) == (320, 320)
    # SCRFD runs every scale it is handed and merges the candidates with NMS, so a
    # list is one multi-scale pass rather than N calls.
    assert _resolve_input_size([128, 640, 1280]) == [(128, 128), (640, 640), (1280, 1280)]
    assert _resolve_input_size([0]) is None


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
    frame = np.zeros((400, 400, 3), dtype=np.uint8)

    embedded = backend.embed_faces(frame, [_landmarked_face()])

    assert recogniser.batch_sizes == [1]
    assert embedded[0].embedding == pytest.approx([1.0, 0.0])
