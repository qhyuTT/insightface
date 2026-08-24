from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import ClassVar

from person_search.backends import InsightFaceBackend
from person_search.config import Settings


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
