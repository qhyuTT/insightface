from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from person_search.config import Settings
from person_search.errors import ModelUnavailableError
from person_search.rknn_backend import (
    RknnFaceBackend,
    RknnPersonDetector,
    RknnSession,
    verify_sha256,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.loaded: str | None = None
        self.initialized: dict[str, object] | None = None
        self.released = False
        self.inference_args: tuple[object, ...] = ()
        self.inference_kwargs: dict[str, object] = {}

    def load_rknn(self, path: str) -> int:
        self.loaded = path
        return 0

    def init_runtime(self, **kwargs: object) -> int:
        self.initialized = kwargs
        return 0

    def inference(self, *args: object, **kwargs: object):
        self.inference_args = args
        self.inference_kwargs = kwargs
        return [np.zeros((1, 3549, 85), dtype=np.float32)]

    def release(self) -> None:
        self.released = True


def test_rknn_session_loads_lazily_and_passes_core_mask(tmp_path: Path) -> None:
    model = tmp_path / "model.rknn"
    model.write_bytes(b"model")
    runtime = FakeRuntime()
    session = RknnSession(model, core_mask=7, runtime_factory=lambda: runtime)

    assert not session.ready
    session.ensure_ready()
    assert session.ready
    assert runtime.loaded == str(model)
    assert runtime.initialized == {"core_mask": 7}
    output = session.inference([np.zeros((1, 3, 4, 4), dtype=np.float32)])
    assert output[0].shape == (1, 3549, 85)
    assert "inputs" in runtime.inference_kwargs
    session.release()
    assert runtime.released


def test_rknn_session_reports_missing_model(tmp_path: Path) -> None:
    session = RknnSession(tmp_path / "missing.rknn", runtime_factory=FakeRuntime)
    with pytest.raises(ModelUnavailableError, match="not found"):
        session.ensure_ready()


def test_rknn_session_reports_checksum_mismatch(tmp_path: Path) -> None:
    model = tmp_path / "model.rknn"
    model.write_bytes(b"model")
    with pytest.raises(ModelUnavailableError, match="checksum mismatch"):
        verify_sha256(model, "0" * 64)


def test_rknn_person_detector_uses_runtime_output() -> None:
    class FakeSession:
        def inference(self, inputs, *, data_format=None):
            assert inputs[0].shape == (1, 3, 416, 416)
            assert data_format == ["nchw"]
            return [np.zeros((1, 3549, 85), dtype=np.float32)]

        def ensure_ready(self) -> None:
            pass

        def release(self) -> None:
            pass

    detector = RknnPersonDetector(Settings(), session=FakeSession())  # type: ignore[arg-type]
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


def test_rknn_session_passes_nchw_data_format(tmp_path: Path) -> None:
    model = tmp_path / "model.rknn"
    model.write_bytes(b"model")
    runtime = FakeRuntime()
    session = RknnSession(model, runtime_factory=lambda: runtime)

    session.inference(
        [np.zeros((1, 3, 4, 4), dtype=np.float32)],
        data_format=["nchw"],
    )

    assert runtime.inference_kwargs["data_format"] == ["nchw"]


def test_rknn_session_rejects_mismatched_data_format_count(tmp_path: Path) -> None:
    model = tmp_path / "model.rknn"
    model.write_bytes(b"model")
    session = RknnSession(model, runtime_factory=FakeRuntime)

    with pytest.raises(ModelUnavailableError, match="count must match"):
        session.inference([np.zeros((1, 3, 4, 4))], data_format=[])


def test_rknn_session_does_not_reinitialize_until_release_finishes(tmp_path: Path) -> None:
    class BlockingReleaseRuntime(FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.release_started = threading.Event()
            self.allow_release = threading.Event()

        def release(self) -> None:
            self.release_started.set()
            assert self.allow_release.wait(timeout=1.0)
            super().release()

    model = tmp_path / "model.rknn"
    model.write_bytes(b"model")
    first = BlockingReleaseRuntime()
    runtimes: list[FakeRuntime] = []

    def factory() -> FakeRuntime:
        runtime = first if not runtimes else FakeRuntime()
        runtimes.append(runtime)
        return runtime

    session = RknnSession(model, runtime_factory=factory)
    session.ensure_ready()
    release_thread = threading.Thread(target=session.release)
    release_thread.start()
    assert first.release_started.wait(timeout=1.0)

    ensure_finished = threading.Event()

    def ensure_again() -> None:
        session.ensure_ready()
        ensure_finished.set()

    ensure_thread = threading.Thread(target=ensure_again)
    ensure_thread.start()
    assert not ensure_finished.wait(timeout=0.05)
    assert len(runtimes) == 1

    first.allow_release.set()
    release_thread.join(timeout=1.0)
    ensure_thread.join(timeout=1.0)
    assert ensure_finished.is_set()
    assert len(runtimes) == 2


def test_rknn_person_detector_rejects_integer_outputs() -> None:
    class FakeSession:
        def inference(self, inputs, *, data_format=None):
            return [np.zeros((1, 3549, 85), dtype=np.int8)]

        def ensure_ready(self) -> None:
            pass

        def release(self) -> None:
            pass

    detector = RknnPersonDetector(Settings(), session=FakeSession())  # type: ignore[arg-type]
    with pytest.raises(ModelUnavailableError, match="quantized integer"):
        detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))


class FakeFaceAdapter:
    model_name = "fake-rknn-face"

    def __init__(self) -> None:
        self.ready_calls = 0
        self.release_calls = 0

    def ensure_ready(self) -> None:
        self.ready_calls += 1

    def analyze(self, frame: np.ndarray, *, enrollment: bool = False):
        return []

    def release(self) -> None:
        self.release_calls += 1


def _face_settings(tmp_path: Path, **overrides: object) -> Settings:
    detection = tmp_path / "scrfd.rknn"
    recognition = tmp_path / "arcface.rknn"
    detection.write_bytes(b"detection")
    recognition.write_bytes(b"recognition")
    values: dict[str, object] = {
        "inference_backend": "rknn",
        "rknn_face_detection_model": detection,
        "rknn_face_recognition_model": recognition,
        "rknn_face_detection_sha256": hashlib.sha256(b"detection").hexdigest(),
        "rknn_face_recognition_sha256": hashlib.sha256(b"recognition").hexdigest(),
    }
    values.update(overrides)
    return Settings(**values)


def test_rknn_face_backend_reports_named_missing_paths() -> None:
    backend = RknnFaceBackend(Settings(inference_backend="rknn"), adapter=FakeFaceAdapter())

    with pytest.raises(ModelUnavailableError) as caught:
        backend.ensure_ready()

    assert "face detection model path is not configured" in caught.value.message
    assert "face recognition model path is not configured" in caught.value.message
    assert "missing None" not in caught.value.message


def test_rknn_face_backend_verifies_checksums_before_adapter(tmp_path: Path) -> None:
    adapter = FakeFaceAdapter()
    settings = _face_settings(tmp_path, rknn_face_detection_sha256="0" * 64)
    backend = RknnFaceBackend(settings, adapter=adapter)

    with pytest.raises(ModelUnavailableError, match="checksum mismatch"):
        backend.ensure_ready()

    assert adapter.ready_calls == 0


def test_rknn_face_backend_initializes_once_and_releases(tmp_path: Path) -> None:
    adapter = FakeFaceAdapter()
    backend = RknnFaceBackend(_face_settings(tmp_path), adapter=adapter)

    backend.ensure_ready()
    backend.ensure_ready()
    assert adapter.ready_calls == 1
    assert backend.model_name == "fake-rknn-face"

    backend.release()
    assert adapter.release_calls == 1


def test_rknn_face_backend_serializes_adapter_inference(tmp_path: Path) -> None:
    class ConcurrentAdapter(FakeFaceAdapter):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.max_active = 0
            self.guard = threading.Lock()

        def analyze(self, frame: np.ndarray, *, enrollment: bool = False):
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self.guard:
                self.active -= 1
            return []

    adapter = ConcurrentAdapter()
    backend = RknnFaceBackend(_face_settings(tmp_path), adapter=adapter)
    frame = np.zeros((32, 32, 3), dtype=np.uint8)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(backend.analyze, frame) for _ in range(2)]
        for future in futures:
            assert future.result() == []

    assert adapter.ready_calls == 1
    assert adapter.max_active == 1
