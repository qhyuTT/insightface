from __future__ import annotations

import gc
import hashlib
import sys
import types
import weakref

import cv2
import numpy as np
import pytest

from person_search.config import Settings
from person_search.domain import EmbeddingContract
from person_search.errors import ModelUnavailableError
from person_search.hybrid_backends import RppScrfdRknnFaceBackend
from person_search.rknn_backends import RknnFaceBackend, RknnYoloXDetector
from person_search.rpp_backends import (
    RppFaceBackend,
    _decode_scrfd,
    _load_rpp_engine,
    _preprocess_scrfd,
    _scrfd_output_shapes,
    _scrfd_static_model_path,
)
from person_search.service import SearchManager, _provider_is_accelerated


def test_hybrid_routes_faces_to_rpp_and_people_to_rknn() -> None:
    manager = SearchManager(Settings(inference_backend="hybrid"))

    assert isinstance(manager.face_backend, RppScrfdRknnFaceBackend)
    assert isinstance(manager.face_backend._recognition_backend, RknnFaceBackend)
    assert isinstance(manager.person_detector, RknnYoloXDetector)


class _FakeRecognitionBackend:
    recognition_provider_name = "RKNN_NPU"
    embedding_contract = EmbeddingContract(
        schema_version="arcface-v1-rknn",
        model_name="buffalo_l-rk3588",
        model_sha256="a" * 64,
        embedding_dimension=512,
        input_size=(112, 112),
        flip_tta=True,
    )

    def __init__(self, provider: str = "RKNN_NPU"):
        self.recognition_provider_name = provider
        self.closed = False

    def ensure_ready(self) -> None:
        return None

    def embed_faces(self, frame, faces):
        return faces

    def close(self) -> None:
        self.closed = True


class _FakeRppEngine:
    def __init__(self, input_shape, output_shapes):
        self.input_shape = input_shape
        self.output_shapes = output_shapes
        self.run_calls = 0
        self.closed = False

    def run(self, input_tensor):
        self.run_calls += 1
        return {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in self.output_shapes.items()
        }

    def close(self) -> None:
        self.closed = True


def test_hybrid_readiness_loads_static_scrfd_on_rpp_and_keeps_rknn_contract(
    tmp_path, monkeypatch
) -> None:
    import person_search.hybrid_backends as hybrid

    settings = Settings(
        inference_backend="hybrid",
        insightface_root=tmp_path,
        rpp_static_model_dir=tmp_path / "static",
        rpp_engine_cache_dir=tmp_path / "plans",
    )
    model_path = tmp_path / "models" / "buffalo_l" / "det_10g.onnx"
    model_path.parent.mkdir(parents=True)
    source_model = b"SCRFD test model"
    model_path.write_bytes(source_model)
    settings.rpp_static_model_dir.mkdir()
    for input_hw in ((736, 1280), (640, 640)):
        artifact = _scrfd_static_model_path(
            source_model, input_hw=input_hw, model_dir=settings.rpp_static_model_dir
        )
        artifact.write_bytes(b"static graph")

    output_names = tuple(f"output_{index}" for index in range(9))
    monkeypatch.setattr(hybrid, "_validate_static_scrfd_model", lambda *_args, **_kwargs: output_names)
    engines = []

    def engine_factory(_model_bytes, *, input_shape, output_shapes, **_kwargs):
        engine = _FakeRppEngine(input_shape, output_shapes)
        engines.append(engine)
        return engine

    recognition_backend = _FakeRecognitionBackend()
    backend = RppScrfdRknnFaceBackend(
        settings,
        recognition_backend=recognition_backend,
        engine_factory=engine_factory,
    )

    backend.ensure_ready()

    assert backend.detection_provider_name == "RPP_BF16"
    assert backend.recognition_provider_name == "RKNN_NPU"
    assert backend.embedding_contract is recognition_backend.embedding_contract
    assert [engine.input_shape for engine in engines] == [
        (3, 736, 1280),
        (3, 640, 640),
    ]
    assert all(engine.run_calls == 1 for engine in engines)


def test_hybrid_readiness_rejects_cpu_recognition_provider() -> None:
    backend = RppScrfdRknnFaceBackend(
        Settings(inference_backend="hybrid"),
        recognition_backend=_FakeRecognitionBackend("CPUExecutionProvider"),
    )

    with pytest.raises(ModelUnavailableError, match="must run on RKNN_NPU"):
        backend.ensure_ready()


def test_rpp_is_scheduled_as_an_accelerated_provider() -> None:
    assert _provider_is_accelerated("RPP_BF16")


def test_cached_rpp_engine_keeps_native_logger_alive(tmp_path, monkeypatch) -> None:
    import person_search.rpp_backends as rpp

    monkeypatch.setenv("PERSON_SEARCH_RPP_ENGINE_CACHE_ENABLED", "true")
    logger_refs = []

    class FakeLogger:
        ERROR = 1

        def __init__(self, _level):
            logger_refs.append(weakref.ref(self))

    class FakeDimensions:
        def __init__(self, values):
            self.values = values

        def get(self):
            return self.values

    class FakeEngine:
        num_bindings = 2

        def __init__(self, runtime):
            self.runtime = runtime

        def createExecutionContext(self):
            assert self.runtime.logger_ref() is not None
            return object()

        def get_binding_name(self, index):
            return ("input", "output")[index]

        def get_binding_shape(self, index):
            return FakeDimensions(((3, 32, 32), (1,))[index])

        def get_binding_dtype(self, _index):
            return "FLOAT"

        def binding_is_input(self, index):
            return index == 0

    class FakeRuntime:
        def __init__(self, logger):
            self.logger_ref = weakref.ref(logger)

        def deserializeCudaEngine(self, plan):
            assert plan == b"cached plan"
            return FakeEngine(self)

    class FakeAllocation:
        def __init__(self, _size):
            pass

        def GetBufferPrt(self):
            return id(self)

    monkeypatch.setitem(
        sys.modules,
        "pyrt",
        types.SimpleNamespace(
            Logger=FakeLogger,
            Runtime=FakeRuntime,
            DeviceAllocation=FakeAllocation,
            HostAllocation=FakeAllocation,
        ),
    )
    monkeypatch.setattr(rpp, "_rpp_sdk_fingerprint", lambda _pyrt: "test-sdk")
    model = b"model"
    key = "scrfd-test"
    hw = (32, 32)
    digest = hashlib.sha256(
        model + f"{key}|{hw}|bf16|test-sdk".encode()
    ).hexdigest()[:20]
    (tmp_path / f"{key}_32x32_{digest}.plan").write_bytes(b"cached plan")

    loaded = _load_rpp_engine(
        model,
        cache_dir=tmp_path,
        model_key=key,
        input_shape=(3, *hw),
        output_shapes={"output": (1,)},
    )
    gc.collect()
    assert logger_refs[0]() is not None
    loaded.close()
    gc.collect()
    assert logger_refs[0]() is None


def test_uncached_rpp_build_does_not_create_an_unused_runtime(tmp_path, monkeypatch) -> None:
    class BuildReached(Exception):
        pass

    class FakeLogger:
        ERROR = 1

        def __init__(self, _level):
            pass

    class FakeBuilder:
        def __init__(self, _logger):
            raise BuildReached

    class ForbiddenRuntime:
        def __init__(self, _logger):
            raise AssertionError("uncached build must not create a Runtime")

    monkeypatch.delenv("PERSON_SEARCH_RPP_ENGINE_CACHE_ENABLED", raising=False)
    monkeypatch.setitem(
        sys.modules,
        "pyrt",
        types.SimpleNamespace(
            Logger=FakeLogger,
            Runtime=ForbiddenRuntime,
            Builder=FakeBuilder,
        ),
    )

    with pytest.raises(BuildReached):
        _load_rpp_engine(
            b"model",
            cache_dir=tmp_path,
            model_key="scrfd-test",
            input_shape=(3, 32, 32),
            output_shapes={"output": (1,)},
        )


def test_rpp_scrfd_output_shapes_preserve_rectangular_grids() -> None:
    assert _scrfd_output_shapes((736, 1280)) == (
        (29440, 1),
        (7360, 1),
        (1840, 1),
        (29440, 4),
        (7360, 4),
        (1840, 4),
        (29440, 10),
        (7360, 10),
        (1840, 10),
    )


def test_rpp_scrfd_preprocessing_matches_insightface_onnx_blob() -> None:
    frame = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)

    actual, ratio = _preprocess_scrfd(frame, (64, 96))

    expected_ratio = min(64 / 7, 96 / 11)
    resized_width = int(11 * expected_ratio)
    resized_height = int(7 * expected_ratio)
    resized = cv2.resize(
        frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    padded = np.zeros((64, 96, 3), dtype=np.uint8)
    padded[:resized_height, :resized_width] = resized
    expected = cv2.dnn.blobFromImage(
        padded,
        1.0 / 128.0,
        (96, 64),
        (127.5, 127.5, 127.5),
        swapRB=True,
    )[0]

    assert ratio == expected_ratio
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("frame_hw,input_hw", [
    ((1080, 1920), (736, 1280)),
    ((720, 1280), (736, 1280)),
    ((83, 171), (736, 1280)),
    ((421, 317), (640, 640)),
])
def test_rpp_scrfd_preprocessing_preserves_rgb_and_letterbox_exactly(
    frame_hw: tuple[int, int], input_hw: tuple[int, int]
) -> None:
    frame = np.random.default_rng(sum(frame_hw)).integers(
        0, 256, (*frame_hw, 3), dtype=np.uint8
    )
    actual, ratio = _preprocess_scrfd(frame, input_hw)
    resized_width = max(1, min(input_hw[1], int(frame_hw[1] * ratio)))
    resized_height = max(1, min(input_hw[0], int(frame_hw[0] * ratio)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    padded = np.zeros((*input_hw, 3), dtype=np.uint8)
    padded[:resized_height, :resized_width] = rgb
    expected = cv2.dnn.blobFromImage(
        padded, 1.0 / 128.0, (input_hw[1], input_hw[0]),
        (127.5, 127.5, 127.5), swapRB=False,
    )[0]
    np.testing.assert_array_equal(actual, expected)


def test_rpp_scrfd_decodes_output_bindings_by_name() -> None:
    input_hw = (64, 64)
    names = (
        "score8",
        "score16",
        "score32",
        "box8",
        "box16",
        "box32",
        "kps8",
        "kps16",
        "kps32",
    )
    outputs = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in zip(names, _scrfd_output_shapes(input_hw), strict=True)
    }
    # Stride-8 grid row 2, column 3, anchor 0: center (24, 16).
    index = (2 * (input_hw[1] // 8) + 3) * 2
    outputs["score8"][index, 0] = 0.95
    outputs["box8"][index] = (2, 2, 2, 2)
    outputs["kps8"][index] = (-1, -1, 1, -1, 0, 0, -1, 1, 1, 1)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    boxes, landmarks, scores = _decode_scrfd(
        frame,
        outputs,
        names,
        input_hw,
        ratio=0.8,
        threshold=0.45,
    )

    assert boxes.shape == (1, 4)
    np.testing.assert_allclose(boxes[0], [10, 0, 50, 40])
    assert landmarks.shape == (1, 5, 2)
    np.testing.assert_allclose(
        landmarks[0],
        [[20, 10], [40, 10], [30, 20], [20, 30], [40, 30]],
    )
    np.testing.assert_allclose(scores, [0.95])


def test_rpp_provider_metadata_is_not_claimed_before_real_readiness() -> None:
    backend = RppFaceBackend(Settings(inference_backend="hybrid"))

    assert backend.provider_name == "uninitialized"
    assert backend.detection_provider_name == "uninitialized"
    assert backend.recognition_provider_name == "uninitialized"
    assert backend.static_input_hw == (736, 1280)
    assert backend.enrollment_static_input_hw == (640, 640)


def _execution_fixture(monkeypatch, *, release_gil=True, enqueue=True, sync_status=0,
                       copy_failure=False, destroy_failure=False):
    import person_search.rpp_backends as rpp

    events = []
    handle = 0xAE5000070000003E

    class Stream:
        def CreateStream(self):
            events.append("create_stream")

        def GetStream(self):
            return handle

        def StreamDestroy(self):
            events.append("destroy_stream")
            if destroy_failure:
                raise RuntimeError("destroy failed")

    class NativeSync:
        def __call__(self, pointer):
            assert rpp._RPP_EXECUTION_LOCK._is_owned()
            assert pointer.value == handle
            events.append("sync")
            if isinstance(sync_status, Exception):
                raise sync_status
            return sync_status

    native_sync = NativeSync()
    library = types.SimpleNamespace(rtStreamSynchronize=native_sync)
    monkeypatch.setattr(rpp.ctypes, "CDLL", lambda name: library)

    class NativeInput:
        def __init__(self, size):
            assert size == 48

        def copy_from_numpy(self, array, device_handle):
            assert rpp._RPP_EXECUTION_LOCK._is_owned()
            assert array.nbytes == 48 and device_handle > 0
            events.append("h2d")
            if copy_failure:
                raise ModelUnavailableError("RPP native input copy failed: 7")

        def clear(self):
            assert rpp._RPP_EXECUTION_LOCK._is_owned()
            events.append("clear_input")

        def close(self):
            events.append("free_input")

    monkeypatch.setattr(rpp, "_RppNativeInput", NativeInput)

    class Context:
        def execute(self, count, bindings):
            assert rpp._RPP_EXECUTION_LOCK._is_owned()
            assert count == 1 and len(bindings) == 2
            events.append("execute")
            return True

        def enqueue(self, count, bindings, stream):
            assert rpp._RPP_EXECUTION_LOCK._is_owned()
            assert count == 1 and len(bindings) == 2 and isinstance(stream, Stream)
            events.append("enqueue")
            if isinstance(enqueue, Exception):
                raise enqueue
            return enqueue

        def __del__(self):
            events.append("destroy_context")

    class Engine:
        num_bindings = 2

        def createExecutionContext(self):
            return Context()

        def get_binding_name(self, index):
            return ("input", "output")[index]

        def get_binding_shape(self, index):
            return types.SimpleNamespace(get=lambda: ((3, 2, 2), (2,))[index])

        def get_binding_dtype(self, index):
            return "FLOAT"

        def binding_is_input(self, index):
            return index == 0

    class Allocation:
        def __init__(self, size):
            self.size = size

        def GetBufferPrt(self):
            return id(self)

        def copy_from_numpy(self, array):
            assert rpp._RPP_EXECUTION_LOCK._is_owned()
            events.append("h2d")

        def copy_to_host(self, host):
            assert rpp._RPP_EXECUTION_LOCK._is_owned()
            events.append("d2h")
            return True

        def numpy_float(self):
            return np.array([0.25, 1.5], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "pyrt", types.SimpleNamespace(
        rtStreamPyExt=Stream, DeviceAllocation=Allocation, HostAllocation=Allocation,
    ))
    monkeypatch.setenv("PERSON_SEARCH_RPP_RELEASE_GIL", str(release_gil).lower())
    engine = rpp._RppExecutionEngine(
        engine=Engine(), runtime=None, input_shape=(3, 2, 2), output_shapes={"output": (2,)},
    )
    return engine, events, native_sync


def test_rpp_async_wait_preserves_output_and_lock_and_cleanup_order(monkeypatch):
    import ctypes

    engine, events, sync = _execution_fixture(monkeypatch)
    assert engine.execution_wait_mode == "native_stream_sync"
    assert sync.argtypes == [ctypes.c_void_p] and sync.restype is ctypes.c_int
    actual = engine.run(np.zeros(engine.input_shape, dtype=np.float32))
    assert events == ["create_stream", "h2d", "enqueue", "sync", "clear_input", "d2h"]
    engine.close()
    assert events[-3:] == ["destroy_context", "destroy_stream", "free_input"]
    engine.close()
    assert events.count("destroy_stream") == 1
    with pytest.raises(ModelUnavailableError, match="closed or failed"):
        engine.run(np.zeros(engine.input_shape, dtype=np.float32))
    baseline, sync_events, _ = _execution_fixture(monkeypatch, release_gil=False)
    assert baseline.execution_wait_mode == "sync_execute"
    expected = baseline.run(np.zeros(baseline.input_shape, dtype=np.float32))
    np.testing.assert_array_equal(actual["output"], expected["output"])
    assert sync_events == ["h2d", "execute", "d2h"]
    baseline.close()


@pytest.mark.parametrize("enqueue,sync_status", [
    (False, 0), (True, 7), (RuntimeError("enqueue exception"), 0),
    (True, RuntimeError("sync exception")),
])
def test_rpp_async_failure_never_reuses_or_falls_back(monkeypatch, enqueue, sync_status):
    import person_search.rpp_backends as rpp

    engine, events, _ = _execution_fixture(monkeypatch, enqueue=enqueue, sync_status=sync_status)
    try:
        with pytest.raises(ModelUnavailableError, match="enqueue failed|synchronization failed|asynchronous execution failed"):
            engine.run(np.zeros(engine.input_shape, dtype=np.float32))
        assert "execute" not in events and "d2h" not in events
        assert engine in rpp._RPP_QUARANTINED_ENGINES
        prior = list(events)
        with pytest.raises(ModelUnavailableError, match="closed or failed"):
            engine.run(np.zeros(engine.input_shape, dtype=np.float32))
        engine.close()
        assert events == prior
        assert engine._context is not None and engine._devices and engine._native_stream
    finally:
        rpp._RPP_QUARANTINED_ENGINES.remove(engine)


def test_rpp_native_stream_rejects_missing_runtime_and_invalid_handle(monkeypatch):
    import person_search.rpp_backends as rpp

    monkeypatch.setattr(rpp.ctypes, "CDLL", lambda name: types.SimpleNamespace())
    with pytest.raises(ModelUnavailableError, match="unsupported"):
        rpp._RppNativeStream(types.SimpleNamespace())
    engine, _, _ = _execution_fixture(monkeypatch)
    engine.close()
    sdk = sys.modules["pyrt"]
    monkeypatch.setattr(sdk.rtStreamPyExt, "GetStream", lambda self: 0)
    with pytest.raises(ModelUnavailableError, match="unsupported"):
        rpp._RppNativeStream(sdk)


def test_rpp_wait_defaults_to_sync_and_opt_in_requires_enqueue(monkeypatch):
    import person_search.rpp_backends as rpp

    seed, _, _ = _execution_fixture(monkeypatch, release_gil=False)
    monkeypatch.delenv("PERSON_SEARCH_RPP_RELEASE_GIL")
    kwargs = {"engine": seed._engine, "runtime": None,
              "input_shape": (3, 2, 2), "output_shapes": {"output": (2,)}}
    default = rpp._RppExecutionEngine(**kwargs)
    assert default.execution_wait_mode == "sync_execute"
    default.close()
    monkeypatch.setenv("PERSON_SEARCH_RPP_RELEASE_GIL", "true")
    monkeypatch.setattr(type(seed._context), "enqueue", None)
    with pytest.raises(ModelUnavailableError, match="requires SDK enqueue"):
        rpp._RppExecutionEngine(**kwargs)
    seed.close()


def test_explicit_release_gil_overrides_the_environment(monkeypatch):
    """Settings, not a bare env var, decides the RPP wait mode."""
    import person_search.rpp_backends as rpp

    seed, _, _ = _execution_fixture(monkeypatch, release_gil=False)
    kwargs = {
        "engine": seed._engine,
        "runtime": None,
        "input_shape": (3, 2, 2),
        "output_shapes": {"output": (2,)},
    }
    monkeypatch.setenv("PERSON_SEARCH_RPP_RELEASE_GIL", "true")
    explicit_off = rpp._RppExecutionEngine(release_gil=False, **kwargs)
    assert explicit_off.execution_wait_mode == "sync_execute"
    explicit_off.close()

    monkeypatch.delenv("PERSON_SEARCH_RPP_RELEASE_GIL", raising=False)
    explicit_on = rpp._RppExecutionEngine(release_gil=True, **kwargs)
    assert explicit_on.execution_wait_mode == "native_stream_sync"
    explicit_on.close()
    seed.close()


def test_hybrid_engine_factory_receives_settings_rpp_flags(tmp_path, monkeypatch):
    """The engine gets the flags from Settings, so the panel cannot claim a mode
    the engine is not using."""
    import person_search.hybrid_backends as hybrid
    from person_search.rpp_backends import _scrfd_static_model_path

    settings = Settings(
        inference_backend="hybrid",
        insightface_root=tmp_path,
        rpp_static_model_dir=tmp_path / "static",
        rpp_engine_cache_dir=tmp_path / "plans",
        rpp_release_gil=True,
        rpp_engine_cache_enabled=True,
    )
    model_path = tmp_path / "models" / "buffalo_l" / "det_10g.onnx"
    model_path.parent.mkdir(parents=True)
    source_model = b"SCRFD test model"
    model_path.write_bytes(source_model)
    settings.rpp_static_model_dir.mkdir()
    for input_hw in ((736, 1280), (640, 640)):
        _scrfd_static_model_path(
            source_model, input_hw=input_hw, model_dir=settings.rpp_static_model_dir
        ).write_bytes(b"static graph")

    output_names = tuple(f"output_{index}" for index in range(9))
    monkeypatch.setattr(
        hybrid, "_validate_static_scrfd_model", lambda *_args, **_kwargs: output_names
    )
    seen = []

    class _ModeEngine(_FakeRppEngine):
        execution_wait_mode = "native_stream_sync"

    def engine_factory(_model_bytes, **kwargs):
        seen.append(kwargs)
        return _ModeEngine(kwargs["input_shape"], kwargs["output_shapes"])

    backend = RppScrfdRknnFaceBackend(
        settings,
        recognition_backend=_FakeRecognitionBackend(),
        engine_factory=engine_factory,
    )
    backend.ensure_ready()

    assert len(seen) == 2, [sorted(item) for item in seen]
    assert all(item["release_gil"] is True for item in seen)
    assert all(item["engine_cache_enabled"] is True for item in seen)
    assert backend.detection_execution_wait_mode == "native_stream_sync"


def test_rpp_native_copy_failure_quarantines_all_resources(monkeypatch):
    import person_search.rpp_backends as rpp

    engine, events, _ = _execution_fixture(monkeypatch, copy_failure=True)
    try:
        with pytest.raises(ModelUnavailableError, match="input copy failed"):
            engine.run(np.zeros(engine.input_shape, dtype=np.float32))
        assert events == ["create_stream", "h2d"]
        assert engine in rpp._RPP_QUARANTINED_ENGINES
        engine.close()
        assert events == ["create_stream", "h2d"]
        assert engine._native_input is not None
        assert engine._context is not None and engine._devices
    finally:
        rpp._RPP_QUARANTINED_ENGINES.remove(engine)


def test_rpp_stream_destroy_failure_retains_input_and_device_memory(monkeypatch):
    import person_search.rpp_backends as rpp

    engine, events, _ = _execution_fixture(monkeypatch, destroy_failure=True)
    try:
        with pytest.raises(RuntimeError, match="destroy failed"):
            engine.close()
        assert engine in rpp._RPP_QUARANTINED_ENGINES
        assert events == ["create_stream", "destroy_context", "destroy_stream"]
        assert engine._native_input is not None and engine._devices
        engine.close()
        assert events.count("destroy_stream") == 1
    finally:
        rpp._RPP_QUARANTINED_ENGINES.remove(engine)
