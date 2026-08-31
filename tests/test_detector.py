from __future__ import annotations

import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from person_search.config import Settings
from person_search.detector import (
    YoloXOnnxDetector,
    _decode_yolox,
    _nms,
    _preprocess,
    _yolox_grid,
)
from person_search.errors import ModelUnavailableError


def test_preprocess_letterboxes_to_model_size() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result, ratio = _preprocess(image, (416, 416))
    assert result.shape == (3, 416, 416)
    assert ratio == 2.08
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    # The image is wider than the target aspect ratio, so the bottom rows are
    # letterbox padding in the final CHW model layout.
    assert np.all(result[:, 208:, :] == 114.0)


def test_decode_yolox_builds_all_three_feature_grids() -> None:
    prediction_count = 52 * 52 + 26 * 26 + 13 * 13
    raw = np.zeros((prediction_count, 85), dtype=np.float32)
    decoded = _decode_yolox(raw, (416, 416))
    assert decoded.shape == raw.shape
    np.testing.assert_allclose(decoded[0, :4], [0, 0, 8, 8])


def test_decode_yolox_reuses_cached_grid_for_same_input_size() -> None:
    _yolox_grid.cache_clear()
    prediction_count = 52 * 52 + 26 * 26 + 13 * 13
    raw = np.zeros((prediction_count, 85), dtype=np.float32)
    _decode_yolox(raw, (416, 416))
    first = _yolox_grid.cache_info()
    _decode_yolox(raw, (416, 416))
    second = _yolox_grid.cache_info()
    assert first.misses == 1
    assert second.hits == first.hits + 1


def test_nms_suppresses_overlapping_lower_score_box() -> None:
    boxes = np.asarray([[0, 0, 100, 100], [5, 5, 95, 95], [200, 200, 250, 250]])
    scores = np.asarray([0.9, 0.8, 0.7])
    assert _nms(boxes, scores, 0.5) == [0, 2]


def test_detector_passes_runtime_threads_and_cuda_device_to_onnxruntime(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model = tmp_path / "yolox.onnx"
    model.write_bytes(b"placeholder")

    class FakeSessionOptions:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    class FakeSession:
        requested: ClassVar[dict[str, object]] = {}

        def __init__(self, path: str, **kwargs: object) -> None:
            type(self).requested = {"path": path, **kwargs}

        def get_inputs(self):
            return [SimpleNamespace(name="images")]

        def get_providers(self):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    ort_module = ModuleType("onnxruntime")
    ort_module.SessionOptions = FakeSessionOptions  # type: ignore[attr-defined]
    ort_module.InferenceSession = FakeSession  # type: ignore[attr-defined]
    ort_module.get_available_providers = lambda: [  # type: ignore[attr-defined]
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)
    monkeypatch.setenv("PERSON_SEARCH_ORT_INTRA_OP_NUM_THREADS", "2")
    monkeypatch.setenv("PERSON_SEARCH_ORT_INTER_OP_NUM_THREADS", "1")
    monkeypatch.setenv("PERSON_SEARCH_ORT_CUDA_DEVICE_ID", "3")

    detector = YoloXOnnxDetector(Settings(yolox_model=model, prefer_cuda=True))
    detector.ensure_ready()

    options = FakeSession.requested["sess_options"]
    assert isinstance(options, FakeSessionOptions)
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1
    assert FakeSession.requested["providers"] == [
        ("CUDAExecutionProvider", {"device_id": 3}),
        "CPUExecutionProvider",
    ]


def test_detector_rejects_invalid_runtime_thread_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model = tmp_path / "yolox.onnx"
    model.write_bytes(b"placeholder")
    ort_module = ModuleType("onnxruntime")
    ort_module.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)
    monkeypatch.setenv("PERSON_SEARCH_ORT_INTRA_OP_NUM_THREADS", "many")

    detector = YoloXOnnxDetector(Settings(yolox_model=model, prefer_cuda=False))
    with pytest.raises(ModelUnavailableError, match="PERSON_SEARCH_ORT_INTRA_OP_NUM_THREADS"):
        detector.ensure_ready()


def test_detector_failed_initialization_does_not_publish_and_can_retry(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model = tmp_path / "yolox.onnx"
    model.write_bytes(b"placeholder")

    class FakeSession:
        attempts: ClassVar[int] = 0

        def __init__(self, _path: str, **_kwargs: object) -> None:
            type(self).attempts += 1

        def get_inputs(self):
            if self.attempts == 1:
                raise RuntimeError("invalid input metadata")
            return [SimpleNamespace(name="images")]

        def get_providers(self):
            return ["CPUExecutionProvider"]

    ort_module = ModuleType("onnxruntime")
    ort_module.InferenceSession = FakeSession  # type: ignore[attr-defined]
    ort_module.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)

    detector = YoloXOnnxDetector(Settings(yolox_model=model, prefer_cuda=False))
    with pytest.raises(ModelUnavailableError, match="invalid input metadata"):
        detector.ensure_ready()
    assert detector.provider_name == "uninitialized"
    assert detector._runtime is None

    detector.ensure_ready()

    assert detector.provider_name == "CPUExecutionProvider"
    assert detector._runtime is not None
    assert FakeSession.attempts == 2


def test_detector_concurrent_initialization_publishes_one_complete_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    model = tmp_path / "yolox.onnx"
    model.write_bytes(b"placeholder")
    construction_started = threading.Event()
    allow_construction = threading.Event()

    class FakeSession:
        attempts: ClassVar[int] = 0

        def __init__(self, _path: str, **_kwargs: object) -> None:
            type(self).attempts += 1
            construction_started.set()
            assert allow_construction.wait(timeout=1.0)

        def get_inputs(self):
            return [SimpleNamespace(name="images")]

        def get_providers(self):
            return ["CPUExecutionProvider"]

    ort_module = ModuleType("onnxruntime")
    ort_module.InferenceSession = FakeSession  # type: ignore[attr-defined]
    ort_module.get_available_providers = lambda: ["CPUExecutionProvider"]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)

    detector = YoloXOnnxDetector(Settings(yolox_model=model, prefer_cuda=False))
    workers = [threading.Thread(target=detector.ensure_ready) for _ in range(2)]
    for worker in workers:
        worker.start()
    assert construction_started.wait(timeout=1.0)
    time.sleep(0.05)
    assert detector.provider_name == "uninitialized"
    assert detector._runtime is None

    allow_construction.set()
    for worker in workers:
        worker.join(timeout=1.0)

    assert all(not worker.is_alive() for worker in workers)
    assert FakeSession.attempts == 1
    assert detector.provider_name == "CPUExecutionProvider"
    assert detector._runtime is not None
