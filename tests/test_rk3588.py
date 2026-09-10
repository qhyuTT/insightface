from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from person_search.config import Settings
from person_search.detector import _preprocess as _preprocess_onnx_yolox
from person_search.domain import SourceConfig, SourceType
from person_search.gstreamer_capture import GStreamerCapture
from person_search.rknn_backends import (
    RknnFaceBackend,
    RknnYoloXDetector,
    _preprocess_yolox,
)
from person_search.rknn_runtime import (
    RKNN_QUERY_IN_OUT_NUM,
    RKNN_QUERY_INPUT_ATTR,
    RKNN_QUERY_OUTPUT_ATTR,
    RKNN_TENSOR_FLOAT32,
    RKNN_TENSOR_NHWC,
    RKNN_TENSOR_UINT8,
    RknnModel,
    TensorInfo,
    _IoNum,
    _RknnInput,
    _RknnOutput,
    _TensorAttr,
)
from person_search.service import SearchManager, _provider_is_accelerated


def _info(index: int, shape: tuple[int, ...], *, fmt: int = RKNN_TENSOR_NHWC) -> TensorInfo:
    return TensorInfo(
        index=index,
        shape=shape,
        name=f"tensor_{index}",
        size=int(np.prod(shape)),
        tensor_format=fmt,
        tensor_type=RKNN_TENSOR_UINT8,
        element_count=int(np.prod(shape)),
        scale=1.0,
        zero_point=0,
    )


class _FakeModel:
    provider_name = "RKNN_NPU"

    def __init__(self, path: Path, _runtime: Path | None):
        self.path = path
        self.ready = False
        self.calls: list[np.ndarray] = []
        if "yolox" in path.name:
            self.input_attrs = (_info(0, (1, 416, 416, 3)),)
            self.output_attrs = (_info(0, (1, 3549, 85)),)
        elif "scrfd" in path.name:
            self.input_attrs = (_info(0, (1, 640, 640, 3)),)
            self.output_attrs = (
                _info(0, (1, 12800, 1)),
                _info(1, (1, 3200, 1)),
                _info(2, (1, 800, 1)),
                _info(3, (1, 12800, 4)),
                _info(4, (1, 3200, 4)),
                _info(5, (1, 800, 4)),
                _info(6, (1, 12800, 10)),
                _info(7, (1, 3200, 10)),
                _info(8, (1, 800, 10)),
            )
        else:
            self.input_attrs = (_info(0, (1, 112, 112, 3)),)
            self.output_attrs = (_info(0, (1, 512)),)

    @property
    def is_ready(self) -> bool:
        return self.ready

    def ensure_ready(self) -> None:
        self.ready = True

    def run(self, data: np.ndarray) -> list[np.ndarray]:
        self.calls.append(np.asarray(data))
        if "yolox" in self.path.name:
            output = np.zeros((1, 3549, 85), dtype=np.float32)
            output[0, 0, :6] = (1.25, 1.25, np.log(20 / 8), np.log(20 / 8), 0.9, 0.9)
            return [output]
        if "scrfd" in self.path.name:
            scores = [
                np.zeros((1, 12800, 1), dtype=np.float32),
                np.zeros((1, 3200, 1), dtype=np.float32),
                np.zeros((1, 800, 1), dtype=np.float32),
            ]
            # Grid cell (20, 20), anchor 0: center (160, 160) at stride 8.
            index = (20 * 80 + 20) * 2
            scores[0][0, index, 0] = 0.99
            boxes = [
                np.zeros((1, 12800, 4), dtype=np.float32),
                np.zeros((1, 3200, 4), dtype=np.float32),
                np.zeros((1, 800, 4), dtype=np.float32),
            ]
            boxes[0][0, index] = (5, 5, 5, 5)
            keypoints = [
                np.zeros((1, 12800, 10), dtype=np.float32),
                np.zeros((1, 3200, 10), dtype=np.float32),
                np.zeros((1, 800, 10), dtype=np.float32),
            ]
            keypoints[0][0, index] = (1, 1, 7, 1, 4, 4, 2, 7, 6, 7)
            return scores + boxes + keypoints
        return [np.ones((1, 512), dtype=np.float32)]

    def close(self) -> None:
        self.ready = False


def test_rknn_abi_structures_match_the_vendor_header() -> None:
    assert ctypes.sizeof(_TensorAttr) == 376
    assert ctypes.sizeof(_IoNum) == 8
    assert ctypes.sizeof(_RknnInput) == 32
    assert ctypes.sizeof(_RknnOutput) == 24


def test_rknn_input_layout_is_adapted_to_declared_nhwc() -> None:
    info = _info(0, (1, 112, 112, 3))
    source = np.zeros((3, 112, 112), dtype=np.uint8)

    prepared, tensor_type, tensor_format = RknnModel._prepare_input(source, info)

    assert prepared.shape == (112, 112, 3)
    assert tensor_type == RKNN_TENSOR_UINT8
    assert tensor_format == RKNN_TENSOR_NHWC


def test_rknn_runtime_queries_abi_and_copies_released_output(monkeypatch, tmp_path: Path) -> None:
    class _Function:
        def __init__(self, callback):
            self.callback = callback
            self.restype = None
            self.argtypes = None

        def __call__(self, *args):
            return self.callback(*args)

    class _Library:
        def __init__(self):
            self.output_buffer = None
            self.rknn_init = _Function(self.init)
            self.rknn_destroy = _Function(lambda _context: 0)
            self.rknn_query = _Function(self.query)
            self.rknn_inputs_set = _Function(lambda *_args: 0)
            self.rknn_run = _Function(lambda *_args: 0)
            self.rknn_outputs_get = _Function(self.outputs_get)
            self.rknn_outputs_release = _Function(lambda *_args: 0)

        @staticmethod
        def init(context_ptr, *_args):
            ctypes.cast(context_ptr, ctypes.POINTER(ctypes.c_uint64))[0] = 7
            return 0

        @staticmethod
        def query(_context, command, attr_ptr, _size):
            if command == RKNN_QUERY_IN_OUT_NUM:
                counts = ctypes.cast(attr_ptr, ctypes.POINTER(_IoNum)).contents
                counts.n_inputs = 1
                counts.n_outputs = 1
                return 0
            attr = ctypes.cast(attr_ptr, ctypes.POINTER(_TensorAttr)).contents
            index = int(attr.index)
            if command == RKNN_QUERY_INPUT_ATTR:
                attr.n_dims = 4
                attr.dims[:4] = (1, 2, 2, 3)
                attr.name = b"input"
                attr.n_elems = 12
                attr.size = 12
                attr.fmt = RKNN_TENSOR_NHWC
                attr.type = RKNN_TENSOR_UINT8
            elif command == RKNN_QUERY_OUTPUT_ATTR and index == 0:
                attr.n_dims = 2
                attr.dims[:2] = (1, 2)
                attr.name = b"output"
                attr.n_elems = 2
                attr.size = 8
                attr.fmt = RKNN_TENSOR_NHWC
                attr.type = RKNN_TENSOR_FLOAT32
            else:
                return -1
            return 0

        def outputs_get(self, _context, _count, output_ptr, _extend):
            self.output_buffer = ctypes.create_string_buffer(
                np.asarray([1.0, 2.0], dtype=np.float32).tobytes()
            )
            outputs = ctypes.cast(output_ptr, ctypes.POINTER(_RknnOutput))
            outputs[0].buf = ctypes.cast(self.output_buffer, ctypes.c_void_p)
            outputs[0].size = 8
            return 0

    library = _Library()
    monkeypatch.setattr("person_search.rknn_runtime.ctypes.CDLL", lambda _path: library)
    runtime_path = tmp_path / "librknnrt.so"
    runtime_path.write_bytes(b"fake")
    model_path = tmp_path / "model.rknn"
    model_path.write_bytes(b"model")
    model = RknnModel(model_path, runtime_path)

    outputs = model.run(np.zeros((2, 2, 3), dtype=np.uint8))
    model.close()

    assert model.input_attrs == ()
    assert outputs[0].shape == (1, 2)
    assert outputs[0].tolist() == [[1.0, 2.0]]


def test_rknn_yolox_detector_decodes_person_head(tmp_path: Path) -> None:
    model_path = tmp_path / "yolox_tiny.rknn"
    model_path.write_bytes(b"fake")
    settings = Settings(
        inference_backend="rknn",
        rknn_yolox_model=model_path,
        person_input_width=416,
        person_input_height=416,
    )
    detector = RknnYoloXDetector(settings, model_factory=_FakeModel)

    detections = detector.detect(np.zeros((416, 416, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.81)
    assert np.allclose(detections[0].bbox, [0.0, 0.0, 20.0, 20.0], atol=1e-5)


def test_rknn_yolox_preprocessing_matches_the_existing_onnx_bgr_path() -> None:
    frame = np.arange(7 * 11 * 3, dtype=np.uint8).reshape(7, 11, 3)

    onnx_image, onnx_ratio = _preprocess_onnx_yolox(frame, (416, 416))
    rknn_image, rknn_ratio = _preprocess_yolox(frame, (416, 416))

    assert rknn_ratio == onnx_ratio
    assert np.array_equal(rknn_image.transpose(2, 0, 1), onnx_image)


def test_rknn_face_backend_decodes_scrfd_and_arcface(tmp_path: Path) -> None:
    detector_path = tmp_path / "scrfd_10g_640.rknn"
    recognizer_path = tmp_path / "w600k_r50.rknn"
    detector_path.write_bytes(b"detector")
    recognizer_path.write_bytes(b"recognizer")
    settings = Settings(
        inference_backend="rknn",
        rknn_face_detection_model=detector_path,
        rknn_face_recognition_model=recognizer_path,
        embedding_flip_tta=True,
    )
    backend = RknnFaceBackend(settings, model_factory=_FakeModel)
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    faces = backend.detect_faces(frame)
    embedded = backend.embed_faces(frame, faces)

    assert len(faces) == 1
    assert faces[0].bbox.tolist() == [120.0, 120.0, 200.0, 200.0]
    assert len(embedded) == 1
    assert embedded[0].embedding is not None
    assert embedded[0].embedding.shape == (512,)
    assert np.isclose(np.linalg.norm(embedded[0].embedding), 1.0)
    assert backend.embedding_contract is not None
    assert backend.embedding_contract.embedding_dimension == 512


def test_search_manager_selects_rknn_only_when_explicit() -> None:
    settings = Settings(inference_backend="rknn")
    manager = SearchManager(settings)

    assert isinstance(manager.face_backend, RknnFaceBackend)
    assert isinstance(manager.person_detector, RknnYoloXDetector)


def test_rknn_provider_uses_accelerated_sampling_budget() -> None:
    assert _provider_is_accelerated("RKNN_NPU")
    assert _provider_is_accelerated("CUDAExecutionProvider")
    assert not _provider_is_accelerated("CPUExecutionProvider")


def test_gstreamer_capture_builds_a_non_shell_helper_command(monkeypatch) -> None:
    class _Process:
        stdout = None

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    calls: list[dict[str, object]] = []

    def fake_popen(command, **kwargs):
        calls.append({"command": command, **kwargs})
        return _Process()

    monkeypatch.setattr("person_search.gstreamer_capture.subprocess.Popen", fake_popen)
    source = SourceConfig(type=SourceType.RTSP, uri="rtsp://camera/live")
    settings = Settings(
        video_backend="gstreamer",
        gstreamer_python=Path("/usr/bin/python3"),
        gstreamer_helper=Path("scripts/rk3588_gstreamer_capture.py"),
    )

    capture = GStreamerCapture(source, settings)
    capture.release()

    assert len(calls) == 1
    command = calls[0]["command"]
    assert isinstance(command, list)
    assert command[0] == "/usr/bin/python3"
    assert Path(command[1]).name == "rk3588_gstreamer_capture.py"
    assert "rtsp://camera/live" in command
    assert calls[0]["stdout"] is not None
