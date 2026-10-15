from __future__ import annotations

import ctypes
import io
import struct
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from person_search.config import Settings
from person_search.detector import _preprocess as _preprocess_onnx_yolox
from person_search.domain import SourceConfig, SourceType
from person_search.gstreamer_capture import GStreamerCapture
from person_search.rga import RgaNv12Converter, RgaUnavailableError, _RgaBuffer
from person_search.rknn_backends import (
    RknnFaceBackend,
    RknnYoloXDetector,
    _preprocess_yolox,
    _similarity_transform,
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

    def __init__(self, path: Path, _runtime: Path | None, *, core_mask: int = 0):
        self.path = path
        self.ready = False
        self.core_mask = core_mask
        self.core_mask_applied = core_mask != 0
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
    assert command[command.index("--drop-on-latency") + 1] == "false"
    assert command[command.index("--output-format") + 1] == "bgr"
    assert calls[0]["stdout"] is not None


def test_gstreamer_capture_converts_nv12_frame_from_helper(monkeypatch) -> None:
    class _Process:
        def __init__(self, stream) -> None:
            self.stdout = stream

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    width, height = 2, 2
    payload = bytes([16, 16, 16, 16, 128, 128])
    stream = io.BytesIO(
        struct.pack("!4sIIII", b"PSN1", width, height, 1, len(payload)) + payload
    )
    calls: list[list[str]] = []

    def fake_popen(command, **kwargs):
        calls.append(command)
        return _Process(stream)

    monkeypatch.setattr("person_search.gstreamer_capture.subprocess.Popen", fake_popen)
    capture = GStreamerCapture(
        SourceConfig(type=SourceType.RTSP, uri="rtsp://camera/live"),
        Settings(video_backend="gstreamer", gstreamer_output_format="nv12"),
    )

    ok, frame = capture.read()
    capture.release()

    assert ok
    assert frame is not None
    assert frame.shape == (height, width, 3)
    assert calls[0][calls[0].index("--output-format") + 1] == "nv12"


def test_rga_buffer_layout_matches_librga_2x_abi() -> None:
    assert ctypes.sizeof(_RgaBuffer) == 96


@pytest.mark.parametrize("dark_luma", [False, True])
@pytest.mark.parametrize("uv", [(128, 128), (90, 240), (54, 34), (240, 110), (0, 255)])
def test_rga_preserves_shared_nv12_and_matches_cpu_luma_floor(dark_luma: bool, uv: tuple[int, int]) -> None:
    converter = RgaNv12Converter.__new__(RgaNv12Converter)
    width, height = 256, 256
    nv12 = np.empty((height * 3 // 2, width), dtype=np.uint8)
    luma = np.arange(256, dtype=np.uint8)
    nv12[:height] = (luma if dark_luma else np.clip(luma, 16, 235))[:, None]
    nv12[height:, 0::2] = uv[0]
    nv12[height:, 1::2] = uv[1]
    original = nv12.copy()
    captured = []

    def wrap(address, w, h, ws, hs, fmt):
        if not captured:
            buffer = (ctypes.c_uint8 * (w * h * 3 // 2)).from_address(address.value)
            captured.append(np.ctypeslib.as_array(buffer).copy())
        return _RgaBuffer()

    converter._wrap = wrap
    converter._convert = lambda *args: 1
    converter.convert(nv12, width, height)
    submitted = captured[0].reshape(nv12.shape)
    np.testing.assert_array_equal(nv12, original)
    np.testing.assert_array_equal(submitted[height:], original[height:])
    np.testing.assert_array_equal(submitted[:height], np.maximum(original[:height], 16))
    np.testing.assert_array_equal(
        cv2.cvtColor(original, cv2.COLOR_YUV2BGR_NV12),
        cv2.cvtColor(submitted, cv2.COLOR_YUV2BGR_NV12),
    )


def test_gstreamer_auto_converter_still_rejects_wrong_colors(monkeypatch) -> None:
    capture = GStreamerCapture.__new__(GStreamerCapture)
    capture._rga_converter = SimpleNamespace(
        convert=lambda *args: np.full((2, 2, 3), 20, dtype=np.uint8)
    )
    monkeypatch.setattr(
        "person_search.gstreamer_capture.cv2.cvtColor",
        lambda *args: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    with pytest.raises(RgaUnavailableError, match="color output disagrees"):
        capture._select_fastest_converter(np.zeros((3, 2), dtype=np.uint8), 2, 2)


def test_gstreamer_auto_converter_keeps_faster_verified_rga(monkeypatch) -> None:
    capture = GStreamerCapture.__new__(GStreamerCapture)
    capture._rga_converter = SimpleNamespace(
        convert=lambda nv12, width, height: np.zeros((height, width, 3), dtype=np.uint8)
    )
    capture._converter_mode = "auto"
    capture._conversion_provider = "auto_pending"
    capture._conversion_fallback_reason = None
    monkeypatch.setattr(
        "person_search.gstreamer_capture.cv2.cvtColor",
        lambda image, code: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    clock = iter((0.0, 0.005, 0.005, 0.025))
    monkeypatch.setattr("person_search.gstreamer_capture.time.perf_counter", lambda: next(clock))

    frame = capture._select_fastest_converter(np.zeros((3, 2), dtype=np.uint8), 2, 2)

    assert frame.shape == (2, 2, 3)
    assert capture.conversion_provider == "rga"
    assert capture._rga_converter is not None


def test_gstreamer_auto_converter_retains_faster_opencv(monkeypatch) -> None:
    capture = GStreamerCapture.__new__(GStreamerCapture)
    capture._rga_converter = SimpleNamespace(
        convert=lambda nv12, width, height: np.zeros((height, width, 3), dtype=np.uint8)
    )
    capture._converter_mode = "auto"
    capture._conversion_provider = "auto_pending"
    capture._conversion_fallback_reason = None
    monkeypatch.setattr(
        "person_search.gstreamer_capture.cv2.cvtColor",
        lambda image, code: np.zeros((2, 2, 3), dtype=np.uint8),
    )
    clock = iter((0.0, 0.020, 0.020, 0.025))
    monkeypatch.setattr("person_search.gstreamer_capture.time.perf_counter", lambda: next(clock))

    capture._select_fastest_converter(np.zeros((3, 2), dtype=np.uint8), 2, 2)

    assert capture.conversion_provider == "opencv_selected"
    assert capture._rga_converter is None
    assert "OpenCV benchmark was faster" in str(capture.conversion_fallback_reason)


def test_rknn_backends_pass_core_mask_and_report_static_input(tmp_path: Path) -> None:
    """The panel must show the shape the static graph runs at, and whether the
    three-core mask took; both are otherwise invisible from Settings alone."""
    for name in ("yolox.rknn", "scrfd.rknn", "arcface.rknn"):
        (tmp_path / name).write_bytes(b"x")
    settings = Settings(
        inference_backend="rknn",
        rknn_core_mask=7,
        rknn_yolox_model=tmp_path / "yolox.rknn",
        rknn_face_detection_model=tmp_path / "scrfd.rknn",
        rknn_face_recognition_model=tmp_path / "arcface.rknn",
    )
    detector = RknnYoloXDetector(settings, model_factory=_FakeModel)
    backend = RknnFaceBackend(settings, model_factory=_FakeModel)
    assert detector.static_input_hw is None
    assert detector.core_mask_applied is None

    detector.ensure_ready()
    backend.ensure_ready()

    assert detector._model is not None and detector._model.core_mask == 7
    assert backend._detector is not None and backend._detector.core_mask == 7
    assert backend._recognizer is not None and backend._recognizer.core_mask == 7
    assert detector.static_input_hw == (416, 416)
    assert backend.static_input_hw == (640, 640)
    assert detector.core_mask_applied is True
    assert backend.core_mask_applied is True


class _CoreMaskLibrary:
    """Minimal fake runtime: one 2x2x3 uint8 input, one 2-float output."""

    def __init__(self, *, with_core_mask: bool, core_mask_result: int = 0):
        self.core_mask_calls: list[int] = []
        self.rknn_init = _CoreMaskLibrary._function(self.init)
        self.rknn_destroy = _CoreMaskLibrary._function(lambda _context: 0)
        self.rknn_query = _CoreMaskLibrary._function(self.query)
        self.rknn_inputs_set = _CoreMaskLibrary._function(lambda *_args: 0)
        self.rknn_run = _CoreMaskLibrary._function(lambda *_args: 0)
        self.rknn_outputs_get = _CoreMaskLibrary._function(lambda *_args: 0)
        self.rknn_outputs_release = _CoreMaskLibrary._function(lambda *_args: 0)
        if with_core_mask:
            self._core_mask_result = core_mask_result
            self.rknn_set_core_mask = _CoreMaskLibrary._function(self.set_core_mask)

    @staticmethod
    def _function(callback):
        class _Function:
            restype = None
            argtypes = None

            def __call__(self, *args):
                return callback(*args)

        return _Function()

    def set_core_mask(self, _context, mask):
        self.core_mask_calls.append(int(getattr(mask, "value", mask)))
        return self._core_mask_result

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
        if command == RKNN_QUERY_INPUT_ATTR:
            attr.n_dims = 4
            attr.dims[:4] = (1, 2, 2, 3)
            attr.fmt = RKNN_TENSOR_NHWC
            attr.type = RKNN_TENSOR_UINT8
            return 0
        if command == RKNN_QUERY_OUTPUT_ATTR and int(attr.index) == 0:
            attr.n_dims = 2
            attr.dims[:2] = (1, 2)
            attr.fmt = RKNN_TENSOR_NHWC
            attr.type = RKNN_TENSOR_FLOAT32
            return 0
        return -1


@pytest.mark.parametrize(
    ("with_core_mask", "core_mask_result", "expected_applied", "expected_calls"),
    [
        (True, 0, True, [7]),
        (True, -1, False, [7]),
        (False, 0, False, []),
    ],
)
def test_rknn_runtime_core_mask_is_best_effort(
    monkeypatch, tmp_path: Path, with_core_mask, core_mask_result, expected_applied, expected_calls
) -> None:
    """A missing symbol or a rejected mask keeps the vendor default and stays ready."""
    library = _CoreMaskLibrary(with_core_mask=with_core_mask, core_mask_result=core_mask_result)
    monkeypatch.setattr("person_search.rknn_runtime.ctypes.CDLL", lambda _path: library)
    runtime_path = tmp_path / "librknnrt.so"
    runtime_path.write_bytes(b"fake")
    model_path = tmp_path / "model.rknn"
    model_path.write_bytes(b"model")

    model = RknnModel(model_path, runtime_path, core_mask=7)
    model.ensure_ready()

    assert model.is_ready
    assert model.core_mask_applied is expected_applied
    assert library.core_mask_calls == expected_calls
    model.close()


def test_rknn_runtime_auto_core_mask_never_calls_the_runtime(monkeypatch, tmp_path: Path) -> None:
    library = _CoreMaskLibrary(with_core_mask=True)
    monkeypatch.setattr("person_search.rknn_runtime.ctypes.CDLL", lambda _path: library)
    (tmp_path / "librknnrt.so").write_bytes(b"fake")
    (tmp_path / "model.rknn").write_bytes(b"model")

    model = RknnModel(tmp_path / "model.rknn", tmp_path / "librknnrt.so")
    model.ensure_ready()

    assert library.core_mask_calls == []
    assert model.core_mask_applied is False
    model.close()


def _nv12_stream(frames: int, *, width: int = 2, height: int = 2) -> io.BytesIO:
    payload = bytes([16, 16, 16, 16, 128, 128])
    packet = struct.pack("!4sIIII", b"PSN1", width, height, 1, len(payload)) + payload
    return io.BytesIO(packet * frames)


class _HelperProcess:
    def __init__(self, stdout: io.BytesIO, stderr: io.BytesIO | None = None, returncode=None):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode


def test_gstreamer_capture_grab_drains_a_frame_without_converting_it(monkeypatch) -> None:
    """A frame destined for eviction must not pay the NV12->BGR conversion."""
    conversions: list[tuple[int, ...]] = []
    real_cvt = GStreamerCapture.read.__globals__["cv2"].cvtColor

    def counting_cvt(image, code):
        conversions.append(tuple(image.shape))
        return real_cvt(image, code)

    monkeypatch.setattr("person_search.gstreamer_capture.cv2.cvtColor", counting_cvt)
    stream = _nv12_stream(2)
    monkeypatch.setattr(
        "person_search.gstreamer_capture.subprocess.Popen",
        lambda command, **kwargs: _HelperProcess(stream),
    )
    capture = GStreamerCapture(
        SourceConfig(type=SourceType.RTSP, uri="rtsp://camera/live"),
        Settings(video_backend="gstreamer", gstreamer_output_format="nv12"),
    )

    assert capture.grab() is True
    assert conversions == []
    ok, frame = capture.read()
    assert ok and frame is not None and frame.shape == (2, 2, 3)
    assert len(conversions) == 1
    # The stream is exhausted: a third packet cannot be produced.
    assert capture.grab() is False
    capture.release()


def test_gstreamer_capture_logs_redacted_helper_stderr_when_it_dies(monkeypatch, caplog) -> None:
    """The reconnect loop's only explanation is the helper's stderr tail, and it
    must never carry the camera credentials into the service log."""
    stderr = io.BytesIO(
        b"0:00:03 WARN rtspsrc Could not open rtsp://admin:secret-pass@192.168.11.3:554/x\n"
        b"GStreamer pipeline read failed\n"
    )
    process = _HelperProcess(io.BytesIO(b""), stderr, returncode=3)
    monkeypatch.setattr(
        "person_search.gstreamer_capture.subprocess.Popen",
        lambda command, **kwargs: process,
    )
    capture = GStreamerCapture(
        SourceConfig(type=SourceType.RTSP, uri="rtsp://admin:secret-pass@192.168.11.3:554/x"),
        Settings(video_backend="gstreamer", gstreamer_output_format="nv12"),
    )
    process.returncode = 3

    with caplog.at_level("WARNING", logger="person_search.gstreamer_capture"):
        ok, frame = capture.read()
        capture.release()

    assert (ok, frame) == (False, None)
    assert "secret-pass" not in caplog.text
    assert "rtsp://admin:***@192.168.11.3:554/x" in caplog.text
    assert "GStreamer pipeline read failed" in caplog.text
    assert "rc=3" in caplog.text


def test_arcface_alignment_recovers_a_known_similarity_transform() -> None:
    """Five landmarks moved by a known scale/rotation/shift must map back exactly."""
    dst = np.asarray(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32,
    )
    angle, scale, shift = np.deg2rad(17.0), 3.4, np.asarray([412.0, 133.0])
    inverse = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) / scale
    src = (dst - shift) @ inverse.T

    transform = _similarity_transform(src, dst)

    assert transform is not None and transform.shape == (2, 3)
    mapped = src @ transform[:, :2].T + transform[:, 2]
    np.testing.assert_allclose(mapped, dst, atol=1e-3)
    assert np.isclose(np.hypot(*transform[:, 0]), scale, rtol=1e-4)


def test_arcface_alignment_matches_insightface_reference_on_noisy_landmarks() -> None:
    """The RKNN crop must be the crop the network was trained on, not an LMEDS variant."""
    trans = pytest.importorskip("skimage.transform")
    rng = np.random.default_rng(7)
    dst = np.asarray(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32,
    )
    for _ in range(20):
        src = dst * 0.6 + rng.normal(0.0, 1.5, size=dst.shape) + 100.0
        reference = trans.SimilarityTransform()
        assert reference.estimate(src, dst)
        ours = _similarity_transform(src, dst)
        assert ours is not None
        np.testing.assert_allclose(ours, reference.params[:2, :], atol=1e-4)


def test_arcface_alignment_rejects_degenerate_landmarks() -> None:
    same_point = np.full((5, 2), 10.0, dtype=np.float32)
    dst = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]], dtype=np.float32)
    assert _similarity_transform(same_point, dst) is None


class _RectFakeModel(_FakeModel):
    """Rectangular static graphs: SCRFD 1280x736 and YOLOX 640x384."""

    def __init__(self, path: Path, _runtime: Path | None, *, core_mask: int = 0):
        super().__init__(path, _runtime, core_mask=core_mask)
        if "yolox" in path.name:
            self.input_attrs = (_info(0, (1, 384, 640, 3)),)
            self.output_attrs = (_info(0, (1, 80 * 48 + 40 * 24 + 20 * 12, 85)),)
        elif "scrfd" in path.name:
            self.input_attrs = (_info(0, (1, 736, 1280, 3)),)
            grids = [(736 // s) * (1280 // s) * 2 for s in (8, 16, 32)]
            self.output_attrs = tuple(
                _info(i, (1, grids[i % 3], (1, 4, 10)[i // 3])) for i in range(9)
            )

    def run(self, data: np.ndarray) -> list[np.ndarray]:
        self.calls.append(np.asarray(data))
        if "yolox" in self.path.name:
            count = 80 * 48 + 40 * 24 + 20 * 12
            output = np.zeros((1, count, 85), dtype=np.float32)
            # Stride-8 cell (10, 70): centre (564, 84) in a 640x384 input.
            index = 10 * 80 + 70
            output[0, index, :6] = (0.5, 0.5, np.log(40 / 8), np.log(80 / 8), 0.9, 0.9)
            return [output]
        if "scrfd" in self.path.name:
            grids = [(736 // s) * (1280 // s) * 2 for s in (8, 16, 32)]
            scores = [np.zeros((1, g, 1), dtype=np.float32) for g in grids]
            boxes = [np.zeros((1, g, 4), dtype=np.float32) for g in grids]
            kps = [np.zeros((1, g, 10), dtype=np.float32) for g in grids]
            # Stride-8 cell (row 30, col 140), anchor 0: centre (1120, 240).
            index = (30 * 160 + 140) * 2
            scores[0][0, index, 0] = 0.95
            boxes[0][0, index] = (4, 4, 4, 4)
            kps[0][0, index] = (-1, -1, 1, -1, 0, 0, -1, 1, 1, 1)
            return scores + boxes + kps
        return [np.ones((1, 512), dtype=np.float32)]


def test_rknn_rectangular_static_inputs_decode_with_correct_geometry(tmp_path: Path) -> None:
    """A 16:9 static graph must letterbox a 16:9 frame edge to edge and map boxes back."""
    for name in ("yolox.rknn", "scrfd.rknn", "arcface.rknn"):
        (tmp_path / name).write_bytes(b"x")
    settings = Settings(
        inference_backend="rknn",
        person_input_width=640,
        person_input_height=384,
        rknn_yolox_model=tmp_path / "yolox.rknn",
        rknn_face_detection_model=tmp_path / "scrfd.rknn",
        rknn_face_recognition_model=tmp_path / "arcface.rknn",
    )
    frame = np.zeros((2160, 3840, 3), dtype=np.uint8)

    detector = RknnYoloXDetector(settings, model_factory=_RectFakeModel)
    persons = detector.detect(frame)
    assert detector.static_input_hw == (384, 640)
    assert detector._model is not None
    assert detector._model.calls[0].shape == (384, 640, 3)
    assert len(persons) == 1
    # ratio = 640/3840 = 1/6; centre (564, 84) -> (3384, 504); size 40x80 -> 240x480.
    np.testing.assert_allclose(persons[0].bbox, [3264, 264, 3504, 744], atol=1.0)

    backend = RknnFaceBackend(settings, model_factory=_RectFakeModel)
    faces = backend.detect_faces(frame)
    assert backend.static_input_hw == (736, 1280)
    assert backend._detector is not None
    assert backend._detector.calls[0].shape == (736, 1280, 3)
    assert len(faces) == 1
    # ratio = min(736/2160, 1280/3840) = 1/3; centre (1120, 240) +-32 -> (3264..3456, 624..816).
    np.testing.assert_allclose(faces[0].bbox, [3264, 624, 3456, 816], atol=1.0)
    assert faces[0].landmarks is not None
    np.testing.assert_allclose(faces[0].landmarks[2], [3360, 720], atol=1.0)


def test_rknn_yolox_rejects_a_model_whose_shape_disagrees_with_settings(tmp_path: Path) -> None:
    """The env must name the shape the .rknn was built with, or readiness fails loudly."""
    for name in ("yolox.rknn",):
        (tmp_path / name).write_bytes(b"x")
    settings = Settings(
        inference_backend="rknn",
        person_input_width=416,
        person_input_height=416,
        rknn_yolox_model=tmp_path / "yolox.rknn",
    )
    detector = RknnYoloXDetector(settings, model_factory=_RectFakeModel)
    with pytest.raises(Exception, match="expected configured"):
        detector.ensure_ready()


def test_rknn_enrollment_uses_its_own_detector_when_configured(tmp_path: Path) -> None:
    """A close-up enrollment photo must not be judged by the far-scene search graph."""
    for name in ("scrfd_search.rknn", "scrfd_enroll.rknn", "arcface.rknn"):
        (tmp_path / name).write_bytes(b"x")
    settings = Settings(
        inference_backend="rknn",
        rknn_face_detection_model=tmp_path / "scrfd_search.rknn",
        rknn_enrollment_face_detection_model=tmp_path / "scrfd_enroll.rknn",
        rknn_face_recognition_model=tmp_path / "arcface.rknn",
    )
    backend = RknnFaceBackend(settings, model_factory=_FakeModel)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    backend.detect_faces(frame, enrollment=True)
    backend.detect_faces(frame, enrollment=False)

    assert backend._enrollment_detector is not None and backend._detector is not None
    assert len(backend._enrollment_detector.calls) == 1
    assert len(backend._detector.calls) == 1
    assert backend.enrollment_static_input_hw == (640, 640)
    backend.close()
    assert backend._enrollment_detector is None


def test_rknn_enrollment_detector_defaults_to_the_search_graph(tmp_path: Path) -> None:
    for name in ("scrfd.rknn", "arcface.rknn"):
        (tmp_path / name).write_bytes(b"x")
    settings = Settings(
        inference_backend="rknn",
        rknn_face_detection_model=tmp_path / "scrfd.rknn",
        rknn_face_recognition_model=tmp_path / "arcface.rknn",
    )
    backend = RknnFaceBackend(settings, model_factory=_FakeModel)
    backend.detect_faces(np.zeros((720, 1280, 3), dtype=np.uint8), enrollment=True)
    assert backend._enrollment_detector is None
    assert backend.enrollment_static_input_hw is None
    assert backend._detector is not None and len(backend._detector.calls) == 1
