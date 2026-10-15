"""RPP engine and SCRFD graph utilities for the RK3588 hybrid backend.

The serving hybrid route lives in :mod:`person_search.hybrid_backends`: it reads
pre-specialized SCRFD graphs and runs them on RPP, while ArcFace and YOLOX use
RKNN NPU. ``_static_scrfd_model`` is an offline preparation helper; service
readiness does not invoke its CPU graph optimizer. No model inference falls back
to CPU when an accelerator is unavailable.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import Settings
from .detector import _nms
from .domain import EmbeddingContract, FaceObservation
from .errors import ModelUnavailableError
from .model_assets import EMBEDDING_MODEL_MANIFESTS, sha256
from .quality import assess_face, normalize_embedding

logger = logging.getLogger(__name__)

_SCRFD_STRIDES = (8, 16, 32)
_SCRFD_ANCHORS = 2
_SCRFD_STATIC_GRAPH_REVISION = "r1"
_RPP_EXECUTION_LOCK = threading.RLock()
# An unsuccessful enqueue/wait can leave native work referencing its buffers.
# Retain those owners until process exit rather than releasing or reusing them.
_RPP_QUARANTINED_ENGINES: list[object] = []


def _env_flag(name: str) -> bool:
    """Fallback for callers without Settings; Settings stays the source of truth."""
    return os.environ.get(name, "false").strip().lower() in {"1", "true", "yes"}


class _RppNativeStream:
    """Wait on the SDK's stream through CDLL, which releases the Python GIL."""

    def __init__(self, pyrt: object):
        try:
            self._library = ctypes.CDLL("liburpp.so.2")
            self._sync = self._library.rtStreamSynchronize
            self._sync.argtypes = [ctypes.c_void_p]
            self._sync.restype = ctypes.c_int
            self.stream = pyrt.rtStreamPyExt()
            self.stream.CreateStream()
            self.handle = int(self.stream.GetStream())
            if not 0 < self.handle < 1 << (ctypes.sizeof(ctypes.c_void_p) * 8):
                raise ValueError("invalid native stream handle")
        except Exception as exc:
            stream = getattr(self, "stream", None)
            if stream is not None:
                try:
                    stream.StreamDestroy()
                except Exception:  # noqa: BLE001 - preserve initialization failure
                    logger.warning("RPP stream initialization cleanup failed")
            raise ModelUnavailableError(
                "RPP GIL-releasing execution is unsupported by this SDK/runtime"
            ) from exc

    def synchronize(self) -> None:
        status = int(self._sync(ctypes.c_void_p(self.handle)))
        if status != 0:
            raise ModelUnavailableError(f"RPP native stream synchronization failed: {status}")

    def close(self) -> None:
        self.stream.StreamDestroy()


class _InferenceEngine(Protocol):
    input_shape: tuple[int, ...]
    output_shapes: Mapping[str, tuple[int, ...]]

    def run(self, input_tensor: np.ndarray) -> dict[str, np.ndarray]: ...

    def close(self) -> None: ...


EngineFactory = Callable[..., _InferenceEngine]


def _scrfd_output_shapes(input_hw: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    height, width = input_hw
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError(f"SCRFD input must be positive multiples of 32, got {input_hw}")
    grid_counts = tuple(
        (height // stride) * (width // stride) * _SCRFD_ANCHORS
        for stride in _SCRFD_STRIDES
    )
    return (
        *((count, 1) for count in grid_counts),
        *((count, 4) for count in grid_counts),
        *((count, 10) for count in grid_counts),
    )


def _set_shape_dimensions(value_info: object, dimensions: Sequence[int]) -> None:
    shape = value_info.type.tensor_type.shape
    if len(shape.dim) != len(dimensions):
        raise ValueError(f"ONNX tensor {value_info.name} has an unexpected rank")
    for dimension, value in zip(shape.dim, dimensions, strict=True):
        dimension.ClearField("dim_param")
        dimension.dim_value = int(value)


def _onnx_shape(value_info: object) -> tuple[int, ...]:
    dimensions = value_info.type.tensor_type.shape.dim
    return tuple(int(item.dim_value) for item in dimensions)


def _validate_static_scrfd_model(
    model_bytes: bytes,
    *,
    input_hw: tuple[int, int],
    output_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    import onnx

    model = onnx.load_model_from_string(model_bytes)
    if len(model.graph.input) != 1 or _onnx_shape(model.graph.input[0]) != (
        1,
        3,
        input_hw[0],
        input_hw[1],
    ):
        raise ValueError("specialized SCRFD graph has an unexpected input shape")
    if len(model.graph.output) != 9:
        raise ValueError("SCRFD graph must expose 9 outputs")
    expected_shapes = _scrfd_output_shapes(input_hw)
    actual_names = tuple(value.name for value in model.graph.output)
    if output_names is not None and actual_names != tuple(output_names):
        raise ValueError("specialized SCRFD output names changed during optimization")
    for value_info, expected_shape in zip(model.graph.output, expected_shapes, strict=True):
        if _onnx_shape(value_info) != expected_shape:
            raise ValueError(
                f"SCRFD output {value_info.name} has shape {_onnx_shape(value_info)}, "
                f"expected {expected_shape}"
            )
    if any(node.op_type == "Shape" for node in model.graph.node):
        raise ValueError("SCRFD graph still contains dynamic Shape nodes")
    if any(node.domain not in {"", "ai.onnx"} for node in model.graph.node):
        raise ValueError("SCRFD graph contains optimizer-specific custom operators")
    return actual_names


def _scrfd_static_model_path(
    model_bytes: bytes, *, input_hw: tuple[int, int], model_dir: Path
) -> Path:
    height, width = input_hw
    source_digest = hashlib.sha256(model_bytes).hexdigest()
    return model_dir / (
        f"scrfd-{_SCRFD_STATIC_GRAPH_REVISION}_{width}x{height}_{source_digest[:16]}.onnx"
    )


def _static_scrfd_model(
    model_bytes: bytes,
    *,
    input_hw: tuple[int, int],
    cache_dir: Path,
) -> tuple[bytes, tuple[str, ...]]:
    """Specialize SCRFD shape metadata and fold its dynamic shape subgraph."""
    import onnx
    import onnxruntime as ort

    cache_dir.mkdir(parents=True, exist_ok=True)
    height, width = input_hw
    expected_shapes = _scrfd_output_shapes(input_hw)
    cache_path = _scrfd_static_model_path(
        model_bytes, input_hw=input_hw, model_dir=cache_dir
    )
    if cache_path.is_file():
        cached = cache_path.read_bytes()
        return cached, _validate_static_scrfd_model(cached, input_hw=input_hw)

    model = onnx.load_model_from_string(model_bytes)
    if len(model.graph.input) != 1 or len(model.graph.output) != 9:
        raise ValueError("SCRFD ONNX graph must have one input and nine outputs")
    output_names = tuple(value.name for value in model.graph.output)
    _set_shape_dimensions(model.graph.input[0], (1, 3, height, width))
    for value_info, expected_shape in zip(
        model.graph.output, expected_shapes, strict=True
    ):
        _set_shape_dimensions(value_info, expected_shape)
    onnx.checker.check_model(model)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".scrfd_{width}x{height}_", suffix=".onnx", dir=cache_dir
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # This session only rewrites shapes; it never performs inference.  An
        # explicit single-thread pool avoids ORT attempting to pin workers to
        # CPUs outside the container's RK3588 cpuset during first readiness.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.log_severity_level = 3
        options.optimized_model_filepath = str(temporary_path)
        session = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        if session.get_providers()[:1] != ["CPUExecutionProvider"]:
            raise ValueError("SCRFD graph specialization did not use the CPU optimizer")
        if len(session.get_inputs()) != 1 or tuple(session.get_inputs()[0].shape) != (
            1,
            3,
            height,
            width,
        ):
            raise ValueError("ONNX Runtime changed the specialized SCRFD input shape")
        session_outputs = session.get_outputs()
        if tuple(item.name for item in session_outputs) != output_names:
            raise ValueError("ONNX Runtime changed SCRFD output ordering")
        if tuple(tuple(item.shape) for item in session_outputs) != expected_shapes:
            raise ValueError("ONNX Runtime changed a specialized SCRFD output shape")
        del session

        optimized_bytes = temporary_path.read_bytes()
        _validate_static_scrfd_model(
            optimized_bytes, input_hw=input_hw, output_names=output_names
        )
        # Concurrent initializers specialize the same bytes deterministically;
        # replacing an identical cache artifact is safe and atomic.
        temporary_path.replace(cache_path)
        return optimized_bytes, output_names
    finally:
        temporary_path.unlink(missing_ok=True)


def _rpp_sdk_fingerprint(pyrt: object) -> str:
    module_path = Path(str(getattr(pyrt, "__file__", "")))
    runtime_candidates = (
        module_path,
        Path("/usr/local/rpp/lib/libRppRT.so.1.0.1"),
    )
    pieces: list[str] = []
    for path in runtime_candidates:
        try:
            stat = path.stat()
            pieces.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            pieces.append(f"{path.name}:missing")
    return hashlib.sha256("|".join(pieces).encode()).hexdigest()[:16]


class _RppExecutionEngine:
    """A static RPP engine with reusable input/output device allocations."""

    def __init__(
        self,
        *,
        engine: object,
        runtime: object,
        input_shape: tuple[int, ...],
        output_shapes: Mapping[str, tuple[int, ...]],
        owners: tuple[object, ...] = (),
        release_gil: bool | None = None,
    ):
        import pyrt

        self._engine = engine
        self._runtime = runtime
        # Native SDK objects borrow their logger and may borrow plan/weight data.
        # Keep these alive until contexts, engines and runtimes are destroyed.
        self._owners: tuple[object, ...] = owners
        self._context = engine.createExecutionContext()
        self._native_stream: _RppNativeStream | None = None
        self._execution_failed = False
        self._closed = False
        self._input_shape = input_shape
        self._output_shapes = dict(output_shapes)
        self._input_index: int | None = None
        self._output_indices: dict[str, int] = {}
        self._devices: dict[int, object] = {}
        self._hosts: dict[int, object] = {}

        for index in range(int(engine.num_bindings)):
            name = str(engine.get_binding_name(index))
            shape = tuple(int(value) for value in engine.get_binding_shape(index).get())
            dtype = str(engine.get_binding_dtype(index)).upper()
            if "FLOAT" not in dtype:
                raise ValueError(f"RPP binding {name} must use float32, got {dtype}")
            count = int(np.prod(shape, dtype=np.int64))
            if count <= 0:
                raise ValueError(f"RPP binding {name} has an invalid shape {shape}")
            device = pyrt.DeviceAllocation(count * np.dtype(np.float32).itemsize)
            self._devices[index] = device
            if engine.binding_is_input(index):
                if self._input_index is not None:
                    raise ValueError("RPP face model must have exactly one input")
                if shape != input_shape:
                    raise ValueError(
                        f"RPP input shape {shape} does not match expected {input_shape}"
                    )
                self._input_index = index
            else:
                if name in self._output_indices:
                    raise ValueError(f"RPP output binding name {name} is duplicated")
                self._output_indices[name] = index
                self._hosts[index] = pyrt.HostAllocation(
                    count * np.dtype(np.float32).itemsize
                )
                if self._output_shapes.get(name) != shape:
                    raise ValueError(
                        f"RPP output {name} has shape {shape}, expected "
                        f"{self._output_shapes.get(name)}"
                    )
        if self._input_index is None:
            raise ValueError("RPP face model has no input binding")
        if set(self._output_indices) != set(self._output_shapes):
            raise ValueError("RPP output bindings do not match the ONNX graph outputs")
        self._bindings = [
            int(self._devices[index].GetBufferPrt())
            for index in range(int(engine.num_bindings))
        ]
        release_gil_requested = (
            _env_flag("PERSON_SEARCH_RPP_RELEASE_GIL")
            if release_gil is None
            else release_gil
        )
        if release_gil_requested:
            if not callable(getattr(self._context, "enqueue", None)):
                raise ModelUnavailableError("RPP GIL-releasing execution requires SDK enqueue")
            self._native_stream = _RppNativeStream(pyrt)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def output_shapes(self) -> Mapping[str, tuple[int, ...]]:
        return self._output_shapes

    @property
    def execution_wait_mode(self) -> str:
        return "native_stream_sync" if self._native_stream is not None else "sync_execute"

    def run(self, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
        array = np.ascontiguousarray(input_tensor, dtype=np.float32)
        if array.shape != self._input_shape:
            raise ValueError(
                f"RPP input shape {array.shape} does not match {self._input_shape}"
            )
        input_index = self._input_index
        if input_index is None:
            raise ModelUnavailableError("RPP input binding is not initialized")
        with _RPP_EXECUTION_LOCK:
            if self._closed or self._execution_failed:
                raise ModelUnavailableError("RPP execution engine is closed or failed")
            self._devices[input_index].copy_from_numpy(array)
            if self._native_stream is None:
                if not self._context.execute(1, self._bindings):
                    raise ModelUnavailableError("RPP execution failed")
            else:
                try:
                    if not self._context.enqueue(1, self._bindings, self._native_stream.stream):
                        raise ModelUnavailableError("RPP enqueue failed")
                    self._native_stream.synchronize()
                except Exception as exc:
                    self._execution_failed = True
                    _RPP_QUARANTINED_ENGINES.append(self)
                    if isinstance(exc, ModelUnavailableError):
                        raise
                    raise ModelUnavailableError("RPP asynchronous execution failed") from exc
            outputs: dict[str, np.ndarray] = {}
            for name, index in self._output_indices.items():
                host = self._hosts[index]
                if not self._devices[index].copy_to_host(host):
                    raise ModelUnavailableError(f"RPP output {name} could not be copied")
                shape = self._output_shapes[name]
                output = host.numpy_float().reshape(shape).copy()
                if not np.isfinite(output).all():
                    raise ModelUnavailableError(f"RPP output {name} contains non-finite values")
                outputs[name] = output
            return outputs

    def close(self) -> None:
        with _RPP_EXECUTION_LOCK:
            if self._closed:
                return
            self._closed = True
            if self._execution_failed:
                logger.error("retaining failed RPP execution resources until process exit")
                return
            self._context = None
            if self._native_stream is not None:
                self._native_stream.close()
                self._native_stream = None
            self._hosts.clear()
            self._devices.clear()
            self._engine = None
            self._runtime = None
            self._owners = ()


def _load_rpp_engine(
    model_bytes: bytes,
    *,
    cache_dir: Path,
    model_key: str,
    input_shape: tuple[int, ...],
    output_shapes: Mapping[str, tuple[int, ...]],
    engine_cache_enabled: bool | None = None,
    release_gil: bool | None = None,
) -> _RppExecutionEngine:
    try:
        import pyrt
    except ImportError as exc:
        raise ModelUnavailableError(
            "RPP Python SDK is unavailable; mount the EIS100 RPP runtime and set PYTHONPATH"
        ) from exc

    logger_obj = pyrt.Logger(pyrt.Logger.ERROR)
    # The EIS100 SDK on RK3588 can serialize BF16 plans that abort the process
    # with a native "Invalid DataType" on deserialize. Opt in only after an SDK
    # version has proven round-trip support; building does not use CPU inference.
    use_plan_cache = (
        _env_flag("PERSON_SEARCH_RPP_ENGINE_CACHE_ENABLED")
        if engine_cache_enabled is None
        else engine_cache_enabled
    )
    cache_path: Path | None = None
    if use_plan_cache:
        sdk_fingerprint = _rpp_sdk_fingerprint(pyrt)
        input_hw = (input_shape[-2], input_shape[-1])
        digest = hashlib.sha256(
            model_bytes
            + f"{model_key}|{input_hw}|bf16|{sdk_fingerprint}".encode()
        ).hexdigest()[:20]
        cache_path = cache_dir / f"{model_key}_{input_hw[1]}x{input_hw[0]}_{digest}.plan"

    runtime = None
    if cache_path is not None and cache_path.is_file():
        try:
            runtime = pyrt.Runtime(logger_obj)
            plan_bytes = cache_path.read_bytes()
            engine = runtime.deserializeCudaEngine(plan_bytes)
            loaded = _RppExecutionEngine(
                engine=engine,
                runtime=runtime,
                input_shape=input_shape,
                output_shapes=output_shapes,
                owners=(logger_obj, plan_bytes),
                release_gil=release_gil,
            )
            logger.info("loaded RPP engine cache %s", cache_path.name)
            return loaded
        except Exception as exc:
            raise ModelUnavailableError(
                f"RPP engine cache {cache_path.name} is invalid: {exc}"
            ) from exc

    builder = pyrt.Builder(logger_obj)
    builder.setMaxBatchSize(1)
    network = builder.createNetwork()
    parser = pyrt.OnnxParser(network, logger_obj)
    if not parser.parse(model_bytes):
        raise ModelUnavailableError(f"RPP could not parse {model_key} ONNX graph")
    if network.num_inputs != 1 or network.num_outputs != len(output_shapes):
        raise ModelUnavailableError(
            f"RPP parsed {model_key} with unexpected input/output counts"
        )
    input_tensor = network.get_input(0)
    if tuple(int(value) for value in input_tensor.dimensions) != input_shape:
        raise ModelUnavailableError(
            f"RPP parsed {model_key} input as {tuple(input_tensor.dimensions)}, "
            f"expected {input_shape}"
        )
    parsed_outputs = {
        str(network.get_output(index).name): tuple(
            int(value) for value in network.get_output(index).dimensions
        )
        for index in range(network.num_outputs)
    }
    if parsed_outputs != dict(output_shapes):
        raise ModelUnavailableError(f"RPP parsed {model_key} with unexpected output shapes")

    config = builder.createBuilderConfig()
    # This SDK exposes FP16 as an alias of BF16. The contract/readiness metadata
    # deliberately says BF16 rather than implying IEEE half precision.
    config.setFlag(pyrt.BuilderFlag.BF16)
    try:
        engine = builder.build_EngineWithConfig(network, config)
        plan = bytes(engine.serialize()) if use_plan_cache else None
    except Exception as exc:
        raise ModelUnavailableError(f"RPP could not build {model_key}: {exc}") from exc
    if use_plan_cache and not plan:
        raise ModelUnavailableError(f"RPP produced an empty engine for {model_key}")

    if cache_path is not None and plan is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{cache_path.name}.", suffix=".tmp", dir=cache_dir
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(plan)
                output.flush()
                os.fsync(output.fileno())
            try:
                temporary_path.replace(cache_path)
            except FileExistsError:
                temporary_path.unlink(missing_ok=True)
        finally:
            temporary_path.unlink(missing_ok=True)

    # Keep builder objects alive with the returned engine: some SDK builds retain
    # parser-owned weights until the engine is destroyed.
    loaded = _RppExecutionEngine(
        engine=engine,
        runtime=runtime,
        input_shape=input_shape,
        output_shapes=output_shapes,
        owners=(logger_obj, model_bytes, builder, network, parser, config),
        release_gil=release_gil,
    )
    logger.info(
        "built RPP BF16 engine for %s%s",
        model_key,
        f" ({cache_path.name})" if cache_path is not None else " (plan cache disabled)",
    )
    return loaded


def _preprocess_scrfd(frame: np.ndarray, input_hw: tuple[int, int]) -> tuple[np.ndarray, float]:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ModelUnavailableError("RPP SCRFD requires a BGR HWC frame")
    if frame.shape[0] <= 0 or frame.shape[1] <= 0:
        raise ModelUnavailableError("RPP SCRFD received an empty frame")
    target_h, target_w = input_hw
    ratio = min(target_h / frame.shape[0], target_w / frame.shape[1])
    resized_w = max(1, min(target_w, int(frame.shape[1] * ratio)))
    resized_h = max(1, min(target_h, int(frame.shape[0] * ratio)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded[:resized_h, :resized_w] = rgb
    blob = cv2.dnn.blobFromImage(
        padded,
        1.0 / 128.0,
        (target_w, target_h),
        (127.5, 127.5, 127.5),
        swapRB=False,
    )
    return np.ascontiguousarray(blob[0], dtype=np.float32), ratio


def _decode_scrfd(
    frame: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    output_names: Sequence[str],
    input_hw: tuple[int, int],
    ratio: float,
    *,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(output_names) != 9 or any(name not in outputs for name in output_names):
        raise ModelUnavailableError("RPP SCRFD output bindings do not match the model")
    height, width = input_hw
    boxes: list[np.ndarray] = []
    landmarks: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    expected_shapes = _scrfd_output_shapes(input_hw)

    for level, stride in enumerate(_SCRFD_STRIDES):
        score = np.asarray(outputs[output_names[level]], dtype=np.float32)
        bbox = np.asarray(outputs[output_names[3 + level]], dtype=np.float32)
        kps = np.asarray(outputs[output_names[6 + level]], dtype=np.float32)
        if (score.shape, bbox.shape, kps.shape) != (
            expected_shapes[level],
            expected_shapes[3 + level],
            expected_shapes[6 + level],
        ):
            raise ModelUnavailableError("RPP SCRFD output tensors have invalid shapes")
        grid_h, grid_w = height // stride, width // stride
        centers_y, centers_x = np.meshgrid(
            np.arange(grid_h, dtype=np.float32),
            np.arange(grid_w, dtype=np.float32),
            indexing="ij",
        )
        centers = np.stack((centers_x, centers_y), axis=-1).reshape(-1, 2) * stride
        centers = np.repeat(centers, _SCRFD_ANCHORS, axis=0)
        scores_flat = score.reshape(-1)
        bbox_flat = bbox.reshape(-1, 4)
        kps_flat = kps.reshape(-1, 10)
        if (
            not np.isfinite(scores_flat).all()
            or not np.isfinite(bbox_flat).all()
            or not np.isfinite(kps_flat).all()
        ):
            raise ModelUnavailableError("RPP SCRFD output contains non-finite values")
        selected = np.flatnonzero(scores_flat >= threshold)
        if selected.size == 0:
            continue
        selected_centers = centers[selected]
        distances = bbox_flat[selected] * stride
        decoded = np.empty((selected.size, 4), dtype=np.float32)
        decoded[:, 0] = selected_centers[:, 0] - distances[:, 0]
        decoded[:, 1] = selected_centers[:, 1] - distances[:, 1]
        decoded[:, 2] = selected_centers[:, 0] + distances[:, 2]
        decoded[:, 3] = selected_centers[:, 1] + distances[:, 3]
        selected_kps = kps_flat[selected].reshape(-1, 5, 2) * stride
        selected_kps += selected_centers[:, None, :]
        boxes.append(decoded)
        landmarks.append(selected_kps)
        scores.append(scores_flat[selected])

    if not boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0, 5, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    all_boxes = np.concatenate(boxes, axis=0) / ratio
    all_landmarks = np.concatenate(landmarks, axis=0) / ratio
    all_scores = np.concatenate(scores, axis=0)
    frame_h, frame_w = frame.shape[:2]
    all_boxes[:, [0, 2]] = np.clip(all_boxes[:, [0, 2]], 0, frame_w - 1)
    all_boxes[:, [1, 3]] = np.clip(all_boxes[:, [1, 3]], 0, frame_h - 1)
    all_landmarks[:, :, 0] = np.clip(all_landmarks[:, :, 0], 0, frame_w - 1)
    all_landmarks[:, :, 1] = np.clip(all_landmarks[:, :, 1], 0, frame_h - 1)
    keep = _nms(all_boxes, all_scores, 0.4)
    return all_boxes[keep], all_landmarks[keep], all_scores[keep]


class RppFaceBackend:
    """SCRFD and ArcFace through the external RPP accelerator (BF16)."""

    def __init__(
        self,
        settings: Settings,
        *,
        engine_factory: EngineFactory = _load_rpp_engine,
    ):
        self.settings = settings
        self.model_name = settings.insightface_model
        self.provider_name = "uninitialized"
        self.detection_provider_name = "uninitialized"
        self.recognition_provider_name = "uninitialized"
        self.embedding_contract: EmbeddingContract | None = None
        self._engine_factory = engine_factory
        self._search_detector: _InferenceEngine | None = None
        self._enrollment_detector: _InferenceEngine | None = None
        self._recognizer: _InferenceEngine | None = None
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
            and self._recognizer is not None
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
            recognizer: _InferenceEngine | None = None
            try:
                manifest = EMBEDDING_MODEL_MANIFESTS.get(self.settings.insightface_model)
                if manifest is None:
                    raise ModelUnavailableError(
                        f"RPP face backend does not support {self.settings.insightface_model}"
                    )
                model_dir = (
                    self.settings.insightface_root
                    / "models"
                    / self.settings.insightface_model
                )
                detector_path = model_dir / "det_10g.onnx"
                recognizer_path = model_dir / manifest.recognition_filename
                if not detector_path.is_file() or not recognizer_path.is_file():
                    raise ModelUnavailableError(
                        "RPP SCRFD/ArcFace ONNX weights are missing from the configured model root"
                    )
                actual_recognizer_sha = sha256(recognizer_path)
                if actual_recognizer_sha != manifest.recognition_sha256:
                    raise ModelUnavailableError("ArcFace ONNX checksum does not match the manifest")

                self.settings.rpp_engine_cache_dir.mkdir(parents=True, exist_ok=True)
                detector_bytes = detector_path.read_bytes()
                search_hw = self.static_input_hw
                enrollment_hw = self.enrollment_static_input_hw
                search_graph, output_names = _static_scrfd_model(
                    detector_bytes,
                    input_hw=search_hw,
                    cache_dir=self.settings.rpp_engine_cache_dir,
                )
                search_shapes = dict(
                    zip(output_names, _scrfd_output_shapes(search_hw), strict=True)
                )
                search_detector = self._engine_factory(
                    search_graph,
                    cache_dir=self.settings.rpp_engine_cache_dir,
                    model_key="scrfd-search",
                    engine_cache_enabled=self.settings.rpp_engine_cache_enabled,
                    release_gil=self.settings.rpp_release_gil,
                    input_shape=(3, *search_hw),
                    output_shapes=search_shapes,
                )

                if enrollment_hw == search_hw:
                    enrollment_detector = search_detector
                else:
                    enrollment_graph, enrollment_names = _static_scrfd_model(
                        detector_bytes,
                        input_hw=enrollment_hw,
                        cache_dir=self.settings.rpp_engine_cache_dir,
                    )
                    if enrollment_names != output_names:
                        raise ModelUnavailableError(
                            "SCRFD output names changed between search and enrollment graphs"
                        )
                    enrollment_shapes = dict(
                        zip(
                            enrollment_names,
                            _scrfd_output_shapes(enrollment_hw),
                            strict=True,
                        )
                    )
                    enrollment_detector = self._engine_factory(
                        enrollment_graph,
                        cache_dir=self.settings.rpp_engine_cache_dir,
                        model_key="scrfd-enrollment",
                        engine_cache_enabled=self.settings.rpp_engine_cache_enabled,
                        release_gil=self.settings.rpp_release_gil,
                        input_shape=(3, *enrollment_hw),
                        output_shapes=enrollment_shapes,
                    )

                import onnx

                recognizer_bytes = recognizer_path.read_bytes()
                recognizer_model = onnx.load_model_from_string(recognizer_bytes)
                if len(recognizer_model.graph.input) != 1 or len(
                    recognizer_model.graph.output
                ) != 1:
                    raise ModelUnavailableError("ArcFace ONNX must have one input and output")
                if _onnx_shape(recognizer_model.graph.input[0])[-3:] != (3, 112, 112):
                    raise ModelUnavailableError("ArcFace ONNX input must be 3x112x112")
                recognizer_name = recognizer_model.graph.output[0].name
                recognizer = self._engine_factory(
                    recognizer_bytes,
                    cache_dir=self.settings.rpp_engine_cache_dir,
                    model_key="arcface-w600k-r50",
                    engine_cache_enabled=self.settings.rpp_engine_cache_enabled,
                    release_gil=self.settings.rpp_release_gil,
                    input_shape=(3, 112, 112),
                    output_shapes={recognizer_name: (manifest.embedding_dimension,)},
                )

                if search_detector.input_shape != (3, *search_hw):
                    raise ModelUnavailableError("RPP search SCRFD input shape is incorrect")
                if enrollment_detector.input_shape != (3, *enrollment_hw):
                    raise ModelUnavailableError("RPP enrollment SCRFD input shape is incorrect")
                if recognizer.input_shape != (3, 112, 112):
                    raise ModelUnavailableError("RPP ArcFace input shape is incorrect")

                # Readiness means the real RPP device executed every configured graph.
                search_detector.run(np.zeros(search_detector.input_shape, dtype=np.float32))
                if enrollment_detector is not search_detector:
                    enrollment_detector.run(
                        np.zeros(enrollment_detector.input_shape, dtype=np.float32)
                    )
                recognizer.run(np.zeros(recognizer.input_shape, dtype=np.float32))

                contract = EmbeddingContract(
                    schema_version="arcface-v1-rpp-bf16",
                    model_name=self.settings.insightface_model,
                    model_sha256=actual_recognizer_sha,
                    embedding_dimension=manifest.embedding_dimension,
                    input_size=manifest.input_size,
                    flip_tta=self.settings.embedding_flip_tta,
                )
            except ModelUnavailableError:
                unique_engines = {
                    id(engine): engine
                    for engine in (search_detector, enrollment_detector, recognizer)
                    if engine is not None
                }
                for engine in unique_engines.values():
                    engine.close()
                raise
            except Exception as exc:
                unique_engines = {
                    id(engine): engine
                    for engine in (search_detector, enrollment_detector, recognizer)
                    if engine is not None
                }
                for engine in unique_engines.values():
                    engine.close()
                raise ModelUnavailableError(
                    f"RPP face model initialization failed: {exc}"
                ) from exc

            self._search_detector = search_detector
            self._enrollment_detector = enrollment_detector
            self._recognizer = recognizer
            self._detection_output_names = output_names
            self.embedding_contract = contract
            self.detection_provider_name = "RPP_BF16"
            self.recognition_provider_name = "RPP_BF16"
            self.provider_name = "RPP_BF16"

    def detect_faces(
        self,
        frame: np.ndarray,
        *,
        enrollment: bool = False,
        detection_size: int | Sequence[int] | None = None,
    ) -> list[FaceObservation]:
        # RPP engines are deliberately static and match the existing RKNN graph
        # dimensions. The protocol argument cannot override their hardware shape.
        del detection_size
        self.ensure_ready()
        detector = self._enrollment_detector if enrollment else self._search_detector
        input_hw = self.enrollment_static_input_hw if enrollment else self.static_input_hw
        if detector is None:
            raise ModelUnavailableError("RPP SCRFD engine did not initialize")
        input_tensor, ratio = _preprocess_scrfd(frame, input_hw)
        outputs = detector.run(input_tensor)
        boxes, landmarks, scores = _decode_scrfd(
            frame,
            outputs,
            self._detection_output_names,
            input_hw,
            ratio,
            threshold=self.settings.face_detection_threshold,
        )
        observations: list[FaceObservation] = []
        for bbox, kps, score in zip(boxes, landmarks, scores, strict=True):
            quality = assess_face(
                frame, bbox, kps, float(score), self.settings, enrollment=enrollment
            )
            observations.append(
                FaceObservation(
                    bbox=np.asarray(bbox, dtype=np.float32),
                    detection_score=float(score),
                    embedding=None,
                    quality=quality.score,
                    landmarks=np.asarray(kps, dtype=np.float32),
                    accepted=quality.accepted,
                    rejection_reasons=quality.reasons,
                    blur_variance=quality.blur_variance,
                )
            )
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
                frame,
                enrollment=enrollment,
                detection_size=detection_size,
            )
            for frame in frames
        ]

    def embed_faces(
        self, frame: np.ndarray, faces: list[FaceObservation]
    ) -> list[FaceObservation]:
        if not faces:
            return []
        self.ensure_ready()
        recognizer = self._recognizer
        if recognizer is None:
            raise ModelUnavailableError("RPP ArcFace engine did not initialize")
        try:
            from insightface.utils import face_align
        except ImportError as exc:
            raise ModelUnavailableError(
                "InsightFace face alignment helpers are required by the RPP backend"
            ) from exc

        embedded: list[FaceObservation] = []
        for face in faces:
            if face.landmarks is None:
                continue
            try:
                crop = face_align.norm_crop(
                    frame, landmark=face.landmarks, image_size=112
                )
                if (
                    not isinstance(crop, np.ndarray)
                    or crop.shape[:2] != (112, 112)
                    or crop.size == 0
                ):
                    continue
                blob = cv2.dnn.blobFromImage(
                    crop,
                    1.0 / 127.5,
                    (112, 112),
                    (127.5, 127.5, 127.5),
                    swapRB=True,
                )[0]
                output = recognizer.run(np.ascontiguousarray(blob, dtype=np.float32))
                feature = self._arcface_feature(output)
                if self.settings.embedding_flip_tta:
                    flipped = cv2.dnn.blobFromImage(
                        cv2.flip(crop, 1),
                        1.0 / 127.5,
                        (112, 112),
                        (127.5, 127.5, 127.5),
                        swapRB=True,
                    )[0]
                    feature = feature + self._arcface_feature(
                        recognizer.run(np.ascontiguousarray(flipped, dtype=np.float32))
                    )
                embedding = normalize_embedding(feature)
            except ModelUnavailableError:
                raise
            except (TypeError, ValueError, OverflowError, FloatingPointError):
                logger.debug("discarding face crop with invalid RPP ArcFace embedding")
                continue
            embedded.append(replace(face, embedding=embedding))
        return embedded

    @staticmethod
    def _arcface_feature(outputs: Mapping[str, np.ndarray]) -> np.ndarray:
        if len(outputs) != 1:
            raise ModelUnavailableError("RPP ArcFace must return one embedding tensor")
        feature = np.asarray(next(iter(outputs.values())), dtype=np.float32).reshape(-1)
        if feature.size != 512 or not np.isfinite(feature).all():
            raise ModelUnavailableError("RPP ArcFace output is not a finite 512-D vector")
        return feature

    def analyze(self, frame: np.ndarray, *, enrollment: bool = False) -> list[FaceObservation]:
        return self.embed_faces(frame, self.detect_faces(frame, enrollment=enrollment))

    def close(self) -> None:
        unique_engines = {
            id(engine): engine
            for engine in (
                self._search_detector,
                self._enrollment_detector,
                self._recognizer,
            )
            if engine is not None
        }
        for engine in unique_engines.values():
            engine.close()
        self._search_detector = None
        self._enrollment_detector = None
        self._recognizer = None
        self._detection_output_names = ()
        self.provider_name = "uninitialized"
        self.detection_provider_name = "uninitialized"
        self.recognition_provider_name = "uninitialized"
        self.embedding_contract = None
