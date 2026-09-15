"""Small ctypes wrapper for the Rockchip RKNN runtime.

The RK3588 image used by the EIS100 box ships ``librknnrt.so`` but does not ship
the Python ``rknnlite`` module.  Keeping this wrapper deliberately narrow lets
the application use the system runtime from Python 3.11 without coupling the
service to a particular Toolkit2 Python wheel.

The conversion side is intentionally separate (see
``scripts/convert_rk3588_models.py``).  This module only loads an already
converted model, validates its tensor metadata, runs it, and copies outputs out
of the runtime-owned buffers before releasing them.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
import platform
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from .errors import ModelUnavailableError

logger = logging.getLogger(__name__)

# Values are part of the public RKNN C ABI.  They are kept here rather than
# importing a Toolkit2 package which is only needed when converting models.
RKNN_SUCC = 0
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_TENSOR_FLOAT32 = 0
RKNN_TENSOR_FLOAT16 = 1
RKNN_TENSOR_INT8 = 2
RKNN_TENSOR_UINT8 = 3
RKNN_TENSOR_INT16 = 4
RKNN_TENSOR_UINT16 = 5
RKNN_TENSOR_INT32 = 6
RKNN_TENSOR_UINT32 = 7
RKNN_TENSOR_INT64 = 8
RKNN_TENSOR_UINT64 = 9
RKNN_TENSOR_BOOL = 10
RKNN_TENSOR_NCHW = 0
RKNN_TENSOR_NHWC = 1
RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256
# rknn_core_mask: RK3588 has three NPU cores; AUTO runs a context on one of them.
RKNN_NPU_CORE_AUTO = 0
RKNN_NPU_CORE_0_1_2 = 7


class _TensorAttr(ctypes.Structure):
    """Mirror ``rknn_tensor_attr`` from rknn_api.h.

    The final ``h_stride`` is a uint32_t.  It is easy to mistake it for the
    preceding one-byte ``pass_through`` flag; doing so shifts every query field
    on ARM64 and produces plausible-looking but wrong shapes.
    """

    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int),
        ("type", ctypes.c_int),
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]


class _IoNum(ctypes.Structure):
    _fields_ = [("n_inputs", ctypes.c_uint32), ("n_outputs", ctypes.c_uint32)]


class _RknnInput(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int),
        ("fmt", ctypes.c_int),
    ]


class _RknnOutput(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]


@dataclass(frozen=True, slots=True)
class TensorInfo:
    index: int
    shape: tuple[int, ...]
    name: str
    size: int
    tensor_format: int
    tensor_type: int
    element_count: int
    scale: float
    zero_point: int


def _context_type() -> type[ctypes._SimpleCData]:
    """Return the context scalar used by the RKNN ABI on this architecture."""

    # The vendor header uses uint32_t only under the 32-bit ``__arm__`` define;
    # aarch64 and x86_64 take the uint64_t branch.  Passing a 32-bit context to
    # rknn_destroy/query truncates the native handle and can segfault after a
    # seemingly successful rknn_init.
    machine = platform.machine().lower()
    return ctypes.c_uint32 if machine.startswith("arm") and "aarch64" not in machine else ctypes.c_uint64


def _candidate_runtime_paths(explicit: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    discovered = ctypes.util.find_library("rknnrt")
    if discovered:
        candidates.append(Path(discovered))
    candidates.extend(
        [
            Path("/usr/lib/librknnrt.so"),
            Path("/usr/lib/aarch64-linux-gnu/librknnrt.so"),
            Path("/usr/local/lib/librknnrt.so"),
            Path("/usr/local/lib/librknn_api.so"),
        ]
    )
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            result.append(candidate)
            seen.add(key)
    return result


def resolve_runtime_library(explicit: Path | None = None) -> Path | str:
    """Resolve an RKNN shared object or raise an actionable model error."""

    for candidate in _candidate_runtime_paths(explicit):
        if candidate.is_file():
            return candidate
        # ``find_library`` can return a soname, which the dynamic loader can
        # resolve even though it is not a filesystem path in the current cwd.
        if not candidate.is_absolute() and "/" not in str(candidate):
            return str(candidate)
    searched = ", ".join(str(path) for path in _candidate_runtime_paths(explicit))
    raise ModelUnavailableError(
        "RKNN runtime library not found; set "
        "PERSON_SEARCH_RKNN_RUNTIME_LIBRARY to librknnrt.so "
        f"(searched: {searched})"
    )


def _bind(function: Any, restype: Any, argtypes: list[Any]) -> Any:
    function.restype = restype
    function.argtypes = argtypes
    return function


class RknnModel:
    """Thread-safe, validated wrapper around one RKNN model context."""

    provider_name = "RKNN_NPU"

    def __init__(
        self,
        model_path: Path,
        runtime_library: Path | None = None,
        *,
        core_mask: int = RKNN_NPU_CORE_AUTO,
    ):
        self.model_path = Path(model_path).expanduser()
        self.runtime_library = runtime_library.expanduser() if runtime_library else None
        self.core_mask = int(core_mask)
        self.core_mask_applied = False
        self._runtime: Any | None = None
        self._context: ctypes._SimpleCData | None = None
        self._model_buffer: Any | None = None
        self._lock = threading.RLock()
        self.input_attrs: tuple[TensorInfo, ...] = ()
        self.output_attrs: tuple[TensorInfo, ...] = ()
        self._closed = False

    @property
    def is_ready(self) -> bool:
        return self._context is not None and not self._closed

    @property
    def input_shape(self) -> tuple[int, ...]:
        if not self.input_attrs:
            return ()
        return self.input_attrs[0].shape

    def ensure_ready(self) -> None:
        if self.is_ready:
            return
        with self._lock:
            if self.is_ready:
                return
            if not self.model_path.is_file():
                raise ModelUnavailableError(f"RKNN model not found at {self.model_path}")
            library = resolve_runtime_library(self.runtime_library)
            try:
                runtime = ctypes.CDLL(str(library))
                self._configure_functions(runtime)
                model_bytes = self.model_path.read_bytes()
                if not model_bytes:
                    raise ValueError("model file is empty")
                model_buffer = ctypes.create_string_buffer(model_bytes)
                context = _context_type()(0)
                result = runtime.rknn_init(
                    ctypes.byref(context),
                    ctypes.cast(model_buffer, ctypes.c_void_p),
                    ctypes.c_uint32(len(model_bytes)),
                    ctypes.c_uint32(0),
                    None,
                )
                self._check_result(result, "rknn_init")
                try:
                    input_attrs, output_attrs = self._query_tensor_info(runtime, context)
                    core_mask_applied = self._apply_core_mask(runtime, context)
                except Exception:
                    runtime.rknn_destroy(context)
                    raise
            except ModelUnavailableError:
                raise
            except Exception as exc:
                raise ModelUnavailableError(
                    f"failed to load RKNN model {self.model_path}: {exc}"
                ) from exc
            self._runtime = runtime
            self._context = context
            # Keep the backing bytes alive for the lifetime of the context. Some
            # vendor builds map model sections lazily instead of copying them.
            self._model_buffer = model_buffer
            self.input_attrs = input_attrs
            self.output_attrs = output_attrs
            self.core_mask_applied = core_mask_applied
            self._closed = False

    def _apply_core_mask(self, runtime: Any, context: ctypes._SimpleCData) -> bool:
        """Spread this context over several NPU cores; RK3588 has three.

        ``RKNN_NPU_CORE_AUTO`` keeps the vendor default of one core per context.
        Measured on the EIS100 box with ``RKNN_NPU_CORE_0_1_2``: ArcFace r50
        24 -> 15.7 ms, SCRFD-10G 87 -> 74 ms, YOLOX-Tiny unchanged.  A runtime
        without the symbol, or one that rejects the mask, keeps the default
        instead of failing readiness: the mask is a speed-up, not a correctness
        requirement, and ``core_mask_applied`` tells the panel whether it took.
        """
        if self.core_mask == RKNN_NPU_CORE_AUTO:
            return False
        try:
            set_core_mask = runtime.rknn_set_core_mask
        except AttributeError:
            logger.warning(
                "RKNN runtime has no rknn_set_core_mask; core mask %d ignored", self.core_mask
            )
            return False
        _bind(set_core_mask, ctypes.c_int, [_context_type(), ctypes.c_int])
        result = int(set_core_mask(context, ctypes.c_int(self.core_mask)))
        if result != RKNN_SUCC:
            logger.warning(
                "rknn_set_core_mask(%d) returned RKNN error code %d; using the runtime default",
                self.core_mask,
                result,
            )
            return False
        return True

    def run(self, data: np.ndarray) -> list[np.ndarray]:
        """Run one input tensor and return copied float32 output arrays.

        ``data`` is semantic image data (normally RGB uint8 HWC).  The wrapper
        reshapes it to the model's declared NCHW/NHWC layout and passes the
        caller's dtype through as the RKNN input dtype.  This works for the
        toolkit's common UINT8/INT8 preprocessed models as well as FP32 builds.
        """

        self.ensure_ready()
        runtime = self._runtime
        context = self._context
        if runtime is None or context is None:
            raise ModelUnavailableError("RKNN model initialization did not complete")
        if len(self.input_attrs) != 1:
            raise ModelUnavailableError(
                f"RKNN model must have one input tensor, got {len(self.input_attrs)}"
            )
        prepared, tensor_type, tensor_format = self._prepare_input(data, self.input_attrs[0])
        input_buffer = ctypes.c_void_p(int(prepared.ctypes.data))
        input_desc = _RknnInput(
            index=0,
            buf=input_buffer,
            size=prepared.nbytes,
            pass_through=0,
            type=tensor_type,
            fmt=tensor_format,
        )
        output_desc = (_RknnOutput * len(self.output_attrs))()
        for index, descriptor in enumerate(output_desc):
            descriptor.want_float = 1
            descriptor.is_prealloc = 0
            descriptor.index = index

        with self._lock:
            try:
                self._check_result(
                    runtime.rknn_inputs_set(context, 1, ctypes.byref(input_desc)),
                    "rknn_inputs_set",
                )
                self._check_result(runtime.rknn_run(context, None), "rknn_run")
                self._check_result(
                    runtime.rknn_outputs_get(
                        context, len(output_desc), output_desc, None
                    ),
                    "rknn_outputs_get",
                )
                try:
                    return [self._copy_output(descriptor, info) for descriptor, info in zip(
                        output_desc, self.output_attrs, strict=True
                    )]
                finally:
                    self._check_result(
                        runtime.rknn_outputs_release(context, len(output_desc), output_desc),
                        "rknn_outputs_release",
                    )
            except ModelUnavailableError:
                raise
            except Exception as exc:
                raise ModelUnavailableError(
                    f"RKNN inference failed for {self.model_path.name}: {exc}"
                ) from exc

    def close(self) -> None:
        with self._lock:
            if self._context is not None and self._runtime is not None:
                try:
                    self._runtime.rknn_destroy(self._context)
                except Exception:
                    logger.debug("failed to destroy RKNN context", exc_info=True)
            self._context = None
            self._runtime = None
            self._model_buffer = None
            self.input_attrs = ()
            self.output_attrs = ()
            self._closed = True

    def __enter__(self) -> Self:
        self.ensure_ready()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown path
        self.close()

    @staticmethod
    def _configure_functions(runtime: Any) -> None:
        context = _context_type()
        _bind(
            runtime.rknn_init,
            ctypes.c_int,
            [ctypes.POINTER(context), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p],
        )
        _bind(runtime.rknn_destroy, ctypes.c_int, [context])
        _bind(
            runtime.rknn_query,
            ctypes.c_int,
            [context, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32],
        )
        _bind(
            runtime.rknn_inputs_set,
            ctypes.c_int,
            [context, ctypes.c_uint32, ctypes.POINTER(_RknnInput)],
        )
        _bind(runtime.rknn_run, ctypes.c_int, [context, ctypes.c_void_p])
        _bind(
            runtime.rknn_outputs_get,
            ctypes.c_int,
            [context, ctypes.c_uint32, ctypes.POINTER(_RknnOutput), ctypes.c_void_p],
        )
        _bind(
            runtime.rknn_outputs_release,
            ctypes.c_int,
            [context, ctypes.c_uint32, ctypes.POINTER(_RknnOutput)],
        )

    @classmethod
    def _query_tensor_info(
        cls, runtime: Any, context: ctypes._SimpleCData
    ) -> tuple[tuple[TensorInfo, ...], tuple[TensorInfo, ...]]:
        counts = _IoNum()
        cls._check_result(
            runtime.rknn_query(
                context,
                RKNN_QUERY_IN_OUT_NUM,
                ctypes.byref(counts),
                ctypes.sizeof(counts),
            ),
            "rknn_query(in_out_num)",
        )
        if counts.n_inputs == 0 or counts.n_outputs == 0:
            raise ValueError("RKNN model reports no input or output tensors")
        inputs = tuple(
            cls._query_attr(runtime, context, RKNN_QUERY_INPUT_ATTR, index)
            for index in range(counts.n_inputs)
        )
        outputs = tuple(
            cls._query_attr(runtime, context, RKNN_QUERY_OUTPUT_ATTR, index)
            for index in range(counts.n_outputs)
        )
        return inputs, outputs

    @staticmethod
    def _query_attr(
        runtime: Any, context: ctypes._SimpleCData, command: int, index: int
    ) -> TensorInfo:
        attr = _TensorAttr()
        attr.index = index
        RknnModel._check_result(
            runtime.rknn_query(
                context, command, ctypes.byref(attr), ctypes.sizeof(attr)
            ),
            f"rknn_query(tensor={index})",
        )
        n_dims = min(int(attr.n_dims), RKNN_MAX_DIMS)
        shape = tuple(int(value) for value in attr.dims[:n_dims])
        if not shape or any(value <= 0 for value in shape):
            raise ValueError(f"RKNN tensor {index} has invalid shape {shape}")
        name = bytes(attr.name).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        return TensorInfo(
            index=int(attr.index),
            shape=shape,
            name=name,
            size=int(attr.size),
            tensor_format=int(attr.fmt),
            tensor_type=int(attr.type),
            element_count=int(attr.n_elems),
            scale=float(attr.scale),
            zero_point=int(attr.zp),
        )

    @staticmethod
    def _prepare_input(
        data: np.ndarray, info: TensorInfo
    ) -> tuple[np.ndarray, int, int]:
        array = np.asarray(data)
        if array.ndim == len(info.shape) - 1:
            expected = info.shape[1:] if info.shape[0] == 1 else info.shape
        else:
            expected = info.shape
        if array.ndim == 3 and len(expected) == 3:
            # Convert between the two image layouts without guessing from dtype.
            if info.tensor_format == RKNN_TENSOR_NCHW and array.shape[-1] in (1, 3, 4):
                array = array.transpose(2, 0, 1)
            elif info.tensor_format == RKNN_TENSOR_NHWC and array.shape[0] in (1, 3, 4):
                array = array.transpose(1, 2, 0)
        if tuple(int(value) for value in array.shape) != tuple(expected):
            raise ModelUnavailableError(
                f"RKNN input shape {array.shape} does not match model shape {info.shape}"
            )
        if array.dtype not in (np.uint8, np.int8, np.float16, np.float32, np.int16):
            array = array.astype(np.float32)
        array = np.ascontiguousarray(array)
        tensor_type = _rknn_type_for_dtype(array.dtype)
        return array, tensor_type, info.tensor_format

    @staticmethod
    def _copy_output(descriptor: _RknnOutput, info: TensorInfo) -> np.ndarray:
        if not descriptor.buf or descriptor.size <= 0:
            raise ValueError(f"RKNN output {info.index} returned an empty buffer")
        raw = ctypes.string_at(descriptor.buf, int(descriptor.size))
        # want_float=1 asks the runtime for float32 regardless of the model's
        # quantized storage type.  If an old runtime ignores that flag, fall back
        # to the declared output dtype and still validate the element count.
        expected_count = int(np.prod(info.shape, dtype=np.int64))
        if len(raw) == expected_count * np.dtype(np.float32).itemsize:
            values = np.frombuffer(raw, dtype=np.float32).copy()
        else:
            dtype = _numpy_dtype_for_rknn_type(info.tensor_type)
            values = np.frombuffer(raw, dtype=dtype).copy()
            if info.tensor_type in (RKNN_TENSOR_INT8, RKNN_TENSOR_UINT8):
                values = (values.astype(np.float32) - info.zero_point) * info.scale
            else:
                values = values.astype(np.float32, copy=False)
        if values.size != expected_count:
            raise ValueError(
                f"RKNN output {info.index} has {values.size} values; expected {expected_count}"
            )
        reshaped = values.reshape(info.shape)
        if not np.isfinite(reshaped).all():
            raise ValueError(f"RKNN output {info.index} contains non-finite values")
        return reshaped

    @staticmethod
    def _check_result(result: int, operation: str) -> None:
        if int(result) != RKNN_SUCC:
            raise ModelUnavailableError(f"{operation} returned RKNN error code {int(result)}")


def _rknn_type_for_dtype(dtype: np.dtype[Any]) -> int:
    if dtype == np.dtype(np.uint8):
        return RKNN_TENSOR_UINT8
    if dtype == np.dtype(np.int8):
        return RKNN_TENSOR_INT8
    if dtype == np.dtype(np.float16):
        return RKNN_TENSOR_FLOAT16
    if dtype == np.dtype(np.int16):
        return RKNN_TENSOR_INT16
    return RKNN_TENSOR_FLOAT32


def _numpy_dtype_for_rknn_type(tensor_type: int) -> np.dtype[Any]:
    return {
        RKNN_TENSOR_FLOAT32: np.dtype(np.float32),
        RKNN_TENSOR_FLOAT16: np.dtype(np.float16),
        RKNN_TENSOR_INT8: np.dtype(np.int8),
        RKNN_TENSOR_UINT8: np.dtype(np.uint8),
        RKNN_TENSOR_INT16: np.dtype(np.int16),
        RKNN_TENSOR_UINT16: np.dtype(np.uint16),
        RKNN_TENSOR_INT32: np.dtype(np.int32),
        RKNN_TENSOR_UINT32: np.dtype(np.uint32),
        RKNN_TENSOR_INT64: np.dtype(np.int64),
        RKNN_TENSOR_UINT64: np.dtype(np.uint64),
        RKNN_TENSOR_BOOL: np.dtype(np.bool_),
    }.get(tensor_type, np.dtype(np.float32))


__all__ = [
    "RKNN_NPU_CORE_0_1_2",
    "RKNN_NPU_CORE_AUTO",
    "RKNN_TENSOR_NCHW",
    "RKNN_TENSOR_NHWC",
    "RknnModel",
    "TensorInfo",
    "resolve_runtime_library",
]
