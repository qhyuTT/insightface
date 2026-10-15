"""Native input ABI and ownership checks using synthetic tensor bytes only."""

from __future__ import annotations

import ctypes
import types

import numpy as np
import pytest

from person_search import rpp_backends as rpp
from person_search.errors import ModelUnavailableError


def _integer(value):
    return value.value if hasattr(value, "value") else value


class _Function:
    def __init__(self, call):
        self.call = call

    def __call__(self, *args):
        return self.call(*args)


@pytest.fixture
def native(monkeypatch):
    state = types.SimpleNamespace(
        pointer=0x123456780000, device=0x234567890000,
        alloc_status=0, copy_status=0, free_status=0,
        events=[], data=bytearray(), copied=None,
    )

    def allocate(address, count):
        state.events.append(("alloc", _integer(count)))
        ctypes.cast(address, ctypes.POINTER(ctypes.c_void_p))[0] = state.pointer
        state.data = bytearray([0xAB] * _integer(count))
        return state.alloc_status

    def free(pointer):
        state.events.append(("free", _integer(pointer)))
        return state.free_status

    def copy_device(destination, source, count, kind):
        assert _integer(source) == state.pointer
        state.events.append(("h2d", _integer(destination), _integer(count), _integer(kind)))
        state.copied = bytes(state.data[:_integer(count)])
        return state.copy_status

    def copy_host(destination, source, count):
        assert _integer(destination) == state.pointer
        state.events.append(("memcpy", _integer(count)))
        state.data[:_integer(count)] = ctypes.string_at(source, _integer(count))
        return state.pointer

    def wipe(pointer, value, count):
        assert _integer(pointer) == state.pointer
        state.events.append(("memset", _integer(count)))
        state.data[:_integer(count)] = bytes([_integer(value)]) * _integer(count)
        return state.pointer

    state.runtime = types.SimpleNamespace(
        rtMallocHost=_Function(allocate), rtFreeHost=_Function(free),
        rtMemcpy=_Function(copy_device),
    )
    state.libc = types.SimpleNamespace(memcpy=_Function(copy_host), memset=_Function(wipe))

    def load(name):
        return {"liburpp.so.2": state.runtime, "libc.so.6": state.libc}[name]

    monkeypatch.setattr(rpp.ctypes, "CDLL", load)
    return state


def test_native_input_uses_full_width_abi_and_exact_tensor_bytes(native):
    tensor = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
    staging = rpp._RppNativeInput(tensor.nbytes)
    assert native.runtime.rtMallocHost.argtypes == [
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t,
    ]
    assert native.runtime.rtMallocHost.restype is ctypes.c_int
    assert native.runtime.rtFreeHost.argtypes == [ctypes.c_void_p]
    assert native.runtime.rtFreeHost.restype is ctypes.c_int
    assert native.runtime.rtMemcpy.argtypes == [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
    ]
    assert native.runtime.rtMemcpy.restype is ctypes.c_int
    assert native.libc.memcpy.argtypes == [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ]
    assert native.libc.memcpy.restype is ctypes.c_void_p
    assert native.libc.memset.argtypes == [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
    ]
    assert native.libc.memset.restype is ctypes.c_void_p
    staging.copy_from_numpy(tensor, native.device)
    assert native.copied == tensor.tobytes()
    assert native.events[-2:] == [("memcpy", 48), ("h2d", native.device, 48, 1)]
    staging.clear()
    assert native.data == bytearray(48)
    staging.close()
    assert native.events[-2:] == [("memset", 48), ("free", native.pointer)]
    events = list(native.events)
    staging.close()
    assert native.events == events


@pytest.mark.parametrize("count", [0, -1, ctypes.c_size_t(-1).value + 1])
def test_native_input_rejects_invalid_size_before_allocation(native, count):
    with pytest.raises(ValueError):
        rpp._RppNativeInput(count)
    assert native.events == []


@pytest.mark.parametrize("array", [
    np.zeros(12, dtype=np.float64),
    np.zeros(12, dtype=np.uint8),
    np.zeros(11, dtype=np.float32),
    np.zeros((3, 4), dtype=np.float32).T,
])
def test_native_input_rejects_invalid_array_before_native_copy(native, array):
    staging = rpp._RppNativeInput(48)
    before = list(native.events)
    with pytest.raises(ValueError):
        staging.copy_from_numpy(array, native.device)
    assert native.events == before
    staging.close()


@pytest.mark.parametrize("pointer", [0, -1, 1 << (ctypes.sizeof(ctypes.c_void_p) * 8)])
def test_native_input_rejects_invalid_device_pointer_before_native_copy(native, pointer):
    staging = rpp._RppNativeInput(48)
    before = list(native.events)
    with pytest.raises(ValueError):
        staging.copy_from_numpy(np.zeros(12, dtype=np.float32), pointer)
    assert native.events == before
    staging.close()


@pytest.mark.parametrize("status,pointer", [(7, 0), (0, 0)])
def test_native_input_rejects_failed_or_null_allocation(native, status, pointer):
    native.alloc_status, native.pointer = status, pointer
    with pytest.raises(ModelUnavailableError):
        rpp._RppNativeInput(48)
    assert not any(event[0] in {"memcpy", "h2d", "free"} for event in native.events)


def test_native_input_rejects_missing_native_symbols_before_allocation(native, monkeypatch):
    monkeypatch.delattr(native.runtime, "rtMemcpy")
    with pytest.raises(ModelUnavailableError, match="unsupported"):
        rpp._RppNativeInput(48)
    assert native.events == []


def test_native_input_failed_copy_reports_failure_without_freeing_dma_buffer(native):
    staging = rpp._RppNativeInput(48)
    native.copy_status = 5
    with pytest.raises(ModelUnavailableError):
        staging.copy_from_numpy(np.ones(12, dtype=np.float32), native.device)
    assert native.events[-1] == ("h2d", native.device, 48, 1)
    assert not any(event[0] == "free" for event in native.events)
    # The execution engine owns quarantine after a failed native copy; do not
    # release staging while the runtime may still reference its memory.


def test_native_input_free_failure_is_not_retried_and_rejects_reuse(native):
    staging = rpp._RppNativeInput(48)
    staging.copy_from_numpy(np.ones(12, dtype=np.float32), native.device)
    native.free_status = 6
    with pytest.raises(ModelUnavailableError):
        staging.close()
    assert native.data == bytearray(48)
    before = list(native.events)
    staging.close()
    assert native.events == before
    with pytest.raises(ModelUnavailableError):
        staging.copy_from_numpy(np.zeros(12, dtype=np.float32), native.device)
    with pytest.raises(ModelUnavailableError):
        staging.clear()


def test_native_input_closed_buffer_cannot_be_copied_or_cleared(native):
    staging = rpp._RppNativeInput(48)
    staging.close()
    before = list(native.events)
    with pytest.raises(ModelUnavailableError):
        staging.copy_from_numpy(np.zeros(12, dtype=np.float32), native.device)
    with pytest.raises(ModelUnavailableError):
        staging.clear()
    assert native.events == before
