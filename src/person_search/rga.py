"""Minimal Rockchip RGA NV12-to-BGR adapter with a safe CPU fallback boundary."""

from __future__ import annotations

import ctypes
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RK_FORMAT_BGR_888 = 0x7 << 8
RK_FORMAT_YCBCR_420_SP = 0xA << 8
IM_YUV_TO_RGB_BT601_LIMIT = 1 << 0
IM_STATUS_SUCCESS = 1
IM_STATUS_NOERROR = 2


class _ColorKeyRange(ctypes.Structure):
    _fields_ = [("max", ctypes.c_int), ("min", ctypes.c_int)]


class _NnInfo(ctypes.Structure):
    _fields_ = [
        ("scale_r", ctypes.c_int),
        ("scale_g", ctypes.c_int),
        ("scale_b", ctypes.c_int),
        ("offset_r", ctypes.c_int),
        ("offset_g", ctypes.c_int),
        ("offset_b", ctypes.c_int),
    ]


class _RgaBuffer(ctypes.Structure):
    """ABI of ``rga_buffer_t`` from librga 2.x ``im2d_type.h``."""

    _fields_ = [
        ("vir_addr", ctypes.c_void_p),
        ("phy_addr", ctypes.c_void_p),
        ("fd", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("wstride", ctypes.c_int),
        ("hstride", ctypes.c_int),
        ("format", ctypes.c_int),
        ("color_space_mode", ctypes.c_int),
        ("global_alpha", ctypes.c_int),
        ("rd_mode", ctypes.c_int),
        ("color", ctypes.c_int),
        ("colorkey_range", _ColorKeyRange),
        ("nn", _NnInfo),
        ("rop_code", ctypes.c_int),
        ("handle", ctypes.c_uint32),
    ]


class RgaUnavailableError(RuntimeError):
    """RGA could not be loaded, opened, or used for this buffer."""


class RgaNv12Converter:
    """Convert packed NV12 frames through librga using virtual-address buffers."""

    def __init__(self, library_path: Path | str):
        try:
            library = ctypes.CDLL(str(library_path))
            wrap = library.wrapbuffer_virtualaddr_t
            convert = library.imcvtcolor_t
        except (OSError, AttributeError) as exc:
            raise RgaUnavailableError(f"could not load librga: {type(exc).__name__}") from exc
        wrap.restype = _RgaBuffer
        wrap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        convert.restype = ctypes.c_int
        convert.argtypes = [
            _RgaBuffer,
            _RgaBuffer,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._wrap = wrap
        self._convert = convert

    def convert(self, nv12: np.ndarray, width: int, height: int) -> np.ndarray:
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("RGA NV12 dimensions must be positive and even")
        source = np.asarray(nv12, dtype=np.uint8)
        if source.size != width * height * 3 // 2:
            raise ValueError("RGA NV12 buffer size does not match its dimensions")
        source = np.ascontiguousarray(source.reshape(height * 3 // 2, width))
        destination = np.empty((height, width, 3), dtype=np.uint8)
        source_buffer = self._wrap(
            ctypes.c_void_p(source.ctypes.data),
            width,
            height,
            width,
            height,
            RK_FORMAT_YCBCR_420_SP,
        )
        destination_buffer = self._wrap(
            ctypes.c_void_p(destination.ctypes.data),
            width,
            height,
            width,
            height,
            RK_FORMAT_BGR_888,
        )
        status = self._convert(
            source_buffer,
            destination_buffer,
            RK_FORMAT_YCBCR_420_SP,
            RK_FORMAT_BGR_888,
            IM_YUV_TO_RGB_BT601_LIMIT,
            1,
        )
        if status not in (IM_STATUS_SUCCESS, IM_STATUS_NOERROR):
            raise RgaUnavailableError(f"librga color conversion failed with status {status}")
        return destination


__all__ = ["RgaNv12Converter", "RgaUnavailableError"]
