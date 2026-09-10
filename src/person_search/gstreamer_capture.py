"""Process bridge to the RK3588 system OpenCV/GStreamer capture stack.

The application runs on the project's Python 3.11 environment, but the box's
OpenCV 3.4 build is the one compiled with GStreamer and Rockchip MPP.  A small
framed stdout protocol keeps the two environments independent and gives the
existing ``LatestFrameReader`` the same ``read/grab/release`` surface as
``cv2.VideoCapture``.
"""

from __future__ import annotations

import logging
import struct
import subprocess
from pathlib import Path
from subprocess import Popen
from typing import BinaryIO

import numpy as np

from .config import Settings
from .domain import SourceConfig, SourceType

logger = logging.getLogger(__name__)

_FRAME_HEADER = struct.Struct("!4sIIII")
_FRAME_MAGIC = b"PSF1"
_MAX_FRAME_WIDTH = 16384
_MAX_FRAME_HEIGHT = 16384
_MAX_FRAME_BYTES = 512 * 1024 * 1024


class GStreamerCapture:
    """A ``VideoCapture``-compatible handle backed by the system helper."""

    def __init__(self, source: SourceConfig, settings: Settings):
        if source.type != SourceType.RTSP or not source.uri:
            raise ValueError("GStreamerCapture currently supports RTSP sources only")
        self.source = source
        self.settings = settings
        self._process: Popen[bytes] | None = None
        self._stdout: BinaryIO | None = None
        self.open_error: str | None = None
        self.command = self._build_command()
        try:
            self._process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self._stdout = self._process.stdout
        except OSError as exc:
            self.open_error = str(exc)
            logger.warning("failed to start GStreamer capture helper: %s", exc)

    def isOpened(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.isOpened() or self._stdout is None:
            return False, None
        header = _read_exact(self._stdout, _FRAME_HEADER.size)
        if header is None:
            self._close_process()
            return False, None
        magic, width, height, channels, payload_size = _FRAME_HEADER.unpack(header)
        if (
            magic != _FRAME_MAGIC
            or channels != 3
            or width <= 0
            or height <= 0
            or width > _MAX_FRAME_WIDTH
            or height > _MAX_FRAME_HEIGHT
            or payload_size != width * height * channels
            or payload_size > _MAX_FRAME_BYTES
        ):
            logger.warning("invalid frame header from GStreamer helper")
            self._close_process()
            return False, None
        payload = _read_exact(self._stdout, payload_size)
        if payload is None:
            self._close_process()
            return False, None
        frame = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, channels).copy()
        return True, frame

    def grab(self) -> bool:
        ok, _ = self.read()
        return ok

    def release(self) -> None:
        self._close_process()

    def _build_command(self) -> list[str]:
        helper = _resolve_helper(self.settings.gstreamer_helper)
        return [
            str(self.settings.gstreamer_python),
            str(helper),
            "--uri",
            str(self.source.uri),
            "--transport",
            self.settings.rtsp_transport,
            "--latency-ms",
            str(self.settings.gstreamer_latency_ms),
            "--codec",
            self.settings.gstreamer_rtsp_codec,
        ]

    def _close_process(self) -> None:
        process = self._process
        self._process = None
        self._stdout = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
                process.wait(timeout=1.0)
            except (OSError, subprocess.TimeoutExpired):
                logger.debug("GStreamer helper did not exit during cleanup", exc_info=True)


def _resolve_helper(configured: Path) -> Path:
    path = configured.expanduser()
    if path.is_file():
        return path
    if not path.is_absolute():
        repository_relative = Path(__file__).resolve().parents[2] / path
        if repository_relative.is_file():
            return repository_relative
    return path


def _read_exact(stream: BinaryIO, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


__all__ = ["GStreamerCapture"]
