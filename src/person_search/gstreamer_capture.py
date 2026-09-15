"""Process bridge to the RK3588 system OpenCV/GStreamer capture stack.

The application runs on the project's Python 3.11 environment, but the box's
OpenCV 3.4 build is the one compiled with GStreamer and Rockchip MPP.  A small
framed stdout protocol keeps the two environments independent and gives the
existing ``LatestFrameReader`` the same ``read/grab/release`` surface as
``cv2.VideoCapture``.
"""

from __future__ import annotations

import logging
import re
import struct
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from subprocess import Popen
from typing import BinaryIO

import cv2
import numpy as np

from .config import Settings
from .domain import SourceConfig, SourceType
from .rga import RgaNv12Converter, RgaUnavailableError

logger = logging.getLogger(__name__)

_FRAME_HEADER = struct.Struct("!4sIIII")
_BGR_FRAME_MAGIC = b"PSF1"
_NV12_FRAME_MAGIC = b"PSN1"
_MAX_FRAME_WIDTH = 16384
_MAX_FRAME_HEIGHT = 16384
_MAX_FRAME_BYTES = 512 * 1024 * 1024
# The helper's stderr is the only place a GStreamer failure explains itself, and
# a reconnect storm with no reason attached is undiagnosable (22 reconnects in
# 40 minutes on the EIS100 box, none explained).  Keep a short tail, not a log.
_STDERR_TAIL_LINES = 12
_STDERR_JOIN_TIMEOUT_SECONDS = 0.5
_CREDENTIALS = re.compile(r"(rtsps?://[^/\s:@]+:)[^@\s]+@")


def _redact_credentials(value: str) -> str:
    return _CREDENTIALS.sub(r"\1***@", value)


class GStreamerCapture:
    """A ``VideoCapture``-compatible handle backed by the system helper."""

    def __init__(self, source: SourceConfig, settings: Settings):
        if source.type != SourceType.RTSP or not source.uri:
            raise ValueError("GStreamerCapture currently supports RTSP sources only")
        self.source = source
        self.settings = settings
        self._process: Popen[bytes] | None = None
        self._stdout: BinaryIO | None = None
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._stderr_thread: threading.Thread | None = None
        self._rga_converter: RgaNv12Converter | None = None
        self._converter_mode = settings.gstreamer_nv12_converter
        self._conversion_provider = "not_applicable"
        self._conversion_fallback_reason: str | None = None
        self.open_error: str | None = None
        self.command = self._build_command()
        if settings.gstreamer_output_format == "nv12":
            self._conversion_provider = "opencv"
            if settings.gstreamer_nv12_converter in {"auto", "rga"}:
                try:
                    self._rga_converter = RgaNv12Converter(settings.rga_runtime_library)
                    self._conversion_provider = (
                        "auto_pending" if self._converter_mode == "auto" else "rga"
                    )
                except RgaUnavailableError as exc:
                    self._conversion_provider = "opencv_fallback"
                    self._conversion_fallback_reason = str(exc)
                    logger.warning("RGA unavailable; falling back to OpenCV NV12 conversion: %s", exc)
        try:
            self._process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            self._stdout = self._process.stdout
            self._start_stderr_reader(self._process)
        except OSError as exc:
            self.open_error = str(exc)
            logger.warning("failed to start GStreamer capture helper: %s", exc)

    def isOpened(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def conversion_provider(self) -> str:
        return self._conversion_provider

    @property
    def conversion_fallback_reason(self) -> str | None:
        return self._conversion_fallback_reason

    def read(self) -> tuple[bool, np.ndarray | None]:
        packet = self._read_packet()
        if packet is None:
            return False, None
        magic, width, height, payload = packet
        if magic == _BGR_FRAME_MAGIC:
            frame = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
        else:
            nv12 = np.frombuffer(payload, dtype=np.uint8).reshape(height * 3 // 2, width)
            try:
                if self._rga_converter is not None:
                    try:
                        if self._converter_mode == "auto":
                            frame = self._select_fastest_converter(nv12, width, height)
                        else:
                            frame = self._rga_converter.convert(nv12, width, height)
                    except (RgaUnavailableError, ValueError) as exc:
                        self._rga_converter = None
                        self._conversion_provider = "opencv_fallback"
                        self._conversion_fallback_reason = str(exc)
                        logger.warning(
                            "RGA NV12 conversion failed; using OpenCV for this and later frames: %s",
                            exc,
                        )
                        frame = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
                else:
                    frame = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
            except cv2.error:
                logger.warning("failed to convert NV12 frame to BGR")
                self._close_process()
                return False, None
        return True, frame

    def _select_fastest_converter(
        self, nv12: np.ndarray, width: int, height: int
    ) -> np.ndarray:
        """Benchmark both paths once and retain the faster verified converter."""
        converter = self._rga_converter
        if converter is None:
            return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        rga_started = time.perf_counter()
        rga_frame = converter.convert(nv12, width, height)
        rga_seconds = time.perf_counter() - rga_started
        opencv_started = time.perf_counter()
        opencv_frame = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
        opencv_seconds = time.perf_counter() - opencv_started
        if rga_frame.shape != opencv_frame.shape:
            raise RgaUnavailableError("RGA output shape disagrees with OpenCV")
        # RGA and OpenCV use the same limited-range BT.601 transform. Integer
        # rounding may differ by one or two levels, so compare a sparse grid and
        # avoid allocating a second full-size difference image at startup.
        rga_sample = rga_frame[::64, ::64].astype(np.int16)
        opencv_sample = opencv_frame[::64, ::64].astype(np.int16)
        if np.max(np.abs(rga_sample - opencv_sample), initial=0) > 3:
            raise RgaUnavailableError("RGA color output disagrees with OpenCV")
        if rga_seconds < opencv_seconds * 0.95:
            self._converter_mode = "rga"
            self._conversion_provider = "rga"
            logger.info(
                "selected RGA NV12 conversion (%.1fms vs OpenCV %.1fms)",
                rga_seconds * 1000.0,
                opencv_seconds * 1000.0,
            )
            return rga_frame
        self._rga_converter = None
        self._converter_mode = "opencv"
        self._conversion_provider = "opencv_selected"
        self._conversion_fallback_reason = (
            f"OpenCV benchmark was faster ({opencv_seconds * 1000.0:.1f}ms vs "
            f"RGA {rga_seconds * 1000.0:.1f}ms)"
        )
        logger.info("%s", self._conversion_fallback_reason)
        return opencv_frame

    def grab(self) -> bool:
        """Consume one frame without converting it.

        ``LatestFrameReader`` calls this when its queue is already full: the frame
        is destined for eviction, so paying the 4K NV12->BGR conversion (19 ms on
        an A76) for it is pure loss.  The bytes still have to be drained, or the
        helper blocks on its stdout pipe.
        """
        return self._read_packet() is not None

    def release(self) -> None:
        self._close_process()

    def _read_packet(self) -> tuple[bytes, int, int, bytes] | None:
        if not self.isOpened() or self._stdout is None:
            return None
        header = _read_exact(self._stdout, _FRAME_HEADER.size)
        if header is None:
            self._close_process()
            return None
        magic, width, height, channels, payload_size = _FRAME_HEADER.unpack(header)
        if magic == _BGR_FRAME_MAGIC:
            valid = channels == 3 and payload_size == width * height * 3
        elif magic == _NV12_FRAME_MAGIC:
            valid = channels == 1 and payload_size == width * height * 3 // 2
        else:
            valid = False
        if (
            not valid
            or width <= 0
            or height <= 0
            or width > _MAX_FRAME_WIDTH
            or height > _MAX_FRAME_HEIGHT
            or payload_size > _MAX_FRAME_BYTES
        ):
            logger.warning("invalid frame header from GStreamer helper")
            self._close_process()
            return None
        payload = _read_exact(self._stdout, payload_size)
        if payload is None:
            self._close_process()
            return None
        return magic, width, height, payload

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
            "--drop-on-latency",
            str(self.settings.gstreamer_drop_on_latency).lower(),
            "--output-format",
            self.settings.gstreamer_output_format,
            "--codec",
            self.settings.gstreamer_rtsp_codec,
        ]

    def _start_stderr_reader(self, process: Popen[bytes]) -> None:
        stream = getattr(process, "stderr", None)
        if stream is None:
            return
        # A helper that is never drained blocks on a full stderr pipe, and a
        # GStreamer pipeline can be very chatty (RGA retries print per frame).
        thread = threading.Thread(
            target=self._drain_stderr,
            args=(stream,),
            name="gst-helper-stderr",
            daemon=True,
        )
        thread.start()
        self._stderr_thread = thread

    def _drain_stderr(self, stream: BinaryIO) -> None:
        try:
            for raw in iter(stream.readline, b""):
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    self._stderr_tail.append(_redact_credentials(line))
        except (OSError, ValueError):
            logger.debug("GStreamer helper stderr reader stopped", exc_info=True)

    def _close_process(self) -> None:
        process = self._process
        self._process = None
        self._stdout = None
        if process is None:
            return
        # Exited on its own before we asked: that is the interesting case, and the
        # stderr tail is the only explanation the reconnect loop will ever get.
        exited_alone = process.poll() is not None
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
        thread = self._stderr_thread
        self._stderr_thread = None
        if thread is not None:
            thread.join(timeout=_STDERR_JOIN_TIMEOUT_SECONDS)
        tail = list(self._stderr_tail)
        self._stderr_tail.clear()
        if not tail:
            return
        message = "GStreamer capture helper %s (rc=%s); last stderr lines: %s"
        if exited_alone:
            logger.warning(message, "exited", process.poll(), " | ".join(tail))
        else:
            logger.debug(message, "released", process.poll(), " | ".join(tail))


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
