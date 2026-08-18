from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Settings
from .domain import SearchStatus, SourceConfig, SourceType


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame_id: int
    captured_at: float
    frame: np.ndarray


class LatestFrameReader:
    # A short first retry keeps transient camera startup failures responsive;
    # subsequent failures back off exponentially to avoid busy-looping a dead
    # RTSP endpoint on a small edge CPU.
    _INITIAL_RECONNECT_DELAY_SECONDS = 0.25

    def __init__(
        self,
        source: SourceConfig,
        settings: Settings,
        on_status: Callable[[SearchStatus, str | None], None],
        on_drop: Callable[[], None],
    ):
        self.source = source
        self.settings = settings
        self.on_status = on_status
        self.on_drop = on_drop
        # A non-positive setting would make ``queue.Queue`` unbounded, which
        # defeats the latest-frame contract and can exhaust an edge device's
        # memory when inference falls behind.
        self.frames: queue.Queue[FramePacket] = queue.Queue(
            maxsize=max(1, settings.frame_queue_size)
        )
        self._stop = threading.Event()
        self.ended = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture_lock = threading.Lock()
        self._capture: cv2.VideoCapture | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.ended.clear()
        self._thread = threading.Thread(target=self._capture_loop, name="frame-reader", daemon=True)
        self._thread.start()

    def get(self, timeout: float = 0.5) -> FramePacket | None:
        # Poll in short intervals so stop() can interrupt a consumer waiting on
        # an empty queue instead of making it wait for the full timeout.
        if self._stop.is_set():
            return None
        timeout = max(0.0, timeout)
        deadline = time.monotonic() + timeout
        while not self._stop.is_set():
            remaining = max(0.0, deadline - time.monotonic())
            try:
                packet = self.frames.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if remaining <= 0.0:
                    return None
                continue

            # The queue is deliberately bounded, but a consumer can still fall
            # behind briefly.  Return the newest packet and discard stale ones
            # so inference never spends time processing an old frame.
            while True:
                try:
                    newer = self.frames.get_nowait()
                except queue.Empty:
                    break
                packet = newer
                self.on_drop()
            return packet
        return None

    def stop(self) -> None:
        self._stop.set()
        self._release_capture()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)
        elif self._thread is None:
            self.ended.set()

    def _capture_loop(self) -> None:
        capture: cv2.VideoCapture | None = None
        reconnect_delay = self._INITIAL_RECONNECT_DELAY_SECONDS
        frame_id = 0
        announced_lost = False
        try:
            while not self._stop.is_set():
                if capture is None or not self._is_opened(capture):
                    self._release_capture(capture)
                    capture = None
                    try:
                        capture = self._open()
                        self._set_capture(capture)
                        opened = self._is_opened(capture)
                    except Exception as exc:  # noqa: BLE001 - reconnect loop must stay alive
                        opened = False
                        self._set_capture(None)
                        if not announced_lost:
                            self.on_status(SearchStatus.SOURCE_LOST, f"unable to open video source: {exc}")
                            announced_lost = True
                    if not opened:
                        self._release_capture(capture)
                        capture = None
                        if not announced_lost:
                            self.on_status(SearchStatus.SOURCE_LOST, "unable to open video source")
                            announced_lost = True
                        if self._wait_reconnect(reconnect_delay):
                            reconnect_delay = self._next_reconnect_delay(reconnect_delay)
                        continue

                try:
                    ok, frame = capture.read()
                except Exception:  # noqa: BLE001 - a failed read enters reconnect handling
                    ok, frame = False, None
                if not ok or frame is None:
                    if self.source.type == SourceType.FILE:
                        self._stop.set()
                        break
                    self._release_capture(capture)
                    capture = None
                    if not announced_lost:
                        self.on_status(SearchStatus.SOURCE_LOST, "video source read failed")
                        announced_lost = True
                    if self._wait_reconnect(reconnect_delay):
                        reconnect_delay = self._next_reconnect_delay(reconnect_delay)
                    continue

                frame = self._limit_frame_size(frame)

                # A successfully decoded frame proves the source is healthy;
                # restart the backoff at the fast initial interval.
                reconnect_delay = self._INITIAL_RECONNECT_DELAY_SECONDS
                if announced_lost or frame_id == 0:
                    self.on_status(SearchStatus.RUNNING, None)
                    announced_lost = False
                packet = FramePacket(frame_id=frame_id, captured_at=time.monotonic(), frame=frame)
                frame_id += 1
                if self.frames.full():
                    try:
                        self.frames.get_nowait()
                    except queue.Empty:
                        pass
                    self.on_drop()
                try:
                    self.frames.put_nowait(packet)
                except queue.Full:
                    self.on_drop()
        finally:
            self._release_capture(capture)
            self.ended.set()

    def _wait_reconnect(self, delay: float) -> bool:
        """Wait for a reconnect attempt, returning false when stopping."""

        return not self._stop.wait(max(0.0, delay))

    def _next_reconnect_delay(self, delay: float) -> float:
        maximum = max(0.0, self.settings.rtsp_reconnect_max_seconds)
        if maximum == 0.0:
            return 0.0
        return min(maximum, max(self._INITIAL_RECONNECT_DELAY_SECONDS, delay * 2.0))

    @staticmethod
    def _is_opened(capture: cv2.VideoCapture) -> bool:
        try:
            return bool(capture.isOpened())
        except Exception:  # noqa: BLE001 - backend failures are treated as a closed source
            return False

    def _set_capture(self, capture: cv2.VideoCapture | None) -> None:
        with self._capture_lock:
            self._capture = capture

    def _release_capture(self, expected: cv2.VideoCapture | None = None) -> None:
        """Detach and release a capture, safely handling concurrent stop calls."""

        with self._capture_lock:
            capture = self._capture if expected is None or self._capture is expected else None
            if capture is not None:
                self._capture = None
        # Release outside the lock: some backends block briefly while closing,
        # and stop() must not prevent the capture loop from progressing.
        if capture is not None:
            try:
                capture.release()
            except Exception:  # noqa: BLE001,S110 - release is best effort during shutdown
                pass

    def _open(self) -> cv2.VideoCapture:
        if self.source.type == SourceType.CAMERA:
            return cv2.VideoCapture(int(self.source.device_index))
        if self.source.type == SourceType.RTSP:
            if self.settings.capture_backend == "gstreamer":
                return self._open_gstreamer()
            # OpenCV delegates RTSP to FFmpeg. Without an explicit transport,
            # FFmpeg commonly selects UDP and damaged H.264 frames persist until
            # the next keyframe when packets are lost.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.settings.rtsp_transport}"
            )
            return cv2.VideoCapture(str(self.source.uri), cv2.CAP_FFMPEG)
        return cv2.VideoCapture(str(self.source.uri))

    def _open_gstreamer(self) -> cv2.VideoCapture:
        """Open an RTSP stream through a RK3588-friendly GStreamer pipeline.

        ``mppvideodec`` is the common Rockchip decoder name.  Vendor images may
        expose an equivalent plugin; in that case this method is the single
        place that needs adjusting.  Explicit ``gstreamer`` mode intentionally
        fails rather than silently switching to software decoding.
        """

        uri = (self.source.uri or "").replace("\\", "\\\\").replace('"', '\\"')
        codec = self.settings.gstreamer_rtsp_codec
        decoder = self.settings.gstreamer_decoder
        pipeline = (
            f'rtspsrc location="{uri}" protocols={self.settings.rtsp_transport} '
            f"latency={self.settings.gstreamer_latency_ms} ! "
            f"rtp{codec}depay ! {codec}parse ! {decoder} ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        cap_api = getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY)
        return cv2.VideoCapture(pipeline, cap_api)

    def _limit_frame_size(self, frame: np.ndarray) -> np.ndarray:
        max_width = int(self.settings.max_capture_width)
        max_height = int(self.settings.max_capture_height)
        if max_width <= 0 and max_height <= 0:
            return frame
        height, width = frame.shape[:2]
        ratios = [1.0]
        if max_width > 0:
            ratios.append(max_width / max(width, 1))
        if max_height > 0:
            ratios.append(max_height / max(height, 1))
        ratio = min(ratios)
        if ratio >= 1.0:
            return frame
        return cv2.resize(
            frame,
            (max(1, round(width * ratio)), max(1, round(height * ratio))),
            interpolation=cv2.INTER_AREA,
        )
