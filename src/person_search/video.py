from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from .config import Settings
from .domain import SearchStatus, SourceConfig, SourceType
from .errors import SearchStopTimeoutError
from .gstreamer_capture import GStreamerCapture

_READER_STOP_TIMEOUT_SECONDS = 3.0


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame_id: int
    captured_at: float
    frame: np.ndarray
    # Incremented whenever a reconnect starts. Consumers can discard a packet
    # that was decoded by the previous camera connection even if it was already
    # in flight when the reader reported SOURCE_LOST.
    source_epoch: int = 0


class LatestFrameReader:
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
        self.frames: queue.Queue[FramePacket] = queue.Queue(maxsize=settings.frame_queue_size)
        self._stop = threading.Event()
        self.ended = threading.Event()
        self._thread: threading.Thread | None = None
        self._lifecycle_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._active_capture: cv2.VideoCapture | GStreamerCapture | None = None
        self._source_epoch = 0
        self._latest_lock = threading.Lock()
        self._latest_packet: FramePacket | None = None
        self._conversion_provider = "not_started"
        self._conversion_fallback_reason: str | None = None
        self._metrics_lock = threading.Lock()
        self._capture_started_at: float | None = None
        self._decoded_frames = 0
        self._decoded_at: deque[float] = deque(maxlen=2400)
        self._capture_latencies_ms: deque[float] = deque(maxlen=1000)

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None:
                return
            thread = threading.Thread(
                target=self._capture_loop, name="frame-reader", daemon=True
            )
            # Publish and start while holding the same lock that stop() uses. This
            # removes both races from the old ordering: stop() cannot miss a
            # just-created thread, and it cannot call join() before start().
            self._thread = thread
            try:
                thread.start()
            except Exception:
                self._thread = None
                raise

    def get(self, timeout: float = 0.5) -> FramePacket | None:
        try:
            packet = self.frames.get(timeout=timeout)
        except queue.Empty:
            return None
        # Drain only a bounded number of packets. A fast producer must not keep
        # the consumer in an unbounded drain loop instead of doing inference.
        for _ in range(self.frames.maxsize - 1):
            try:
                newer = self.frames.get_nowait()
            except queue.Empty:
                break
            self.on_drop()
            packet = newer
        return packet

    def latest(self) -> FramePacket | None:
        """Return the freshest decoded packet without consuming the work queue."""
        with self._latest_lock:
            return self._latest_packet

    def metrics_snapshot(self) -> dict[str, int | float | None]:
        """Measure successful BGR reads, not sensor or raw RTP frame rates."""
        now = time.monotonic()
        with self._metrics_lock:
            timestamps = list(self._decoded_at)
            latencies = list(self._capture_latencies_ms)
            started_at = self._capture_started_at
            decoded_frames = self._decoded_frames
        elapsed = min(10.0, max(0.0, now - started_at)) if started_at is not None else 0.0
        count = sum(timestamp > now - 10.0 for timestamp in timestamps)
        return {
            "decode_fps": count / elapsed if elapsed > 0 else 0.0,
            "decoded_frames": decoded_frames,
            "capture_p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "capture_p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "capture_max_latency_ms": max(latencies, default=0.0),
            "queue_length": self.frames.qsize(),
            "queue_capacity": self.frames.maxsize,
            "last_decoded_frame_age_ms": (
                max(0.0, now - timestamps[-1]) * 1000.0 if timestamps else None
            ),
        }

    def _enqueue_latest(self, packet: FramePacket) -> None:
        # One producer, one AI consumer. Queue operations are atomic; callbacks
        # run after their mutex is released so view()/cleanup cannot deadlock.
        while True:
            try:
                self.frames.put_nowait(packet)
                return
            except queue.Full:
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    continue
                self.on_drop()

    @property
    def conversion_provider(self) -> str:
        return self._conversion_provider

    @property
    def conversion_fallback_reason(self) -> str | None:
        return self._conversion_fallback_reason

    def stop(self) -> None:
        self._stop.set()
        with self._capture_lock:
            capture = self._active_capture
        interrupt = getattr(capture, "interrupt", None)
        if callable(interrupt):
            interrupt()
        with self._lifecycle_lock:
            thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=_READER_STOP_TIMEOUT_SECONDS)
            if thread.is_alive():
                raise SearchStopTimeoutError(
                    "video reader thread did not exit within the stop timeout"
                )

    def clear(self) -> None:
        """Drop decoded frames that would otherwise keep camera buffers alive."""
        while True:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                break
            self.on_drop()
        with self._latest_lock:
            self._latest_packet = None

    def _capture_loop(self) -> None:
        capture: cv2.VideoCapture | GStreamerCapture | None = None
        reconnect_delay = 1.0
        frame_id = 0
        announced_lost = False
        with self._metrics_lock:
            self._capture_started_at = time.monotonic()
        try:
            while not self._stop.is_set():
                if capture is None or not capture.isOpened():
                    if capture is not None:
                        self._release_capture(capture)
                    capture = self._open()
                    self._publish_capture(capture)
                    self._update_conversion_status(capture)
                    if not capture.isOpened():
                        self._release_capture(capture)
                        capture = None
                        if not announced_lost:
                            self._source_epoch += 1
                            self.on_status(SearchStatus.SOURCE_LOST, "unable to open video source")
                            announced_lost = True
                        self._stop.wait(reconnect_delay)
                        reconnect_delay = min(
                            reconnect_delay * 2.0, self.settings.rtsp_reconnect_max_seconds
                        )
                        continue

                # Preview and AI share one decoded image, but AI backlog must
                # never suppress decoding/refreshing the preview's latest frame.
                read_started = time.monotonic()
                ok, frame = capture.read()
                captured_at = time.monotonic()
                if not ok or frame is None:
                    if self.source.type == SourceType.FILE:
                        self._stop.set()
                        break
                    self._release_capture(capture)
                    capture = None
                    self.clear()
                    if not announced_lost:
                        self._source_epoch += 1
                        self.on_status(SearchStatus.SOURCE_LOST, "video source read failed")
                        announced_lost = True
                    self._stop.wait(reconnect_delay)
                    reconnect_delay = min(
                        reconnect_delay * 2.0, self.settings.rtsp_reconnect_max_seconds
                    )
                    continue

                self._update_conversion_status(capture)

                reconnect_delay = 1.0
                if announced_lost or frame_id == 0:
                    self.on_status(SearchStatus.RUNNING, None)
                    announced_lost = False
                with self._metrics_lock:
                    self._decoded_frames += 1
                    self._decoded_at.append(captured_at)
                    self._capture_latencies_ms.append((captured_at - read_started) * 1000.0)
                packet = FramePacket(
                    frame_id=frame_id,
                    captured_at=captured_at,
                    frame=frame,
                    source_epoch=self._source_epoch,
                )
                with self._latest_lock:
                    self._latest_packet = packet
                frame_id += 1
                self._enqueue_latest(packet)
        finally:
            if capture is not None:
                self._release_capture(capture)
            self.ended.set()

    def _publish_capture(self, capture: cv2.VideoCapture | GStreamerCapture) -> None:
        with self._capture_lock:
            self._active_capture = capture
            should_interrupt = self._stop.is_set()
        if should_interrupt:
            interrupt = getattr(capture, "interrupt", None)
            if callable(interrupt):
                interrupt()

    def _release_capture(self, capture: cv2.VideoCapture | GStreamerCapture) -> None:
        with self._capture_lock:
            if self._active_capture is capture:
                self._active_capture = None
        capture.release()

    def _update_conversion_status(self, capture: object) -> None:
        self._conversion_provider = str(
            getattr(capture, "conversion_provider", "not_applicable")
        )
        reason = getattr(capture, "conversion_fallback_reason", None)
        self._conversion_fallback_reason = str(reason) if reason else None

    def _open(self) -> cv2.VideoCapture | GStreamerCapture:
        if self.source.type == SourceType.CAMERA:
            return cv2.VideoCapture(int(self.source.device_index))
        if self.source.type == SourceType.RTSP:
            if self.settings.video_backend == "gstreamer":
                return GStreamerCapture(self.source, self.settings)
            # OpenCV delegates RTSP to FFmpeg. Without an explicit transport,
            # FFmpeg commonly selects UDP and damaged H.264 frames persist until
            # the next keyframe when packets are lost.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.settings.rtsp_transport}"
            )
            return cv2.VideoCapture(str(self.source.uri), cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                int(self.settings.rtsp_open_timeout_seconds * 1000),
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                int(self.settings.rtsp_read_timeout_seconds * 1000),
            ])
        return cv2.VideoCapture(str(self.source.uri))
