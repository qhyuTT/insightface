from __future__ import annotations

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

    def start(self) -> None:
        self._thread = threading.Thread(target=self._capture_loop, name="frame-reader", daemon=True)
        self._thread.start()

    def get(self, timeout: float = 0.5) -> FramePacket | None:
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)

    def _capture_loop(self) -> None:
        capture: cv2.VideoCapture | None = None
        reconnect_delay = 1.0
        frame_id = 0
        announced_lost = False
        try:
            while not self._stop.is_set():
                if capture is None or not capture.isOpened():
                    if capture is not None:
                        capture.release()
                    capture = self._open()
                    if not capture.isOpened():
                        if not announced_lost:
                            self.on_status(SearchStatus.SOURCE_LOST, "unable to open video source")
                            announced_lost = True
                        self._stop.wait(reconnect_delay)
                        reconnect_delay = min(
                            reconnect_delay * 2.0, self.settings.rtsp_reconnect_max_seconds
                        )
                        continue
                    reconnect_delay = 1.0

                ok, frame = capture.read()
                if not ok or frame is None:
                    if self.source.type == SourceType.FILE:
                        self._stop.set()
                        break
                    capture.release()
                    capture = None
                    if not announced_lost:
                        self.on_status(SearchStatus.SOURCE_LOST, "video source read failed")
                        announced_lost = True
                    continue

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
            if capture is not None:
                capture.release()
            self.ended.set()

    def _open(self) -> cv2.VideoCapture:
        if self.source.type == SourceType.CAMERA:
            return cv2.VideoCapture(int(self.source.device_index))
        return cv2.VideoCapture(str(self.source.uri))
