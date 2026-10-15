from __future__ import annotations

import os
import threading

import cv2
import numpy as np
import pytest

from person_search.config import Settings
from person_search.domain import SearchStatus, SourceConfig
from person_search.errors import SearchStopTimeoutError
from person_search.video import FramePacket, LatestFrameReader


def test_slow_ai_does_not_stop_latest_preview_frames(monkeypatch) -> None:
    """Five frames must refresh preview even when AI never consumes its queue."""
    class Capture:
        def __init__(self) -> None:
            self.next_id = 0

        def isOpened(self) -> bool:
            return True

        def read(self):
            if self.next_id == 5:
                return False, None
            frame = np.full((2, 2, 3), self.next_id, dtype=np.uint8)
            self.next_id += 1
            return True, frame

        def grab(self):
            raise AssertionError("AI backlog must not suppress fresh preview frames")

        def release(self):
            pass

    drops = []
    reader = LatestFrameReader(
        SourceConfig(type="file", uri="/test/video.mp4"),
        Settings(frame_queue_size=2),
        lambda *_: None,
        lambda: drops.append(1),
    )
    monkeypatch.setattr(reader, "_open", Capture)
    reader._capture_loop()

    latest = reader.latest()
    assert latest is not None and latest.frame_id == 4
    assert np.all(latest.frame == 4)
    assert reader.frames.qsize() == 2
    assert len(drops) == 3
    newest = reader.get(timeout=0)
    assert newest is not None and newest.frame_id == 4
    assert reader.get(timeout=0) is None
    # Every input is now either the selected AI frame or an exactly counted drop.
    assert len(drops) + 1 == 5
    metrics = reader.metrics_snapshot()
    assert metrics["decoded_frames"] == 5
    assert metrics["queue_length"] == 0
    assert metrics["capture_max_latency_ms"] >= metrics["capture_p50_latency_ms"] >= 0


def test_reader_rates_expire_and_do_not_derive_from_ai_drops(monkeypatch) -> None:
    reader = LatestFrameReader(
        SourceConfig(type="camera", device_index=0), Settings(), lambda *_: None, lambda: None
    )
    reader._capture_started_at = 0.0
    reader._decoded_frames = 3
    reader._decoded_at.extend([1.0, 2.0, 3.0])
    monkeypatch.setattr("person_search.video.time.monotonic", lambda: 5.0)
    assert reader.metrics_snapshot()["decode_fps"] == pytest.approx(3 / 5)
    monkeypatch.setattr("person_search.video.time.monotonic", lambda: 14.0)
    metrics = reader.metrics_snapshot()
    assert metrics["decode_fps"] == 0
    assert metrics["decoded_frames"] == 3
    assert metrics["last_decoded_frame_age_ms"] == 11000


def test_source_loss_clears_old_epoch_and_retries_with_backoff(monkeypatch) -> None:
    statuses = []
    drops = []
    waits = []
    reader = LatestFrameReader(
        SourceConfig(type="rtsp", uri="rtsp://camera.test/live"),
        Settings(), lambda status, _: statuses.append(status), lambda: drops.append(1),
    )
    reader._enqueue_latest(FramePacket(0, 0.0, np.zeros((2, 2, 3), dtype=np.uint8)))

    class Capture:
        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

    def wait(delay):
        waits.append(delay)
        if len(waits) == 3:
            reader._stop.set()

    monkeypatch.setattr(reader, "_open", Capture)
    monkeypatch.setattr(reader._stop, "wait", wait)
    reader._capture_loop()
    assert waits == [1.0, 2.0, 4.0]
    assert statuses == [SearchStatus.SOURCE_LOST]
    assert reader.frames.empty() and reader.latest() is None
    assert len(drops) == 1
    assert reader._source_epoch == 1


def test_rtsp_capture_uses_configured_ffmpeg_transport(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    capture = object()
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: calls.append(args) or capture)

    reader = LatestFrameReader(
        SourceConfig(type="rtsp", uri="rtsp://camera.test/live"),
        Settings(rtsp_transport="tcp"),
        lambda *_: None,
        lambda: None,
    )

    assert reader._open() is capture
    assert calls == [
        (
            "rtsp://camera.test/live",
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                5000,
            ],
        )
    ]
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"


def test_stop_raises_when_reader_thread_remains_alive() -> None:
    class StuckThread:
        def join(self, timeout: float) -> None:
            assert timeout == 3.0

        def is_alive(self) -> bool:
            return True

    reader = LatestFrameReader(
        SourceConfig(type="camera", device_index=0),
        Settings(),
        lambda *_: None,
        lambda: None,
    )
    reader._thread = StuckThread()  # type: ignore[assignment]

    with pytest.raises(SearchStopTimeoutError, match="video reader thread"):
        reader.stop()

    assert reader._stop.is_set()


def test_stop_interrupts_active_capture_blocked_in_read(monkeypatch) -> None:
    entered_read = threading.Event()
    interrupted = threading.Event()

    class BlockingCapture:
        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, None]:
            entered_read.set()
            assert interrupted.wait(timeout=1.0)
            return False, None

        def interrupt(self) -> None:
            interrupted.set()

        def release(self) -> None:
            interrupted.set()

    capture = BlockingCapture()
    monkeypatch.setattr("person_search.video.GStreamerCapture", lambda *_: capture)
    reader = LatestFrameReader(
        SourceConfig(type="rtsp", uri="rtsp://camera.test/live"),
        Settings(video_backend="gstreamer"),
        lambda *_: None,
        lambda: None,
    )

    reader.start()
    assert entered_read.wait(timeout=1.0)
    reader.stop()

    assert interrupted.is_set()
    assert reader.ended.is_set()


def test_concurrent_start_and_stop_never_join_before_thread_starts(monkeypatch) -> None:
    real_thread = threading.Thread
    start_entered = threading.Event()
    allow_start = threading.Event()
    joined = threading.Event()

    class ControlledThread:
        def __init__(self, **_kwargs: object) -> None:
            self.started = False

        def start(self) -> None:
            start_entered.set()
            assert allow_start.wait(timeout=1.0)
            self.started = True

        def join(self, timeout: float) -> None:
            assert timeout == 3.0
            assert self.started
            joined.set()

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr("person_search.video.threading.Thread", ControlledThread)
    reader = LatestFrameReader(
        SourceConfig(type="camera", device_index=0),
        Settings(),
        lambda *_: None,
        lambda: None,
    )
    starter = real_thread(target=reader.start)
    stopper = real_thread(target=reader.stop)

    starter.start()
    assert start_entered.wait(timeout=1.0)
    stopper.start()
    assert not joined.wait(timeout=0.05)
    allow_start.set()
    starter.join(timeout=1.0)
    stopper.join(timeout=1.0)

    assert joined.is_set()
    assert not starter.is_alive()
    assert not stopper.is_alive()
