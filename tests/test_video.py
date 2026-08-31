from __future__ import annotations

import os
import threading

import cv2
import pytest

from person_search.config import Settings
from person_search.domain import SourceConfig
from person_search.errors import SearchStopTimeoutError
from person_search.video import LatestFrameReader


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
