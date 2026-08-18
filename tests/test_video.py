from __future__ import annotations

import os

import cv2
import numpy as np

from person_search.config import Settings
from person_search.domain import SourceConfig
from person_search.video import FramePacket, LatestFrameReader


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
    assert calls == [("rtsp://camera.test/live", cv2.CAP_FFMPEG)]
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"


def test_rtsp_capture_can_use_gstreamer_pipeline(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    capture = object()
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: calls.append(args) or capture)

    reader = LatestFrameReader(
        SourceConfig(type="rtsp", uri="rtsp://camera.test/live"),
        Settings(capture_backend="gstreamer", rtsp_transport="tcp"),
        lambda *_: None,
        lambda: None,
    )

    assert reader._open() is capture
    assert len(calls) == 1
    assert calls[0][1] == getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY)
    assert "mppvideodec" in str(calls[0][0])
    assert "drop=true" in str(calls[0][0])


def test_gstreamer_pipeline_supports_h265_and_vendor_decoder(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: calls.append(args) or object())
    reader = LatestFrameReader(
        SourceConfig(type="rtsp", uri="rtsp://camera.test/live"),
        Settings(
            capture_backend="gstreamer",
            gstreamer_rtsp_codec="h265",
            gstreamer_decoder="v4l2h265dec",
            gstreamer_latency_ms=250,
        ),
        lambda *_: None,
        lambda: None,
    )

    reader._open()

    pipeline = str(calls[0][0])
    assert "latency=250" in pipeline
    assert "rtph265depay ! h265parse ! v4l2h265dec" in pipeline


def test_capture_frame_size_is_bounded_for_edge_memory() -> None:
    reader = LatestFrameReader(
        SourceConfig(type="camera", device_index=0),
        Settings(max_capture_width=640, max_capture_height=480),
        lambda *_: None,
        lambda: None,
    )
    resized = reader._limit_frame_size(np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert resized.shape[:2] == (360, 640)


def test_get_returns_newest_queued_frame_and_reports_stale_frames() -> None:
    dropped = 0

    def on_drop() -> None:
        nonlocal dropped
        dropped += 1

    reader = LatestFrameReader(
        SourceConfig(type="camera", device_index=0),
        Settings(frame_queue_size=3),
        lambda *_: None,
        on_drop,
    )
    for frame_id in range(3):
        reader.frames.put_nowait(
            FramePacket(frame_id, float(frame_id), np.full((2, 2, 3), frame_id, dtype=np.uint8))
        )

    packet = reader.get(timeout=0)
    assert packet is not None
    assert packet.frame_id == 2
    assert dropped == 2


def test_stop_releases_active_capture_even_without_started_thread() -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.released = False

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    reader = LatestFrameReader(
        SourceConfig(type="camera", device_index=0),
        Settings(),
        lambda *_: None,
        lambda: None,
    )
    reader._set_capture(capture)  # type: ignore[arg-type]
    reader.stop()
    assert capture.released
    assert reader.ended.is_set()


def test_reconnect_delay_is_exponential_and_bounded() -> None:
    reader = LatestFrameReader(
        SourceConfig(type="rtsp", uri="rtsp://camera.test/live"),
        Settings(rtsp_reconnect_max_seconds=1.0),
        lambda *_: None,
        lambda: None,
    )
    assert reader._next_reconnect_delay(0.25) == 0.5
    assert reader._next_reconnect_delay(0.75) == 1.0
    assert reader._next_reconnect_delay(1.0) == 1.0
