from __future__ import annotations

import os

import cv2

from person_search.config import Settings
from person_search.domain import SourceConfig
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
    assert calls == [("rtsp://camera.test/live", cv2.CAP_FFMPEG)]
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"
