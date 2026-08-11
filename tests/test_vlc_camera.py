from __future__ import annotations

from pathlib import Path

import pytest

from person_search.vlc_camera import VlcStreamConfig, build_vlc_command


def test_builds_expected_local_rtsp_command(tmp_path: Path) -> None:
    vlc = tmp_path / "VLC"
    vlc.touch()
    config = VlcStreamConfig(vlc_path=vlc)
    command = build_vlc_command(config)
    assert command[4] == "avcapture://0x8020000005ac8514"
    assert "sdp=rtsp://127.0.0.1:8554/camera" in command[5]
    assert "vcodec=h264" in command[5]


def test_rejects_unsafe_stream_path(tmp_path: Path) -> None:
    vlc = tmp_path / "VLC"
    vlc.touch()
    with pytest.raises(ValueError, match="stream path"):
        build_vlc_command(VlcStreamConfig(vlc_path=vlc, stream_path="../camera"))
