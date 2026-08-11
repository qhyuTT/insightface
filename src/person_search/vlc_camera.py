from __future__ import annotations

import argparse
import signal
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VLC = Path("/Applications/VLC.app/Contents/MacOS/VLC")
DEFAULT_DEVICE_UID = "0x8020000005ac8514"


@dataclass(frozen=True, slots=True)
class VlcStreamConfig:
    vlc_path: Path = DEFAULT_VLC
    device_uid: str = DEFAULT_DEVICE_UID
    host: str = "127.0.0.1"
    port: int = 8554
    stream_path: str = "camera"
    width: int = 1280
    height: int = 720
    fps: int = 15
    bitrate_kbps: int = 1200

    @property
    def rtsp_uri(self) -> str:
        return f"rtsp://{self.host}:{self.port}/{self.stream_path}"


def build_vlc_command(config: VlcStreamConfig) -> list[str]:
    if not config.vlc_path.is_file():
        raise ValueError(f"VLC executable not found: {config.vlc_path}")
    if not 1 <= config.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if not config.stream_path or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in config.stream_path):
        raise ValueError("stream path may contain only letters, numbers, '-' and '_'")
    if min(config.width, config.height, config.fps, config.bitrate_kbps) <= 0:
        raise ValueError("width, height, fps and bitrate must be positive")

    transcode = (
        "#transcode{"
        "vcodec=h264,"
        "venc=x264{preset=ultrafast,tune=zerolatency,keyint=30},"
        f"vb={config.bitrate_kbps},fps={config.fps},"
        f"width={config.width},height={config.height},acodec=none"
        "}:rtp{sdp="
        f"{config.rtsp_uri}"
        "}"
    )
    return [
        str(config.vlc_path),
        "-I",
        "dummy",
        "-vvv",
        f"avcapture://{config.device_uid}",
        f"--sout={transcode}",
        "--sout-keep",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a macOS camera as a local RTSP stream through VLC"
    )
    parser.add_argument("--vlc-path", type=Path, default=DEFAULT_VLC)
    parser.add_argument("--device-uid", default=DEFAULT_DEVICE_UID)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument("--stream-path", default="camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--bitrate-kbps", type=int, default=1200)
    parser.add_argument(
        "--print-only", action="store_true", help="print the VLC command without opening the camera"
    )
    args = parser.parse_args()
    config = VlcStreamConfig(
        vlc_path=args.vlc_path,
        device_uid=args.device_uid,
        host=args.host,
        port=args.port,
        stream_path=args.stream_path,
        width=args.width,
        height=args.height,
        fps=args.fps,
        bitrate_kbps=args.bitrate_kbps,
    )
    try:
        command = build_vlc_command(config)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"RTSP stream: {config.rtsp_uri}")
    print("VLC command:")
    print(" ".join(_quote(argument) for argument in command))
    if args.print_only:
        return
    if not _port_is_available(config.host, config.port):
        parser.error(f"{config.host}:{config.port} is already in use")

    print("Starting VLC. Press Ctrl+C to stop the stream.")
    process = subprocess.Popen(command)
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        try:
            return_code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            return_code = process.wait(timeout=3)
    if return_code not in (0, -signal.SIGINT):
        raise SystemExit(return_code)


def _port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex((host, port)) != 0


def _quote(value: str) -> str:
    if value and all(char.isalnum() or char in "/:._=-" for char in value):
        return value
    return repr(value)


if __name__ == "__main__":
    main()
