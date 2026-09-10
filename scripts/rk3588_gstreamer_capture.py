#!/usr/bin/env python3
"""Decode one RTSP stream with the box's system OpenCV/GStreamer build.

Frames are written to stdout as:

    magic (4 bytes) + width + height + channels + payload_size (big-endian u32)
    followed by BGR uint8 bytes.

Do not send diagnostics to stdout: the parent process treats it as a binary
frame stream.
"""

from __future__ import annotations

import argparse
import struct
import sys

import cv2

_HEADER = struct.Struct("!4sIIII")
_MAGIC = b"PSF1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True)
    parser.add_argument("--transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--latency-ms", type=int, default=100)
    parser.add_argument("--codec", choices=("auto", "h264", "h265"), default="auto")
    args = parser.parse_args()
    pipeline = build_pipeline(args.uri, args.transport, args.latency_ms, args.codec)
    capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not capture.isOpened():
        print("GStreamer pipeline could not be opened", file=sys.stderr)
        return 2
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("GStreamer pipeline read failed", file=sys.stderr)
                return 3
            if frame.ndim != 3 or frame.shape[2] != 3:
                print(f"unexpected decoded frame shape: {frame.shape}", file=sys.stderr)
                return 4
            frame = frame if frame.flags.c_contiguous else frame.copy()
            height, width, channels = (int(value) for value in frame.shape)
            payload = frame.tobytes()
            sys.stdout.buffer.write(
                _HEADER.pack(_MAGIC, width, height, channels, len(payload))
            )
            sys.stdout.buffer.write(payload)
            sys.stdout.buffer.flush()
    except BrokenPipeError:
        return 0
    finally:
        capture.release()


def build_pipeline(uri: str, transport: str, latency_ms: int, codec: str) -> str:
    # The URI is passed as a subprocess argument, not through a shell. Escaping
    # here is for the GStreamer launch-string parser only.
    escaped_uri = uri.replace("\\", "\\\\").replace('"', '\\"')
    source = (
        f'rtspsrc location="{escaped_uri}" protocols={transport} '
        f"latency={max(0, latency_ms)} drop-on-latency=true"
    )
    if codec == "h264":
        decoder = "rtph264depay ! h264parse ! mppvideodec"
    elif codec == "h265":
        decoder = "rtph265depay ! h265parse ! mppvideodec"
    else:
        # mppvideodec is installed with a higher rank than software decoders on
        # the EIS100 image, so decodebin selects Rockchip MPP for supported codecs.
        decoder = "decodebin"
    return (
        f"{source} ! {decoder} ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false"
    )


if __name__ == "__main__":
    raise SystemExit(main())
