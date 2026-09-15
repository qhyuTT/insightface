#!/usr/bin/env python3
"""Decode one RTSP stream with the box's system OpenCV/GStreamer build.

Frames are written to stdout as:

    magic (4 bytes) + width + height + channels + payload_size (big-endian u32)
    followed by either BGR uint8 bytes (``PSF1``) or packed NV12 bytes
    (``PSN1``).  NV12 is converted to BGR by the application process, avoiding
    the RK3588 system OpenCV's slow 4K ``videoconvert`` path.

Do not send diagnostics to stdout: the parent process treats it as a binary
frame stream.
"""

from __future__ import annotations

import argparse
import struct
import sys

import cv2

_HEADER = struct.Struct("!4sIIII")
_BGR_MAGIC = b"PSF1"
_NV12_MAGIC = b"PSN1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True)
    parser.add_argument("--transport", choices=("tcp", "udp"), default="tcp")
    parser.add_argument("--latency-ms", type=int, default=500)
    parser.add_argument("--drop-on-latency", choices=("true", "false"), default="false")
    parser.add_argument("--output-format", choices=("bgr", "nv12"), default="bgr")
    parser.add_argument("--codec", choices=("auto", "h264", "h265"), default="auto")
    args = parser.parse_args()
    drop_on_latency = args.drop_on_latency == "true"
    if args.output_format == "nv12":
        return stream_nv12(
            args.uri,
            args.transport,
            args.latency_ms,
            args.codec,
            drop_on_latency,
        )
    return stream_bgr(
        args.uri,
        args.transport,
        args.latency_ms,
        args.codec,
        drop_on_latency,
    )


def stream_bgr(
    uri: str, transport: str, latency_ms: int, codec: str, drop_on_latency: bool
) -> int:
    pipeline = build_pipeline(uri, transport, latency_ms, codec, drop_on_latency, "bgr")
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
            _write_frame(_BGR_MAGIC, width, height, channels, frame.tobytes())
    except BrokenPipeError:
        return 0
    finally:
        capture.release()


def stream_nv12(
    uri: str, transport: str, latency_ms: int, codec: str, drop_on_latency: bool
) -> int:
    """Pull MPP's native NV12 buffers through GI and bridge them without conversion."""
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except (ImportError, ValueError) as exc:
        print(f"GStreamer GI is required for NV12 output: {exc}", file=sys.stderr)
        return 5

    Gst.init(None)
    try:
        pipeline = Gst.parse_launch(
            build_pipeline(uri, transport, latency_ms, codec, drop_on_latency, "nv12")
        )
    except Exception as exc:  # noqa: BLE001 - helper reports the concrete pipeline failure
        print(f"GStreamer NV12 pipeline could not be built: {exc}", file=sys.stderr)
        return 2
    sink = pipeline.get_by_name("sink")
    if sink is None:
        print("GStreamer NV12 pipeline has no appsink", file=sys.stderr)
        pipeline.set_state(Gst.State.NULL)
        return 2
    state_change = pipeline.set_state(Gst.State.PLAYING)
    if state_change == Gst.StateChangeReturn.FAILURE:
        print("GStreamer NV12 pipeline could not be started", file=sys.stderr)
        pipeline.set_state(Gst.State.NULL)
        return 2
    try:
        while True:
            sample = sink.emit("pull-sample")
            if sample is None:
                print("GStreamer NV12 pipeline returned no sample", file=sys.stderr)
                return 3
            structure = sample.get_caps().get_structure(0)
            width = int(structure.get_value("width"))
            height = int(structure.get_value("height"))
            expected_size = width * height * 3 // 2
            buffer = sample.get_buffer()
            mapped, map_info = buffer.map(Gst.MapFlags.READ)
            if not mapped:
                print("GStreamer NV12 buffer could not be mapped", file=sys.stderr)
                return 4
            try:
                payload = bytes(map_info.data)
            finally:
                buffer.unmap(map_info)
            if len(payload) != expected_size:
                print(
                    f"unexpected NV12 payload size: {len(payload)} != {expected_size}",
                    file=sys.stderr,
                )
                return 4
            _write_frame(_NV12_MAGIC, width, height, 1, payload)
    except BrokenPipeError:
        return 0
    finally:
        pipeline.set_state(Gst.State.NULL)


def _write_frame(magic: bytes, width: int, height: int, channels: int, payload: bytes) -> None:
    sys.stdout.buffer.write(_HEADER.pack(magic, width, height, channels, len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def build_pipeline(
    uri: str,
    transport: str,
    latency_ms: int,
    codec: str,
    drop_on_latency: bool = False,
    output_format: str = "bgr",
) -> str:
    # The URI is passed as a subprocess argument, not through a shell. Escaping
    # here is for the GStreamer launch-string parser only.
    escaped_uri = uri.replace("\\", "\\\\").replace('"', '\\"')
    source = (
        f'rtspsrc location="{escaped_uri}" protocols={transport} '
        f"latency={max(0, latency_ms)} "
        f"drop-on-latency={'true' if drop_on_latency else 'false'}"
    )
    if codec == "h264":
        decoder = "rtph264depay ! h264parse ! mppvideodec"
    elif codec == "h265":
        decoder = "rtph265depay ! h265parse ! mppvideodec"
    else:
        # mppvideodec is installed with a higher rank than software decoders on
        # the EIS100 image, so decodebin selects Rockchip MPP for supported codecs.
        decoder = "decodebin"
    if output_format == "nv12":
        return f"{source} ! {decoder} ! video/x-raw,format=NV12 ! " "appsink name=sink drop=true max-buffers=1 sync=false"
    if output_format != "bgr":
        raise ValueError(f"unsupported output format: {output_format}")
    return f"{source} ! {decoder} ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink drop=true max-buffers=1 sync=false"


if __name__ == "__main__":
    raise SystemExit(main())
