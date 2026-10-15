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
from collections.abc import Sequence
from typing import Any

_HEADER = struct.Struct("!4sIIII")
_BGR_MAGIC = b"PSF1"
_NV12_MAGIC = b"PSN1"


def pack_nv12(
    data: bytes | bytearray | memoryview,
    width: int,
    height: int,
    strides: Sequence[int],
    offsets: Sequence[int],
) -> bytes:
    """Pack visible Y/UV rows using the decoder's declared layout, never its size.

    MPP allocations can include both row padding and extra rows before UV. The
    wire format has neither, so copying the first width * height * 3/2 bytes is
    only valid after verifying both strides and both plane offsets.
    """
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or width % 2
        or height % 2
        or width * height * 3 // 2 > 0xFFFFFFFF
    ):
        raise ValueError("NV12 dimensions must be positive even integers fitting the frame header")
    if len(strides) < 2 or len(offsets) < 2:
        raise ValueError("NV12 requires two plane strides and offsets")
    view = memoryview(data).cast("B")
    ranges = []
    for stride, offset, rows in zip(strides[:2], offsets[:2], (height, height // 2)):
        if type(stride) is not int or stride < width:
            raise ValueError("NV12 plane stride is smaller than the visible width")
        if type(offset) is not int or offset < 0:
            raise ValueError("NV12 plane offset must be a nonnegative integer")
        end = offset + (rows - 1) * stride + width
        if end > view.nbytes:
            raise ValueError(
                f"NV12 plane exceeds mapped buffer: offset={offset} stride={stride} "
                f"rows={rows} width={width} size={view.nbytes}"
            )
        ranges.append((offset, end))
    if ranges[0][0] < ranges[1][1] and ranges[1][0] < ranges[0][1]:
        raise ValueError("NV12 plane regions overlap")

    y_size = width * height
    expected_size = y_size * 3 // 2
    if tuple(strides[:2]) == (width, width) and tuple(offsets[:2]) == (0, y_size):
        # bytes(bytes) retains the original object for the ordinary packed path.
        return bytes(data) if view.nbytes == expected_size else bytes(view[:expected_size])

    # Join buffer views into the final bytes allocation once; constructing a
    # bytearray and then bytes would copy the entire decoded frame twice.
    pieces = []
    for stride, offset, rows in zip(strides[:2], offsets[:2], (height, height // 2)):
        if stride == width:
            pieces.append(view[offset : offset + width * rows])
        else:
            for row in range(rows):
                start = offset + row * stride
                pieces.append(view[start : start + width])
    return b"".join(pieces)


def pack_sample_nv12(sample: Any, gst: Any, gst_video: Any) -> tuple[int, int, bytes]:
    """Resolve per-buffer video metadata before the default caps layout."""
    caps = sample.get_caps()
    if caps is None or not caps.is_fixed() or caps.get_size() != 1:
        raise ValueError("NV12 sample must have one fixed video caps structure")
    structure = caps.get_structure(0)
    if structure.get_name() != "video/x-raw" or structure.get_value("format") != "NV12":
        raise ValueError("NV12 sample has an unsupported video format")
    width = structure.get_value("width")
    height = structure.get_value("height")
    buffer = sample.get_buffer()
    if buffer is None:
        raise ValueError("NV12 sample has no buffer")
    meta = gst_video.buffer_get_video_meta(buffer)
    if meta is not None:
        if (
            meta.format != gst_video.VideoFormat.NV12
            or meta.width != width
            or meta.height != height
            or meta.n_planes != 2
        ):
            raise ValueError("NV12 video metadata does not match the negotiated format/dimensions")
        strides, offsets = meta.stride, meta.offset
    else:
        # new_from_caps is available since GStreamer 1.20; boxes on 1.16 use
        # the mutable VideoInfo constructor and from_caps instead.
        if hasattr(gst_video.VideoInfo, "new_from_caps"):
            info = gst_video.VideoInfo.new_from_caps(caps)
        else:
            info = gst_video.VideoInfo()
            if not info.from_caps(caps):
                info = None
        if info is None or info.finfo.format != gst_video.VideoFormat.NV12:
            raise ValueError("NV12 caps could not describe a valid video layout")
        strides, offsets = info.stride, info.offset
    mapped, map_info = buffer.map(gst.MapFlags.READ)
    if not mapped:
        raise ValueError("GStreamer NV12 buffer could not be mapped")
    try:
        return width, height, pack_nv12(map_info.data, width, height, strides, offsets)
    finally:
        buffer.unmap(map_info)


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
    import cv2

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
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo
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
            try:
                width, height, payload = pack_sample_nv12(sample, Gst, GstVideo)
            except (TypeError, ValueError, OverflowError) as exc:
                print(f"invalid NV12 buffer layout: {exc}", file=sys.stderr)
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
