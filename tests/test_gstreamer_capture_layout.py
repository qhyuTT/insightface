from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

_HELPER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "rk3588_gstreamer_capture.py")
)
pack_nv12 = _HELPER["pack_nv12"]
pack_sample_nv12 = _HELPER["pack_sample_nv12"]


def _padded_frame(width, height, strides, offsets, size):
    """Distinct synthetic Y/UV rows make padding or plane-offset errors visible."""
    data = bytearray(b"\xff" * size)
    expected = bytearray()
    for plane, rows in enumerate((height, height // 2)):
        for row in range(rows):
            value = (16 + row % 32) if plane == 0 else (128 + row % 32)
            visible = bytes([value]) * width
            start = offsets[plane] + row * strides[plane]
            data[start : start + width] = visible
            expected.extend(visible)
    return data, bytes(expected)


def test_packed_nv12_retains_bytes_fast_path():
    data = bytes(range(24))
    assert pack_nv12(data, 4, 4, [4, 4, 0, 0], [0, 16, 0, 0]) is data


def test_declared_packed_layout_allows_unused_allocation_tail():
    data = bytes(range(24)) + b"unused padding"
    assert pack_nv12(data, 4, 4, (4, 4), (0, 16)) == bytes(range(24))


@pytest.mark.parametrize(
    ("strides", "offsets", "size"),
    [
        ((8, 10), (3, 50), 64),  # Independent Y/UV row pitches and nonzero origins.
        ((4, 4), (2, 32), 48),  # Extra allocated Y rows before the UV origin.
        ((8, 8), (0, 32), 48),  # Per-row padding in both planes.
        ((4, 4), (12, 0), 28),  # Plane order comes from offsets, not allocation order.
    ],
)
def test_packs_visible_rows_from_declared_layout(strides, offsets, size):
    data, expected = _padded_frame(4, 4, strides, offsets, size)
    assert pack_nv12(data, 4, 4, strides, offsets) == expected


def test_actual_rk3588_1080p_layout_removes_384_byte_row_padding():
    # Regression for the observed MPP allocation; no image or camera data.
    width, height = 1920, 1080
    strides, offsets = (2304, 2304), (0, 2488320)
    data, expected = _padded_frame(width, height, strides, offsets, 3732480)
    assert offsets[1] == strides[0] * height
    assert len(data) == offsets[1] + strides[1] * (height // 2)
    packed = pack_nv12(data, width, height, strides, offsets)
    assert len(packed) == 3110400
    assert packed == expected
    assert bytes(data[:3110400]) != expected


@pytest.mark.parametrize(
    ("width", "height"),
    [(0, 4), (-2, 4), (3, 4), (4, 0), (4, 3), (True, 4), (4, "4"), (65536, 65536)],
)
def test_rejects_invalid_or_unrepresentable_dimensions(width, height):
    with pytest.raises(ValueError, match="dimensions"):
        pack_nv12(bytes(24), width, height, (4, 4), (0, 16))


@pytest.mark.parametrize(
    ("strides", "offsets", "size", "message"),
    [
        ((4,), (0, 16), 24, "two plane"),
        ((4, 4), (0,), 24, "two plane"),
        ((3, 4), (0, 16), 24, "stride"),
        ((4, -4), (0, 16), 24, "stride"),
        ((4, 4), (-1, 16), 24, "offset"),
        ((4, 4), (0, -1), 24, "offset"),
        ((4, 4), (0, 16), 23, "exceeds"),
        ((8, 4), (0, 32), 27, "exceeds"),
        ((4, 4), (0, 12), 24, "overlap"),
    ],
)
def test_rejects_invalid_plane_layouts(strides, offsets, size, message):
    with pytest.raises(ValueError, match=message):
        pack_nv12(bytes(size), 4, 4, strides, offsets)


class _Caps:
    def __init__(self, **overrides):
        self.values = {"format": "NV12", "width": 4, "height": 4, **overrides}

    def is_fixed(self):
        return True

    def get_size(self):
        return 1

    def get_structure(self, _index):
        return SimpleNamespace(get_name=lambda: "video/x-raw", get_value=self.values.get)


class _Buffer:
    def __init__(self, data, *, mapped=True):
        self.data = data
        self.mapped = mapped
        self.unmapped = False

    def map(self, _flags):
        return self.mapped, SimpleNamespace(data=self.data)

    def unmap(self, _info):
        self.unmapped = True


def _sample_and_gi(data, *, meta=None, caps=None, video_info=None, mapped=True):
    buffer = _Buffer(data, mapped=mapped)
    sample = SimpleNamespace(get_caps=lambda: caps or _Caps(), get_buffer=lambda: buffer)
    gst = SimpleNamespace(MapFlags=SimpleNamespace(READ=1))
    video = SimpleNamespace(
        VideoFormat=SimpleNamespace(NV12="NV12"),
        buffer_get_video_meta=lambda _buffer: meta,
        VideoInfo=video_info,
    )
    return sample, gst, video, buffer


def _meta(**overrides):
    return SimpleNamespace(
        **{
            "format": "NV12",
            "width": 4,
            "height": 4,
            "n_planes": 2,
            "stride": [8, 10, 0, 0],
            "offset": [3, 50, 0, 0],
            **overrides,
        }
    )


def test_per_buffer_metadata_takes_priority_over_caps_default_layout():
    data, expected = _padded_frame(4, 4, (8, 10), (3, 50), 64)

    def unexpected_fallback(_caps):
        pytest.fail("VideoInfo must not override per-buffer metadata")

    sample, gst, video, buffer = _sample_and_gi(
        data,
        meta=_meta(),
        video_info=SimpleNamespace(new_from_caps=unexpected_fallback),
    )
    assert pack_sample_nv12(sample, gst, video) == (4, 4, expected)
    assert buffer.unmapped


@pytest.mark.parametrize("legacy", [False, True])
def test_caps_video_info_fallback_supports_current_and_legacy_gstreamer(legacy):
    # The caps layout itself can have padding (e.g. a width not divisible by 4).
    data, expected = _padded_frame(6, 4, (8, 8), (0, 32), 48)
    info = SimpleNamespace(
        finfo=SimpleNamespace(format="NV12"), stride=[8, 8, 0, 0], offset=[0, 32, 0, 0]
    )

    class LegacyInfo:
        def from_caps(self, _caps):
            self.finfo, self.stride, self.offset = info.finfo, info.stride, info.offset
            return True

    api = LegacyInfo if legacy else SimpleNamespace(new_from_caps=lambda _caps: info)
    sample, gst, video, buffer = _sample_and_gi(data, caps=_Caps(width=6), video_info=api)
    assert pack_sample_nv12(sample, gst, video) == (6, 4, expected)
    assert buffer.unmapped


@pytest.mark.parametrize(
    "meta",
    [_meta(format="NV21"), _meta(width=8), _meta(height=8), _meta(n_planes=1)],
)
def test_invalid_metadata_fails_closed_instead_of_using_caps(meta):
    sample, gst, video, _buffer = _sample_and_gi(bytes(64), meta=meta)
    with pytest.raises(ValueError, match="metadata"):
        pack_sample_nv12(sample, gst, video)


def test_layout_validation_failure_still_unmaps_the_sample():
    sample, gst, video, buffer = _sample_and_gi(bytes(63), meta=_meta())
    with pytest.raises(ValueError, match="exceeds"):
        pack_sample_nv12(sample, gst, video)
    assert buffer.unmapped


def test_mapping_failure_does_not_unmap_an_unmapped_buffer():
    sample, gst, video, buffer = _sample_and_gi(bytes(64), meta=_meta(), mapped=False)
    with pytest.raises(ValueError, match="could not be mapped"):
        pack_sample_nv12(sample, gst, video)
    assert not buffer.unmapped


def test_non_nv12_caps_are_rejected():
    sample, gst, video, _buffer = _sample_and_gi(bytes(64), meta=_meta(), caps=_Caps(format="NV21"))
    with pytest.raises(ValueError, match="unsupported video format"):
        pack_sample_nv12(sample, gst, video)
