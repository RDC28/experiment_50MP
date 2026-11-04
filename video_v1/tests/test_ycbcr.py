"""Colour decode math: range, matrix, chroma siting, transfers and primaries (H.273, BT.709, BT.2100)."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from video_v1 import ycbcr
from video_v1.tests.conftest import encode_planes, needs_ffmpeg, tiny_rgb


def signal(**overrides) -> ycbcr.Signal:
    base = dict(
        matrix="bt709", primaries="bt709", transfer="bt709", full_range=False, bits=8, subsampling=(1, 1),
        chroma_location="none",
    )  # fmt: skip
    base.update(overrides)
    return ycbcr.Signal(**base)


@pytest.mark.parametrize(("bits", "black", "white", "neutral"), [(8, 16, 235, 128), (10, 64, 940, 512)])
def test_limited_range_levels(bits, black, white, neutral):
    dtype = np.uint8 if bits == 8 else np.uint16
    for level, expected in ((black, 0.0), (white, 1.0)):
        planes = [np.full((2, 2), v, dtype) for v in (level, neutral, neutral)]
        rgb = ycbcr.planes_to_rgb(planes, signal(bits=bits))
        np.testing.assert_allclose(rgb, expected, atol=1e-6)


def test_full_range_levels():
    for level, expected in ((0, 0.0), (255, 1.0)):
        planes = [np.full((2, 2), v, np.uint8) for v in (level, 128, 128)]
        rgb = ycbcr.planes_to_rgb(planes, signal(full_range=True))
        np.testing.assert_allclose(rgb, expected, atol=2e-3)  # full-range chroma 128 is 0.5/255 off zero


@pytest.mark.parametrize("matrix", ["bt709", "smpte170m", "bt2020nc"])
@pytest.mark.parametrize("bits", [8, 10])
def test_matrix_inverts_the_encoder(matrix, bits):
    rgb = tiny_rgb(0)
    planes = encode_planes(rgb, bits=bits, subsampling=(1, 1), matrix=matrix)
    decoded = ycbcr.planes_to_rgb(planes, signal(matrix=matrix, bits=bits))
    np.testing.assert_allclose(decoded, rgb, atol=3.0 / (219 * 2 ** (bits - 8)))


def test_gbr_planes_are_reordered_to_rgb():
    planes = [np.full((2, 2), v, np.uint8) for v in (51, 102, 255)]  # G, B, R
    rgb = ycbcr.planes_to_rgb(planes, signal(matrix=ycbcr.RGB_MATRIX, full_range=True))
    np.testing.assert_allclose(rgb[0, 0], [1.0, 0.2, 0.4], atol=1e-6)


@pytest.mark.parametrize(("location", "expected"), [("left", 45.0), ("center", 42.5), ("topleft", 45.0)])
def test_chroma_siting_places_samples_on_luma(location, expected):
    chroma = np.tile(np.arange(16, dtype=np.float32) * 10.0, (8, 1))  # a horizontal ramp, 10 per chroma sample
    up = ycbcr.upsample_chroma(chroma, (16, 32), (2, 2), location)
    # Luma column x reads chroma at (x - x0) / 2: x = 9 is sample 4.5 (co-sited left) or 4.25 (centred).
    assert up[8, 9] == pytest.approx(expected, abs=1e-3)
    assert up[8, 10] == pytest.approx(expected + 5.0, abs=1e-3)


def test_chroma_upsampling_keeps_constant_planes():
    up = ycbcr.upsample_chroma(np.full((4, 4), 0.3, np.float32), (8, 8), (2, 2), "left")
    np.testing.assert_allclose(up, 0.3, atol=1e-6)


def test_bt709_transfer_round_trip_and_toe():
    values = np.linspace(-0.05, 1.1, 200, dtype=np.float32)
    np.testing.assert_allclose(ycbcr.bt709_inverse_oetf(ycbcr.bt709_oetf(values.clip(0))), values.clip(0), atol=2e-6)
    assert ycbcr.bt709_inverse_oetf(np.float32(0.045)) == pytest.approx(0.01)
    assert ycbcr.bt709_inverse_oetf(np.float32(-0.045)) == pytest.approx(-0.01)  # the toe extends below zero


def test_hlg_and_pq_reference_points():
    assert ycbcr.hlg_inverse_oetf(np.float32(0.5)) == pytest.approx(1.0 / 12.0, rel=1e-5)
    assert ycbcr.hlg_inverse_oetf(np.float32(1.0)) == pytest.approx(1.0, rel=1e-5)
    assert ycbcr.pq_eotf_nits(np.float32(1.0)) == pytest.approx(10000.0, rel=1e-4)
    assert ycbcr.to_linear(np.float32(0.0), "smpte2084") == pytest.approx(0.0, abs=1e-6)


def test_primaries_conversion():
    np.testing.assert_allclose(ycbcr.to_bt709_matrix("bt709"), np.eye(3))
    m = ycbcr.to_bt709_matrix("bt2020")
    np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-9)  # D65 white stays white
    np.testing.assert_allclose(m[0], [1.6605, -0.5876, -0.0728], atol=2e-4)  # ITU-R BT.2087


def test_resolve_fills_untagged_hd_as_bt709_limited():
    s = ycbcr.resolve({}, width=1920, height=1080, bits=8, subsampling=(2, 2), rgb=False, full_range_format=False)
    assert (s.matrix, s.primaries, s.transfer, s.full_range, s.chroma_location) == (
        "bt709", "bt709", "bt709", False, "left",
    )  # fmt: skip
    assert set(s.assumed) == {"matrix", "primaries", "transfer", "range", "chroma_location"}


def test_resolve_sd_jpeg_and_unsupported_values():
    sd = ycbcr.resolve({}, width=720, height=480, bits=8, subsampling=(2, 2), rgb=False, full_range_format=False)
    assert (sd.matrix, sd.primaries) == ("smpte170m", "smpte170m")
    jpeg = ycbcr.resolve({}, width=1920, height=1080, bits=8, subsampling=(2, 2), rgb=False, full_range_format=True)
    assert jpeg.full_range and "range" not in jpeg.assumed
    odd = ycbcr.resolve(
        {"color_space": "bt2020c", "color_transfer": "log100", "color_range": "pc"},
        width=1920, height=1080, bits=10, subsampling=(2, 2), rgb=False, full_range_format=False,
    )  # fmt: skip
    assert odd.matrix == "bt2020nc" and odd.transfer == "bt709" and odd.full_range and len(odd.warnings) == 2


@needs_ffmpeg
def test_matrix_matches_ffmpeg_swscale():
    """An independent decoder (swscale, BT.709 limited range, 4:4:4) gives the same R'G'B'."""
    rgb = tiny_rgb(1)
    planes = encode_planes(rgb, bits=8, subsampling=(1, 1))
    h, w = planes[0].shape
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv444p", "-s", f"{w}x{h}", "-i", "pipe:0",
         "-vf", "scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=accurate_rnd+full_chroma_int",
         "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1"],
        input=b"".join(p.tobytes() for p in planes), capture_output=True, check=True,
    )  # fmt: skip
    theirs = np.frombuffer(done.stdout, "<u2").reshape(h, w, 3) / 65535.0
    ours = ycbcr.planes_to_rgb(planes, signal())
    np.testing.assert_allclose(ours, theirs, atol=1.5 / 255)
