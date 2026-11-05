"""ffprobe metadata and the ffmpeg frame stream on tiny lossless clips (a few 64x48 frames)."""

from __future__ import annotations

import subprocess
from fractions import Fraction

import numpy as np
import pytest

from prd_v6 import colour
from video_v1 import source, ycbcr
from video_v1.tests.conftest import TINY_HW, make_clip, needs_ffmpeg


@pytest.mark.parametrize(
    ("pix_fmt", "target", "bits", "subsampling", "rgb"),
    [
        ("yuv420p", "yuv420p", 8, (2, 2), False),
        ("yuv420p10le", "yuv420p10le", 10, (2, 2), False),
        ("yuv422p10le", "yuv422p10le", 10, (2, 1), False),
        ("yuvj420p", "yuvj420p", 8, (2, 2), False),
        ("nv12", "yuv420p", 8, (2, 2), False),
        ("p010le", "yuv420p10le", 10, (2, 2), False),
        ("yuyv422", "yuv422p", 8, (2, 1), False),
        ("gbrp", "gbrp", 8, (1, 1), True),
        ("rgb24", "gbrp", 8, (1, 1), True),
        ("gbrp10le", "gbrp16le", 16, (1, 1), True),
    ],
)
def test_layout_keeps_the_decoder_samples(pix_fmt, target, bits, subsampling, rgb):
    layout = source.layout_for(pix_fmt)
    assert (layout.pix_fmt, layout.bits, layout.subsampling, layout.rgb) == (target, bits, subsampling, rgb)
    assert layout.full_range_format == (rgb or pix_fmt.startswith("yuvj"))


def test_layout_rejects_gray_and_converts_exotic_formats():
    with pytest.raises(source.SourceError):
        source.layout_for("gray")
    exotic = source.layout_for("yuv410p")
    assert exotic.pix_fmt == "yuv444p16le" and "swscale" in exotic.converted_by_ffmpeg


def test_rotation_from_display_matrix_and_legacy_tag():
    assert source.rotation_ccw({"side_data_list": [{"rotation": 90}]}) == 90
    assert source.rotation_ccw({"side_data_list": [{"rotation": -90}]}) == 270
    assert source.rotation_ccw({"tags": {"rotate": "90"}}) == 270  # the legacy tag is clockwise
    assert source.rotation_ccw({}) == 0
    with pytest.raises(source.SourceError):
        source.rotation_ccw({"side_data_list": [{"rotation": 45}]})


def test_even_crop():
    assert source.even_crop([1081, 1919]) == (1080, 1918)
    assert source.even_crop((1080, 1920)) == (1080, 1920)


@needs_ffmpeg
@pytest.mark.parametrize(("bits", "sub"), [(8, "420"), (10, "420"), (8, "422")])
def test_probe_and_frames_are_exact_on_a_lossless_clip(tmp_path, bits, sub):
    planes: list = []
    path = make_clip(tmp_path / "tiny.mkv", 3, bits=bits, sub=sub, planes_out=planes)
    clip = source.probe(path)
    assert (clip.codec, clip.frames, clip.size, clip.frame_rate) == ("ffv1", 3, list(TINY_HW), "30000/1001")
    assert not clip.signal["assumed"] and clip.signal["matrix"] == "bt709" and not clip.signal["full_range"]
    assert clip.audio_streams == [] and clip.rotation_ccw == 0
    signal = source.signal_of(clip)
    frames = list(source.frames(clip))
    assert len(frames) == 3
    for got, expected in zip(frames, planes, strict=True):
        assert got.dtype == np.float32 and got.shape == (*TINY_HW, 3)
        np.testing.assert_array_equal(got, ycbcr.frame_to_linear(expected, signal))


@needs_ffmpeg
def test_frame_limit_stops_ffmpeg_early(tmp_path):
    clip = source.probe(make_clip(tmp_path / "tiny.mkv", 4))
    assert len(list(source.frames(clip, limit=2))) == 2
    first = source.frames(clip)
    next(first)
    first.close()  # a consumer that stops early must not hang or raise


@needs_ffmpeg
def test_untagged_clip_is_decoded_as_sd_bt601_and_says_so(tmp_path):
    clip = source.probe(make_clip(tmp_path / "untagged.mkv", 2, tags=False, rate="25"))
    assert clip.signal["matrix"] == "smpte170m"  # 64x48 is SD: players assume BT.601
    assert {"matrix", "primaries", "transfer"} <= set(clip.signal["assumed"]) and not clip.signal["full_range"]
    assert clip.fps == Fraction(25)


@needs_ffmpeg
def test_audio_is_counted_and_never_decoded(tmp_path):
    path = make_clip(
        tmp_path / "with_audio.mkv",
        2,
        extra_inputs=("-f", "lavfi", "-t", "0.2", "-i", "sine=frequency=440"),
        extra_output=("-c:a", "pcm_s16le", "-shortest"),
    )
    clip = source.probe(path)
    assert [a["codec"] for a in clip.audio_streams] == ["pcm_s16le"]
    assert "-an" in source.decode_command(clip)
    assert len(list(source.frames(clip))) == clip.frames


@needs_ffmpeg
def test_rotation_matches_ffmpeg_autorotate(tmp_path):
    """A display-matrix clip comes out the way players show it (ffmpeg's own autorotate as reference)."""
    upright = make_clip(tmp_path / "upright.mp4", 1, codec=("-c:v", "libx264", "-qp", "0"))
    path = tmp_path / "rotated.mp4"  # a remux that only adds the display matrix, as phones record portrait
    subprocess.run(
        ["ffmpeg", "-v", "error", "-display_rotation", "90", "-i", str(upright), "-c", "copy", str(path)], check=True
    )
    clip = source.probe(path)
    assert clip.rotation_ccw == 90 and clip.size == [TINY_HW[1], TINY_HW[0]]
    ours = next(source.frames(clip))
    h, w = clip.size
    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"],
        capture_output=True,
        check=True,
    )
    reference = np.frombuffer(done.stdout, np.uint8).reshape(h, w).astype(np.float32)
    luma = colour.srgb_encode(ours) @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # Y' like ffmpeg's gray
    assert np.corrcoef(luma.ravel(), reference.ravel())[0, 1] > 0.98
