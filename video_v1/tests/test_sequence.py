"""The CinemaDNG-style frame sequence: names, SMPTE time codes and the frame DNG tag set."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import tifffile

from prd_v6 import dng as v6dng
from prd_v6 import pipeline
from video_v1 import sequence
from video_v1.tests.conftest import TINY_HW

NTSC = Fraction(30000, 1001)
FRAME_TAGS = {37393, 51043, 51044}  # ImageNumber, TimeCodes, FrameRate
PER_FILE_TAGS = {270, 273, 305}  # ImageDescription and Software name the frame; StripOffsets follow them


def test_clip_and_frame_names():
    assert sequence.clip_name("/content/My Clip (final).MOV") == "My_Clip_final"
    assert sequence.clip_name("/content/....mp4") == "clip"
    assert len(sequence.clip_name("x" * 90 + ".mp4")) == 40
    assert sequence.frame_name("beach", 7) == "beach_000007.dng"


@pytest.mark.parametrize(
    ("index", "rate", "text"),
    [
        (0, Fraction(25), "00:00:00:00"),
        (26, Fraction(25), "00:00:01:01"),
        (90_000, Fraction(25), "01:00:00:00"),
        (1799, NTSC, "00:00:59;29"),
        (1800, NTSC, "00:01:00;02"),  # frame numbers 00 and 01 are dropped each minute
        (17_982, NTSC, "00:10:00;00"),  # but not every tenth minute
        (59, Fraction(24000, 1001), "00:00:02:11"),  # 23.976 counts at 24 without drops
    ],
)
def test_timecode(index, rate, text):
    assert sequence.timecode_text(index, rate) == text


def test_timecode_bytes_are_bcd_with_the_drop_flag():
    assert sequence.timecode_bytes(1800, NTSC) == (0x42, 0x00, 0x01, 0x00, 0, 0, 0, 0)
    assert sequence.timecode_bytes(25 * 59 + 1, Fraction(25)) == (0x01, 0x59, 0x00, 0x00, 0, 0, 0, 0)


def test_no_timecode_above_30_fps():
    assert sequence.timecode(5, Fraction(60)) is None
    assert sequence.frame_tags(5, Fraction(60)) == [
        (37393, "I", 1, (5,), False),
        (51044, "2i", 1, (60, 1), False),
    ]


def _tags(path) -> dict:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        return {tag.code: tag.value for tag in page.tags.values()}, page.asarray()


def test_frame_dng_is_the_v6_dng_plus_frame_tags(tmp_path):
    rng = np.random.default_rng(3)
    samples = rng.integers(512, 60000, size=(16, 24), dtype=np.uint16)
    neutral = np.array([0.5, 1.0, 0.7])
    matrix = pipeline.camera_matrix()
    v6_path = v6dng.write_dng(tmp_path / "v6.dng", samples, cfa_pattern="RGGB", color_matrix=matrix,
                              as_shot_neutral=neutral)  # fmt: skip
    frame_path = sequence.write_frame(
        tmp_path / "f.dng", samples, index=1800, frames=2000, clip="beach", rate=NTSC, color_matrix=matrix,
        as_shot_neutral=neutral,
    )  # fmt: skip
    v6_tags, v6_samples = _tags(v6_path)
    frame_tags, frame_samples = _tags(frame_path)
    np.testing.assert_array_equal(frame_samples, v6_samples)
    assert set(frame_tags) - set(v6_tags) == FRAME_TAGS
    for code in set(v6_tags) - PER_FILE_TAGS:
        assert np.array_equal(np.asarray(frame_tags[code]), np.asarray(v6_tags[code])), code
    assert frame_tags[37393] == 1800
    assert tuple(frame_tags[51044]) == (30000, 1001)
    assert bytes(frame_tags[51043]) == bytes((0x42, 0x00, 0x01, 0x00, 0, 0, 0, 0))
    assert "Frame 1800 of 2000" in frame_tags[270] and frame_tags[305] == sequence.SOFTWARE


def test_frame_dng_opens_in_libraw(tmp_path):
    pytest.importorskip("rawpy")
    samples = np.full(TINY_HW, 4000, np.uint16)  # LibRaw rejects images under 22 pixels a side
    path = sequence.write_frame(
        tmp_path / "f.dng", samples, index=0, frames=1, clip="c", rate=Fraction(25),
        color_matrix=pipeline.camera_matrix(), as_shot_neutral=[0.5, 1.0, 0.7],
    )  # fmt: skip
    record = v6dng.libraw_check(path, expected_hw=TINY_HW, cfa_pattern="RGGB")
    assert record["status"] == "passed", record


def test_baseline_exposure_is_tagged_and_leaves_the_libraw_develop_unchanged(tmp_path):
    pytest.importorskip("rawpy")
    rng = np.random.default_rng(5)
    samples = rng.integers(2048, 40000, size=TINY_HW, dtype=np.uint16)
    kwargs = dict(index=0, frames=1, clip="c", rate=Fraction(25), color_matrix=pipeline.camera_matrix(),
                  as_shot_neutral=[0.5, 1.0, 0.7])  # fmt: skip
    plain = sequence.write_frame(tmp_path / "plain.dng", samples, **kwargs)
    placed = sequence.write_frame(tmp_path / "placed.dng", samples, baseline_exposure=-0.5, **kwargs)
    assert tuple(_tags(placed)[0][50730]) == (-1, 2)
    np.testing.assert_array_equal(pipeline.develop_cfa_dng(placed), pipeline.develop_cfa_dng(plain))


def test_write_frame_rejects_odd_or_wrong_samples(tmp_path):
    kwargs = dict(index=0, frames=1, clip="c", rate=Fraction(25), color_matrix=np.eye(3), as_shot_neutral=[1, 1, 1])
    with pytest.raises(TypeError):
        sequence.write_frame(tmp_path / "a.dng", np.zeros((4, 4), np.float32), **kwargs)
    with pytest.raises(ValueError):
        sequence.write_frame(tmp_path / "b.dng", np.zeros((5, 4), np.uint16), **kwargs)
