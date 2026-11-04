"""Steps 4 and 5 end to end on a 3-frame 64x48 clip (dng_validate stubbed: it is built on Colab only)."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import tifffile

from video_v1 import check, convert, source
from video_v1.tests.conftest import TINY_HW, make_clip, needs_ffmpeg

pytest.importorskip("rawpy")
FRAMES = 3


@pytest.fixture
def converted(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "run_dng_validate", lambda path, binary: {"status": "passed", "error_lines": []})
    clip = source.probe(make_clip(tmp_path / "beach day.mkv", FRAMES))
    inputs = {"clip": clip, "tools": source.tools(), "dng_validate": None, "path": clip.path}
    run_dir = tmp_path / "run"
    summary = convert.run(run_dir, inputs, {"MAX_FRAMES": None}, log=lambda message: None)
    return run_dir, inputs, summary


def _samples(path) -> np.ndarray:
    return tifffile.imread(path)


@needs_ffmpeg
def test_every_frame_becomes_a_validated_dng(converted):
    run_dir, _inputs, summary = converted
    assert summary["frames_converted"] == FRAMES == summary["libraw_passed"] == summary["dng_validate_passed"]
    assert summary["validation_failures"] == 0 and "frame_count_mismatch" not in summary
    assert summary["clip_name"] == "beach_day" and summary["size"] == list(TINY_HW)
    folder = run_dir / summary["sequence_dir"]
    assert sorted(p.name for p in folder.glob("*.dng")) == [f"beach_day_{i:06d}.dng" for i in range(FRAMES)]
    manifest = json.loads((run_dir / summary["manifest"]).read_text())
    assert manifest["frames_written"] == FRAMES and manifest["frame_rate"] == "30000/1001"
    assert [f["timecode"] for f in manifest["frames"]] == ["00:00:00;00", "00:00:00;01", "00:00:00;02"]
    with open(run_dir / "timings.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == FRAMES and set(convert.STAGES) <= set(rows[0])
    estimate = summary["estimate_4k"]
    assert estimate["sr_tiles_per_frame"] == convert.sr_tiles(TINY_HW) == 1
    assert estimate["hours_for_clip_x8_ensemble"] > estimate["hours_for_clip_no_ensemble"] > 0


@needs_ffmpeg
def test_calibration_is_fixed_and_noise_is_fresh_per_frame(converted, tmp_path):
    run_dir, inputs, summary = converted
    calibration = json.loads((run_dir / "calibration.json").read_text())
    assert calibration["per_frame_estimation"] is False
    manifest = json.loads((run_dir / summary["manifest"]).read_text())
    assert [f["noise_seed"][1] for f in manifest["frames"]] == list(range(FRAMES))
    neutral = np.asarray(calibration["as_shot_neutral"])
    with tifffile.TiffFile(run_dir / summary["sequence_dir"] / "beach_day_000002.dng") as tif:
        stored = tif.pages[0].tags[50728].value
    np.testing.assert_allclose([n / d for n, d in zip(stored[::2], stored[1::2], strict=True)], neutral, rtol=1e-6)
    # The same frame from the same seed is reproducible; a different seed gives different noise.
    first = next(source.frames(inputs["clip"]))
    layout = convert.production.rggb_layout(TINY_HW)
    again = convert.raw_frame(first, neutral, layout, 0)
    np.testing.assert_array_equal(again, _samples(run_dir / summary["sequence_dir"] / "beach_day_000000.dng"))
    assert not np.array_equal(again, convert.raw_frame(first, neutral, layout, 1))


@needs_ffmpeg
def test_check_develops_every_frame_and_finds_no_flicker(converted):
    run_dir, inputs, _summary = converted
    result = check.run(run_dir, inputs, {}, log=lambda message: None)
    assert result["frames_checked"] == FRAMES and result["passed"], result["metrics"]
    assert (run_dir / result["preview"]).stat().st_size > 0
    assert len(result["thumbnails"]) == FRAMES
    stored = json.loads((run_dir / "check.json").read_text())
    assert stored["passed"] and len(stored["per_frame"]) == FRAMES


def _stats(level: float) -> dict:
    return {"src": {"rgb": [0.2, 0.2, 0.2], "luma": 0.2}, "dev": {"rgb": [0.2 * level] * 3, "luma": 0.2 * level}}


def test_stability_flags_a_level_jump_and_skips_dark_frames():
    levels = [1.0, 1.0, 1.03, 1.03]
    frames = [{"frame": i, "stats": _stats(v), "ratios": check.ratios(_stats(v))} for i, v in enumerate(levels)]
    result = check.stability(frames)
    assert not result["passed"] and result["metrics"]["level"]["worst_step_frame"] == 2
    assert result["metrics"]["level"]["over_step"] == [2] and result["metrics"]["red_green"]["passed"]
    assert check.stability(frames[:2])["passed"]
    dark = {"src": {"rgb": [0.001] * 3, "luma": 0.001}, "dev": {"rgb": [0.5] * 3, "luma": 0.5}}
    with_dark = [*frames[:2], {"frame": 2, "stats": dark, "ratios": check.ratios(dark)}]
    assert check.stability(with_dark)["passed"] and check.stability(with_dark)["frames_dark"] == 1
