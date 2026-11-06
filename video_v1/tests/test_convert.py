"""Steps 4 and 5 on a 3-frame 64x48 clip (dng_validate stubbed: it is built on Colab only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from prd_v6 import colour, constants, pipeline
from video_v1 import check, convert, mlv, source
from video_v1.tests.conftest import TINY_HW, make_clip, needs_ffmpeg

pytest.importorskip("rawpy")
FRAMES = 3


def _convert(tmp_path, monkeypatch, settings: dict, clip_path=None):
    monkeypatch.setattr(convert, "run_dng_validate", lambda path, binary: {"status": "passed", "error_lines": []})
    clip = source.probe(clip_path or make_clip(tmp_path / "beach day.mkv", FRAMES))
    inputs = {"clip": clip, "tools": source.tools(), "dng_validate": None, "path": clip.path}
    run_dir = tmp_path / "run"
    summary = convert.run(run_dir, inputs, settings, log=lambda message: None)
    return run_dir, inputs, summary


@pytest.fixture
def converted(tmp_path, monkeypatch):
    return _convert(tmp_path, monkeypatch, {"WRITE_DNG": True})


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
    assert manifest["frames_written"] == manifest["source_frames"] == FRAMES and manifest["frame_rate"] == "30000/1001"
    assert [f["timecode"] for f in manifest["frames"]] == ["00:00:00;00", "00:00:00;01", "00:00:00;02"]
    with open(run_dir / "timings.csv") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == FRAMES and set(convert.STAGES) <= set(rows[0])
    assert summary["tiff_passed"] == FRAMES and summary["mlv"]["status"] == "passed", summary["mlv"]
    assert summary["mlv"]["files"] == ["mlv/beach_day.MLV"] and summary["mlv"]["bits"] == 14
    estimate = summary["estimate_4k"]
    assert estimate["sr_tiles_per_frame"] == convert.sr_tiles(TINY_HW) == 1
    assert estimate["hours_for_clip_x8_ensemble"] > estimate["hours_for_clip_no_ensemble"] > 0


def test_4k_estimate_scales_only_the_output_stages():
    seconds = {"decode": 1.0, "raw": 2.0, "write": 3.0, "libraw": 4.0, "dng_validate": 5.0, "develop": 0.0,
               "tiff": 0.0, "mlv": 0.0}  # fmt: skip
    estimate = convert.estimate_4k(seconds, (1080, 1920), 150)
    sr = convert.sr_tiles((1080, 1920)) * convert.HAT_L_SECONDS / convert.HAT_L_TILES
    assert estimate["pixel_ratio"] == 4.0
    assert estimate["source_stages_seconds_per_frame"] == 1.0
    assert estimate["output_stages_seconds_per_frame"] == 14.0 * 4
    assert estimate["seconds_per_frame_x8_ensemble"] == pytest.approx(sr + 57.0)
    assert estimate["seconds_per_frame_no_ensemble"] == pytest.approx(sr / 8 + 57.0)
    assert estimate["hours_for_clip_x8_ensemble"] == pytest.approx((sr + 57.0) * 150 / 3600)


@needs_ffmpeg
def test_calibration_is_fixed_and_noise_is_fresh_per_frame(converted, tmp_path):
    run_dir, inputs, summary = converted
    calibration = json.loads((run_dir / "calibration.json").read_text())
    assert calibration["per_frame_estimation"] is False
    manifest = json.loads((run_dir / summary["manifest"]).read_text())
    digest = convert._sha256(Path(inputs["clip"].path))
    assert summary["source_sha256"] == digest
    assert [f["noise_seed"] for f in manifest["frames"]] == [[6, digest, i] for i in range(FRAMES)]
    neutral = np.asarray(calibration["as_shot_neutral"])
    with tifffile.TiffFile(run_dir / summary["sequence_dir"] / "beach_day_000002.dng") as tif:
        stored = tif.pages[0].tags[50728].value
    np.testing.assert_allclose([n / d for n, d in zip(stored[::2], stored[1::2], strict=True)], neutral, rtol=1e-6)
    # The same frame from the same seed is reproducible; a different seed gives different noise.
    first = next(source.frames(inputs["clip"]))
    layout = convert.production.rggb_layout(TINY_HW)
    again = convert.raw_frame(first, neutral, layout, 0, digest)
    np.testing.assert_array_equal(again, _samples(run_dir / summary["sequence_dir"] / "beach_day_000000.dng"))
    assert not np.array_equal(again, convert.raw_frame(first, neutral, layout, 1, digest))
    # Another clip with the same frame index gets a different noise field.
    assert not np.array_equal(again, convert.raw_frame(first, neutral, layout, 0, "0" * 64))


def test_frame_seed_is_per_clip_and_reproducible():
    clip_a, clip_b = "ab" * 32, "cd" * 32
    draw = lambda seed: np.random.default_rng(seed).normal(size=64)  # noqa: E731
    np.testing.assert_array_equal(draw(convert.frame_seed(5, clip_a)), draw(convert.frame_seed(5, clip_a)))
    assert not np.array_equal(draw(convert.frame_seed(5, clip_a)), draw(convert.frame_seed(5, clip_b)))
    assert not np.array_equal(draw(convert.frame_seed(5, clip_a)), draw(convert.frame_seed(6, clip_a)))


def test_a_textured_frame_gets_the_full_camera_noise_at_mid_grey():
    """Regression (run 20261106T043244Z): fur and blur texture read as existing noise.

    v6's top-up fitted slopes far above the camera model and cancelled the shot noise, so every frame got
    only the read noise (variance about 60 DN16^2 at every level). The delivered variance at mid grey
    must match the A1 II model.
    """
    hw = (512, 768)
    rng = np.random.default_rng(3)
    levels = np.repeat(np.geomspace(0.003, 0.5, hw[1] // 32), 32).astype(np.float32)
    texture = np.exp(0.3 * rng.normal(size=hw)).astype(np.float32)  # fur-like, at every scale of signal
    linear = np.clip((levels[None, :] * texture)[..., None] * np.ones(3, np.float32), 0.0, 1.0)
    neutral = colour.daylight_neutral(pipeline.camera_matrix())
    layout = convert.production.rggb_layout(hw)
    noisy = convert.raw_frame(linear, neutral, layout, 0, "ab" * 32)
    clean, _ = pipeline.cfa_samples(linear, neutral, layout, noise_seed=None)
    signal = pipeline.stored_to_signal(clean)
    added = pipeline.stored_to_signal(noisy) - signal
    slope, read_variance = pipeline.camera_noise_dn16()
    mid_grey = 0.18 * constants.SIGNAL_16
    near = (signal > 0.7 * mid_grey) & (signal < 1.4 * mid_grey)
    assert near.sum() > 10_000
    assert added[near].var() / np.mean(slope * signal[near] + read_variance) == pytest.approx(1.0, abs=0.05)
    # v6's top-up on the same frame shows the defect: the texture cancels the shot noise.
    _, topped_up = pipeline.cfa_samples(linear, neutral, layout, noise_seed=6)
    assert (pipeline.stored_to_signal(topped_up) - signal)[near].var() < 2 * read_variance


@needs_ffmpeg
def test_the_source_white_is_the_neutral_and_the_dng_carries_the_baseline_exposure(converted):
    run_dir, _inputs, summary = converted
    calibration = json.loads((run_dir / "calibration.json").read_text())
    np.testing.assert_allclose(calibration["as_shot_neutral"], colour.daylight_neutral(pipeline.camera_matrix()))
    assert calibration["illuminant"] == "source white (D65)"
    with tifffile.TiffFile(run_dir / summary["sequence_dir"] / "beach_day_000000.dng") as tif:
        assert tuple(tif.pages[0].tags[50730].value) == (-1, 2)  # BaselineExposure -0.5 EV


@needs_ffmpeg
def test_tiffs_are_the_libraw_develop_of_each_frame(converted):
    run_dir, _inputs, summary = converted
    for index in range(FRAMES):
        name = f"beach_day_{index:06d}"
        with tifffile.TiffFile(run_dir / summary["tiff_dir"] / f"{name}.tif") as tif:
            page = tif.pages[0]
            assert page.dtype == np.uint16 and page.shape == (*TINY_HW, 3) and "InterColorProfile" in page.tags
            developed = page.asarray()
        expected = colour.to_uint16(pipeline.develop_cfa_dng(run_dir / summary["sequence_dir"] / f"{name}.dng"))
        np.testing.assert_array_equal(developed, expected)
    manifest = json.loads((run_dir / summary["tiff_manifest"]).read_text())
    assert [f["tiff"] for f in manifest["frames"]] == [f"beach_day_{i:06d}.tif" for i in range(FRAMES)]


@needs_ffmpeg
def test_the_mlv_holds_each_dng_mosaic_at_14_bits(converted):
    run_dir, _inputs, summary = converted
    clip = mlv.read(run_dir / summary["mlv"]["files"][0])
    assert (clip.info["black"], clip.info["white"], clip.info["cfa_pattern"]) == (512, 15360, "RGGB")
    assert clip.headers[0]["video_frames"] == FRAMES and (clip.headers[0]["fps_nom"], clip.headers[0]["fps_denom"]) == (
        30000,
        1001,
    )
    for index in range(FRAMES):
        dng_samples = _samples(run_dir / summary["sequence_dir"] / f"beach_day_{index:06d}.dng")
        np.testing.assert_array_equal(clip.frame(index), mlv.reduce_bits(dng_samples, 14))


@needs_ffmpeg
def test_by_default_no_dngs_are_kept_and_the_tiffs_and_mlv_are_the_same(converted, tmp_path, monkeypatch):
    run_dir, inputs, summary = converted
    (tmp_path / "no_dng").mkdir()
    other, _, without = _convert(tmp_path / "no_dng", monkeypatch, {}, clip_path=inputs["clip"].path)
    assert without["dng_written"] is False and without["sequence_dir"] is None and without["validation_failures"] == 0
    assert not (other / "dng").exists()
    for index in range(FRAMES):
        name = f"beach_day_{index:06d}.tif"
        np.testing.assert_array_equal(
            tifffile.imread(other / without["tiff_dir"] / name), tifffile.imread(run_dir / summary["tiff_dir"] / name)
        )
    assert (other / "mlv" / "beach_day.MLV").read_bytes() == (run_dir / "mlv" / "beach_day.MLV").read_bytes()


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
