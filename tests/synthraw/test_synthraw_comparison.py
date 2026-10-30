"""PRD_v3 2K-versus-4K source-resolution comparison on two tiny end-to-end runs."""

import json
import re
import shutil
from datetime import datetime

import numpy as np
import pytest
import tifffile
from PIL import Image

from synthraw.comparison import (
    COMPARISON_JSON,
    COMPARISON_NAME,
    ESTIMATOR_UNSTABLE,
    USER_STUDY_KEY,
    USER_STUDY_NAME,
    CropBox,
    SourceResolutionComparator,
    UserStudyPairSheet,
    compare_and_report,
    detail_energy,
    dng_statistics,
    full_resolution_window,
    grid_crops,
    image_agreement,
    make_downsampled_source,
    parse_crops,
)
from synthraw.config import ExperimentConfig
from synthraw.pipeline import run_pipeline
from synthraw.provenance import SCIENTIFIC_BOUNDARY, SYNTHETIC_CAPTURE_DISCLOSURE


def _run(source, out, config):
    return run_pipeline(source, config, out, timestamp=datetime(2026, 1, 2, 3, 4, 5), fast_dev_max_side=96).root


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("comparison_runs")
    rows, cols = np.mgrid[0:40, 0:96]
    pixels = np.stack([255 * cols / 95, 255 * rows / 39, 128 + 100 * np.sin(cols / 6)], axis=-1)
    high = tmp_path / "source_4k.png"
    Image.fromarray(np.clip(np.round(pixels), 0, 255).astype(np.uint8)).save(high)
    low = make_downsampled_source(high, tmp_path / "source_2k.png")
    config = ExperimentConfig(OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL", FAST_DEV_RUN=True, SR_BACKEND="LANCZOS")
    return _run(low, tmp_path / "low", config), _run(high, tmp_path / "high", config)


@pytest.fixture(scope="module")
def reported(runs, tmp_path_factory):
    low, high = runs
    out = tmp_path_factory.mktemp("comparison_out")
    # Passed high first: the comparator orders runs by source pixel count itself.
    return compare_and_report(high, low, out, low_source_derived=True)


def _copy(root, tmp_path, name):
    return shutil.copytree(root, tmp_path / name / root.name)


def _edit_records(path, **values):
    records = json.loads(path.read_text())
    for record in records:
        if record["field"] in values:
            record["value"] = values[record["field"]]
    path.write_text(json.dumps(records))


def test_tiny_end_to_end_comparison_writes_a_self_contained_report(runs, reported):
    low, high = runs
    result, path = reported
    assert path.name == COMPARISON_NAME and (path.parent / COMPARISON_JSON).is_file()
    assert result.low.root == low and result.high.root == high
    assert result.low.source_size == (48, 20) and result.high.source_size == (96, 40)
    assert result.status == "COMPLETED" and result.flags == []
    names = [check.name for check in result.checks]
    assert names == [
        "source_pairing",
        "locked_settings",
        "estimator_exposure",
        "estimator_camera_geometry",
        "estimator_white_balance",
        "final_difference_map",
        "edge_agreement",
        "texture_agreement",
        "detail_above_low_source_nyquist",
    ]
    locked = result.checks[1]
    assert locked.passed and locked.measured["settings_compared"] > 100 and not result.locked_mismatches
    difference = result.checks[5].measured
    assert 0 < difference["mean_abs_difference"] < 0.1 and 0.9 < difference["ssim"] <= 1
    assert len(result.crops) == 6 and all(np.isfinite(crop["metrics"]["ssim"]) for crop in result.crops)
    # Crops are analysed at full resolution (factor 1), so each window is a slice of the master itself.
    for crop in result.crops:
        x0, y0, x1, y1 = crop["window_pixels"]
        assert crop["analysis_factor"] == 1 and crop["analysed_shape"] == [y1 - y0, x1 - x0]
        assert crop["box_pixels"][0] <= x0 < x1 <= crop["box_pixels"][2]
    detail = result.checks[8].measured
    assert detail["low_source_nyquist_cycles_per_px"] < detail["high_source_nyquist_cycles_per_px"] <= 0.5
    assert 0 < detail["high_above_low_source_nyquist"] < 1
    assert {row["mask"] for row in result.hdr_masks} >= {
        "generated_hdr_area_percentage",
        "Highlight radiance fabricated",
    }
    assert set(result.dng_stats["low"]) == {"0_R", "1_G", "2_G", "3_B"}

    text = path.read_text()
    assert "COMPARISON COMPLETED" in text and 'class="banner flag">ESTIMATOR_UNSTABLE' not in text
    assert "Lanczos-downsampled copy" in text
    assert SYNTHETIC_CAPTURE_DISCLOSURE in text and SCIENTIFIC_BOUNDARY in text
    for heading in (
        "Final 50 MP difference map",
        "Edge agreement",
        "Named crops (full resolution, 1:1)",
        "Exposure posterior",
        "HDR and confidence masks",
        "DNG statistics",
    ):
        assert heading in text
    sources = re.findall(r'src="([^"]+)"', text)
    assert len(sources) == 5 + 3 * 6 and all(src.startswith("data:image/png;base64,") for src in sources)
    assert not re.search(r'(href|src)="(https?:|file:|/)', text) and "<script" not in text
    assert "None" not in text and "—" not in text
    saved = json.loads((path.parent / COMPARISON_JSON).read_text())
    assert saved["status"] == "COMPLETED" and saved["low_source_derived"] is True and len(saved["crops"]) == 6


def test_locked_setting_mismatch_fails_the_comparison(runs, tmp_path):
    low, high = runs
    changed = _copy(low, tmp_path, "changed")
    config = json.loads((changed / "metadata" / "config.json").read_text())
    config["SENSOR_NOISE_SEED"] += 1
    (changed / "metadata" / "config.json").write_text(json.dumps(config))
    _edit_records(changed / "metadata" / "optical_parameters.json", lens_sigma_px=0.9)
    result, path = compare_and_report(changed, high, tmp_path / "out")
    assert result.status == "FAILED"
    assert {"config.SENSOR_NOISE_SEED", "lens.lens_sigma_px"} <= set(result.locked_mismatches)
    assert result.locked_mismatches["lens.lens_sigma_px"] == (0.9, 0.5)
    text = path.read_text()
    assert "COMPARISON FAILED" in text and "COMPARISON COMPLETED" not in text
    assert "config.SENSOR_NOISE_SEED" in text


def test_unstable_camera_estimate_is_flagged(runs, tmp_path):
    low, high = runs
    changed = _copy(high, tmp_path, "unstable")
    _edit_records(changed / "metadata" / "exposure_posterior.json", PhotographicSensitivity=400)
    _edit_records(changed / "metadata" / "camera_geometry.json", FocalLength=50.0)
    result, path = compare_and_report(low, changed, tmp_path / "out")
    assert result.flags == [ESTIMATOR_UNSTABLE]
    assert result.status == "COMPLETED"  # an unstable estimator is flagged, not a broken experiment
    by_name = {check.name: check for check in result.checks}
    assert by_name["estimator_exposure"].passed is False
    assert by_name["estimator_exposure"].measured["PhotographicSensitivity"] == pytest.approx(2.0)
    assert by_name["estimator_camera_geometry"].measured["unstable"] == ["FocalLength"]
    assert by_name["estimator_white_balance"].passed is True
    text = path.read_text()
    assert f'class="banner flag">{ESTIMATOR_UNSTABLE}' in text and "PhotographicSensitivity, FocalLength" in text


def test_loosened_tolerance_changes_the_flag_not_the_measurement(runs, tmp_path):
    low, high = runs
    changed = _copy(high, tmp_path, "iso")
    _edit_records(changed / "metadata" / "exposure_posterior.json", PhotographicSensitivity=400)
    from synthraw.comparison import EstimatorTolerances

    loose = SourceResolutionComparator(EstimatorTolerances(exposure_stops=3.0))
    result = loose.compare(low, changed, crops={})
    assert result.flags == [] and result.crops == []
    assert result.checks[2].measured["PhotographicSensitivity"] == pytest.approx(2.0)


def test_same_resolution_pair_and_mismatched_masters_fail_hard_gates(runs, tmp_path):
    low, _ = runs
    twin = _copy(low, tmp_path, "twin")
    result = SourceResolutionComparator().compare(low, twin, crops={})
    assert result.status == "FAILED" and not result.checks[0].passed
    _, high = runs
    cropped = _copy(high, tmp_path, "cropped")
    master = cropped / "masters" / "developed_50mp_16bit.tif"
    tifffile.imwrite(master, tifffile.imread(master)[:, :90])
    result = SourceResolutionComparator().compare(low, cropped)
    failed = [check.name for check in result.checks if check.hard and not check.passed]
    assert failed == ["final_difference_map"] and result.crops == []


def test_crop_boxes_parse_and_validate():
    grid = grid_crops(2, 3)
    assert [crop.name for crop in grid][:2] == ["grid_r0_c0", "grid_r0_c1"]
    assert grid[-1].pixels(96, 40) == (64, 20, 96, 40)
    assert (
        sum((c.pixels(96, 40)[2] - c.pixels(96, 40)[0]) * (c.pixels(96, 40)[3] - c.pixels(96, 40)[1]) for c in grid)
        == 96 * 40
    )
    crops = parse_crops({"face_skin": (0.25, 0.1, 0.5, 0.6)})
    assert crops == [CropBox("face_skin", 0.25, 0.1, 0.5, 0.6)]
    assert CropBox.from_pixels("hair", (24, 4, 48, 24), 96, 40).pixels(96, 40) == (24, 4, 48, 24)
    assert CropBox("thin", 0.5, 0.5, 0.5001, 0.5001).pixels(96, 40) == (48, 20, 49, 21)
    for bad in ((0.5, 0, 0.4, 1), (0, 0, 1.1, 1), (0, float("nan"), 1, 1)):
        with pytest.raises(ValueError):
            CropBox("bad", *bad)
    with pytest.raises(TypeError):
        parse_crops([(0, 0, 1, 1)])


def test_downsampled_source_keeps_icc_and_halves_size(tmp_path):
    source = tmp_path / "in.png"
    Image.new("RGB", (101, 40), (120, 60, 30)).save(source, icc_profile=b"fake-icc")
    out = make_downsampled_source(source, tmp_path / "sub" / "out.png")
    with Image.open(out) as image:
        assert image.size == (50, 20) and image.info.get("icc_profile") == b"fake-icc"
        assert np.allclose(np.asarray(image), (120, 60, 30), atol=1)
    with pytest.raises(ValueError):
        make_downsampled_source(source, tmp_path / "x.png", factor=1)


def test_dng_statistics_keep_cfa_phase_under_striding(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "metadata").mkdir()
    mosaic = np.empty((300, 400), np.uint16)
    for index, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        mosaic[dy::2, dx::2] = 1000 * (index + 1)
    mosaic[0, 0] = 16383
    np.save(tmp_path / "raw" / "sensor_mosaic_uint16.npy", mosaic)
    profile = [
        {"field": "applied_cfa_phase", "value": "BGGR"},
        {"field": "applied_adc_black_level", "value": 1000.0},
        {"field": "applied_adc_white_level", "value": 16383},
    ]
    (tmp_path / "metadata" / "sensor_profile.json").write_text(json.dumps(profile))
    stats = dng_statistics(tmp_path, max_samples=500)
    assert list(stats) == ["0_B", "1_G", "2_G", "3_R"]
    assert [stats[k]["p50"] for k in stats] == [1000, 2000, 3000, 4000]
    assert stats["0_B"]["black_clipped_pct"] > 99 and stats["1_G"]["black_clipped_pct"] == 0
    assert stats["0_B"]["white_clipped_pct"] > 0 and stats["3_R"]["white_clipped_pct"] == 0


def test_image_agreement_separates_identical_and_shifted_texture():
    rng = np.random.default_rng(0)
    base = np.clip(0.5 + 0.2 * rng.standard_normal((48, 64, 3)), 0, 1)
    same, _ = image_agreement(base, base)
    assert same["mean_abs_difference"] == 0 and same["ssim"] == pytest.approx(1)
    assert same["edge_correlation"] == pytest.approx(1) and same["texture_correlation"] == pytest.approx(1)
    assert same["psnr_db"] == float("inf")
    other = np.clip(0.5 + 0.2 * rng.standard_normal((48, 64, 3)), 0, 1)
    changed, maps = image_agreement(base, other)
    assert changed["ssim"] < 0.5 and changed["edge_correlation"] < 0.3
    assert maps["difference"].shape == (48, 64)
    tiny, _ = image_agreement(base[:2, :2], base[:2, :2])
    assert np.isnan(tiny["ssim"])
    with pytest.raises(ValueError):
        image_agreement(base, base[:-1])


def test_user_study_pair_sheet_is_blinded_and_keyed(runs, tmp_path):
    low, high = runs
    sheet = UserStudyPairSheet(seed=3, window=16)
    path = sheet.write([("2K-derived", low, "4K-derived", high)], tmp_path, crops=grid_crops(1, 2))
    assert path.name == USER_STUDY_NAME
    text = path.read_text()
    assert "2K-derived" not in text and "4K-derived" not in text
    assert text.count("data:image/png;base64,") == 4 and "Which looks less AI-generated?" in text
    key = json.loads((tmp_path / USER_STUDY_KEY).read_text())
    assert key["seed"] == 3 and len(key["trials"]) == 2
    for trial in key["trials"]:
        assert {trial["left"], trial["right"]} == {"2K-derived", "4K-derived"}
        x0, y0, x1, y1 = trial["box_pixels"]
        assert x1 - x0 == 16 and y1 - y0 == 16  # matched 100% windows, not downsampled
    again = UserStudyPairSheet(seed=3, window=16).write(
        [("2K-derived", low, "4K-derived", high)], tmp_path / "b", crops=grid_crops(1, 2)
    )
    assert json.loads((again.parent / USER_STUDY_KEY).read_text()) == key


def test_detail_energy_separates_detail_above_and_below_a_nyquist_cutoff():
    """Regression: the 2K/4K crops were judged at a 9x downsample, which hid all detail above the 2K Nyquist."""
    rows, cols = np.mgrid[0:128, 0:128]
    coarse = np.sin(2 * np.pi * 0.05 * cols)
    fine = np.sin(2 * np.pi * 0.4 * cols)
    cutoffs = {"above_2k": 0.25}
    assert detail_energy(coarse, cutoffs)["above_2k"] < 0.01
    assert detail_energy(fine, cutoffs)["above_2k"] > 0.99
    mixed = detail_energy(coarse + fine, cutoffs)["above_2k"]
    assert mixed == pytest.approx(0.5, abs=0.05)
    # A smooth gradient is scene shading, not detail.
    assert detail_energy(rows / 128.0 + coarse, cutoffs)["above_2k"] < 0.01
    assert detail_energy(np.zeros((4, 4)), cutoffs) == {"above_2k": None}


def test_full_resolution_window_is_centred_and_clipped_to_the_box():
    assert full_resolution_window((100, 50, 1100, 1050), 512) == (344, 294, 856, 806)
    assert full_resolution_window((0, 0, 40, 30), 512) == (0, 0, 40, 30)
