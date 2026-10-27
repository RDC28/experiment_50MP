"""PRD_v3 automated validation sequence on a tiny end-to-end pipeline run folder."""

import json
import shutil
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from synthraw.config import ExperimentConfig
from synthraw.pipeline import run_pipeline
from synthraw.validation import ValidationThresholds
from synthraw.validation_runner import RESULTS_NAME, ValidationRunner, load_dng_records

EXPECTED_ROWS = {
    "numeric_health",
    "cfa_phase",
    "cfa_phase_noisy",
    "adc_contract",
    "libraw_reopen",
    "metadata_dependencies",
    "color_matrix_invertible",
    "white_point_near_neutral",
    "color_matrix_interpolation",
    "transform_finite_nonnegative",
    "noise_profile_agreement",
    "noise_fit_r2",
    "demosaic_ahd",
    "demosaic_additional",
    "exposure_range",
    "white_balance_response",
    "roundtrip_supported_ssim",
    "roundtrip_whole_ssim",
    "colour_delta_e00",
    "roundtrip_unaligned",
    "sr_seam_ratio",
    "hdr_negative_radiance",
    "hdr_generated_area",
    "determinism",
    "noise_seed_structure",
    "body_seed_signature",
    "sensor_realisations",
    "rawtherapee_neutral_render",
}


@pytest.fixture(scope="module")
def run_root(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("validation_run")
    rows, cols = np.mgrid[0:20, 0:48]
    pixels = np.stack([255 * cols / 47, 255 * rows / 19, 128 + 100 * np.sin(cols / 6)], axis=-1)
    source = tmp_path / "source.png"
    Image.fromarray(np.clip(np.round(pixels), 0, 255).astype(np.uint8)).save(source)
    config = ExperimentConfig(OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL", FAST_DEV_RUN=True, SR_BACKEND="LANCZOS")
    return run_pipeline(
        source, config, tmp_path / "runs", timestamp=datetime(2026, 1, 2, 3, 4, 5), fast_dev_max_side=96
    ).root


def _copy(run_root, tmp_path):
    return shutil.copytree(run_root, tmp_path / run_root.name)


def test_tiny_run_passes_every_hard_gate_and_writes_results(run_root, tmp_path):
    root = _copy(run_root, tmp_path)
    messages = []
    run = ValidationRunner().run(root, log=messages.append)
    names = [check.name for check in run.checks]
    assert set(names) == EXPECTED_ROWS and len(names) == len(EXPECTED_ROWS)
    failed = [check.name for check in run.checks if check.passed is False]
    assert failed == [] and run.status == "PASSED"
    written = json.loads((root / "validation" / RESULTS_NAME).read_text())
    assert written["status"] == "PASSED" and written["hard_failures"] == []
    assert {row["name"] for row in written["checks"]} == EXPECTED_ROWS
    assert written["summary"]["cfa_phase"] == "RGGB"
    assert 0 < written["summary"]["source_supported_fraction"] <= 1
    # Products for the validation figures are kept, already downsampled.
    assert run.noise_fit is not None and len(run.noise_fit.levels) >= 5
    assert run.roundtrip is not None and run.roundtrip.difference.shape == (40, 96)
    assert "AHD" in run.demosaic_thumbs
    assert set(run.editability_thumbs) == {"exposure", "white_balance"}
    assert messages[-1].startswith("validation PASSED")


def test_dng_records_round_trip_from_json(run_root):
    records = load_dng_records(run_root / "metadata" / "dng_metadata.json")
    assert records["Make"].value == "Synthetic"
    assert records["CFAPattern"].provenance.value in "DIFM"


def test_negative_radiance_makes_the_run_failed(run_root, tmp_path):
    root = _copy(run_root, tmp_path)
    path = root / "intermediates" / "04_hdr_radiance.npy"
    radiance = np.load(path)
    radiance[3, 5, 1] = -0.25
    np.save(path, radiance)
    run = ValidationRunner().run(root)
    assert run.status == "FAILED"
    written = json.loads((root / "validation" / RESULTS_NAME).read_text())
    assert written["status"] == "FAILED"
    assert set(written["hard_failures"]) == {"numeric_health", "hdr_negative_radiance"}


def test_negative_noisy_electrons_from_read_noise_are_healthy(run_root, tmp_path):
    """Regression: zero-signal photosites go negative with read noise, which failed numeric_health."""
    root = _copy(run_root, tmp_path)
    for name in ("08_noisy_electrons.npy", "10_noisy_cfa.npy"):
        path = root / "intermediates" / name
        noisy = np.load(path)
        noisy[:2] = -4.5
        np.save(path, noisy)
    rows = {check.name: check for check in ValidationRunner().run(root).checks}
    assert rows["numeric_health"].passed, rows["numeric_health"].measured
    path = root / "intermediates" / "10_noisy_cfa.npy"
    noisy = np.load(path)
    noisy[0, 0] = np.nan
    np.save(path, noisy)
    rows = {check.name: check for check in ValidationRunner().run(root).checks}
    assert rows["numeric_health"].measured["arrays"]["10_noisy_cfa.npy"]["nonfinite"] == 1
    assert not rows["numeric_health"].passed


def test_sensor_margin_run_with_zero_signal_passes_numeric_health(tmp_path):
    """End to end: FIT_WITH_SENSOR_MARGINS fills margins with zero signal, so read noise makes them negative."""
    rows, cols = np.mgrid[0:20, 0:48]
    pixels = np.stack([255 * cols / 47, 255 * rows / 19, np.full_like(cols, 128)], axis=-1)
    source = tmp_path / "wide.png"
    Image.fromarray(np.round(pixels).astype(np.uint8)).save(source)
    config = ExperimentConfig(
        OUTPUT_GEOMETRY="SONY_A1II_NATIVE",
        SENSOR_FIT_MODE="FIT_WITH_SENSOR_MARGINS",
        FAST_DEV_RUN=True,
        SR_BACKEND="LANCZOS",
    )
    root = run_pipeline(source, config, tmp_path / "runs", fast_dev_max_side=96).root
    assert np.load(root / "intermediates" / "10_noisy_cfa.npy").min() < 0
    run = ValidationRunner().run(root)
    rows = {check.name: check for check in run.checks}
    assert rows["numeric_health"].passed, rows["numeric_health"].measured
    assert run.status == "PASSED", [check.name for check in run.checks if check.hard and not check.passed]


def test_dng_noise_profile_disagreeing_with_the_flats_fails_the_run(run_root, tmp_path):
    root = _copy(run_root, tmp_path)
    path = root / "metadata" / "dng_metadata.json"
    records = json.loads(path.read_text())
    for record in records:
        if record["field"] == "NoiseProfile":
            record["value"] = [value * 1.2 for value in record["value"]]
    path.write_text(json.dumps(records))
    run = ValidationRunner().run(root)
    assert run.status == "FAILED"
    written = json.loads((root / "validation" / RESULTS_NAME).read_text())
    assert written["hard_failures"] == ["noise_profile_agreement"]


def test_metadata_tampering_and_cfa_mismatch_fail_hard_gates(run_root, tmp_path):
    root = _copy(run_root, tmp_path)
    path = root / "intermediates" / "09_clean_cfa.npy"
    cfa = np.load(path)
    cfa[0, 0] += 1.0
    np.save(path, cfa)
    metadata = root / "metadata" / "metadata.json"
    payload = json.loads(metadata.read_text())
    for record in payload["groups"]["dng"]:
        if record["field"] == "Make":
            record["value"] = "SONY"
    metadata.write_text(json.dumps(payload))
    run = ValidationRunner().run(root)
    hard = {check.name for check in run.checks if check.hard and check.passed is not True}
    assert hard == {"cfa_phase", "metadata_dependencies"}


def test_thresholds_are_configuration_and_measured_values_are_unchanged(run_root, tmp_path):
    root = _copy(run_root, tmp_path)
    strict = ValidationRunner(ValidationThresholds(roundtrip_supported_ssim=0.99999, median_delta_e00=1e-6)).run(root)
    rows = {check.name: check for check in strict.checks}
    assert rows["roundtrip_supported_ssim"].passed is False
    assert rows["colour_delta_e00"].passed is False
    # QA targets are not hard gates: missing them is reported, not a FAILED run.
    assert strict.status == "PASSED"
    written = json.loads((root / "validation" / RESULTS_NAME).read_text())
    assert {"roundtrip_supported_ssim", "colour_delta_e00"} <= set(written["qa_targets_missed"])
    default = {check.name: check for check in ValidationRunner().run(root).checks}
    assert default["roundtrip_supported_ssim"].measured["ssim"] == rows["roundtrip_supported_ssim"].measured["ssim"]


def test_reference_run_determinism_and_missing_tiff(run_root, tmp_path):
    root = _copy(run_root, tmp_path)
    (root / "masters" / "developed_50mp_16bit.tif").unlink()
    run = ValidationRunner().run(root, reference_run=run_root)
    rows = {check.name: check for check in run.checks}
    assert rows["pipeline_determinism"].passed is True
    assert rows["roundtrip_supported_ssim"].passed is None and run.roundtrip is None
    mosaic = root / "raw" / "sensor_mosaic_uint16.npy"
    np.save(mosaic, np.load(mosaic) + 1)
    rows = {check.name: check for check in ValidationRunner().run(root, reference_run=run_root).checks}
    assert rows["pipeline_determinism"].passed is False


@pytest.fixture(scope="module")
def realisations_root(tmp_path_factory):
    """Lossless DNGs with three noise realisations, validated as one run."""
    tmp_path = tmp_path_factory.mktemp("realisations_run")
    rows, cols = np.mgrid[0:20, 0:48]
    pixels = np.stack([255 * cols / 47, 255 * rows / 19, 128 + 100 * np.sin(cols / 6)], axis=-1)
    source = tmp_path / "source.png"
    Image.fromarray(np.clip(np.round(pixels), 0, 255).astype(np.uint8)).save(source)
    config = ExperimentConfig(
        FAST_DEV_RUN=True, SR_BACKEND="LANCZOS", N_SENSOR_REALISATIONS=3, DNG_COMPRESSION="LOSSLESS"
    )
    return run_pipeline(
        source, config, tmp_path / "runs", timestamp=datetime(2026, 1, 2, 3, 4, 5), fast_dev_max_side=96
    ).root


def test_lossless_multi_realisation_run_passes_every_hard_gate(realisations_root, tmp_path):
    root = _copy(realisations_root, tmp_path)
    run = ValidationRunner().run(root)
    assert [check.name for check in run.checks if check.passed is False] == [] and run.status == "PASSED"
    row = next(check for check in run.checks if check.name == "sensor_realisations")
    assert row.hard and row.measured["count"] == 3 and len(set(row.measured["seeds"])) == 3
    differences = row.measured["differences"]
    assert [entry["index"] for entry in differences] == [1, 2, 3]
    assert differences[0]["rms_difference_from_primary_dn"] == 0
    assert all(entry["rms_difference_from_primary_dn"] > 0 for entry in differences[1:])


@pytest.mark.parametrize("tamper", ["replaced", "missing", "seed"])
def test_tampered_realisations_fail_their_hard_gate(realisations_root, tmp_path, tamper):
    root = _copy(realisations_root, tmp_path)
    second = root / "raw" / "synthetic_50mp_realisation_002.dng"
    record = root / "validation" / "sensor_realisations.json"
    if tamper == "replaced":
        shutil.copyfile(root / "raw" / "synthetic_50mp_realisation_003.dng", second)
    elif tamper == "missing":
        second.unlink()
    else:
        payload = json.loads(record.read_text())
        payload["realisations"][1]["sensor_noise_seed"] += 1
        record.write_text(json.dumps(payload))
    run = ValidationRunner().run(root)
    assert run.status == "FAILED"
    assert json.loads((root / "validation" / RESULTS_NAME).read_text())["hard_failures"] == ["sensor_realisations"]
    row = next(check for check in run.checks if check.name == "sensor_realisations")
    assert {"replaced": "sha256", "missing": "is missing", "seed": "recorded seeds"}[tamper] in row.detail
