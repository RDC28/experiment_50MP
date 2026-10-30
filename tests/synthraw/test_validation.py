"""PRD_v3 Validation milestone checks on tiny synthetic DNG and sensor fixtures."""

from dataclasses import replace

import numpy as np
import pytest

from synthraw.adc import ADCSimulator
from synthraw.camera_colour import CameraColourModel
from synthraw.cfa import CFAMosaicer
from synthraw.config import ExperimentConfig
from synthraw.dng_metadata import CFA_PATTERNS, DNGMetadataBuilder
from synthraw.dng_writer import DNGWriter
from synthraw.geometry import TargetGeometryPlanner
from synthraw.validation import (
    ADCContractValidator,
    CFAValidator,
    ColourValidator,
    DemosaicValidator,
    EditabilityValidator,
    NoiseValidator,
    NumericHealthValidator,
    ValidationCheck,
    ValidationThresholds,
    hard_failures,
    libraw_reopen_check,
)

NEUTRAL = (0.5, 1.0, 0.75)


def tiny_dng(tmp_path, *, width=64, height=48, phase="RGGB", neutral=NEUTRAL, scale=0.4):
    geometry = replace(
        TargetGeometryPlanner().plan(width, height),
        outpaint_bands=(),  # The bare test sensor has no planned outpaint bands.
        width=width,
        height=height,
        active_area=(0, 0, height, width),
        default_crop_origin=(0, 0),
        default_crop_size=(width, height),
    )
    config = ExperimentConfig()
    colour = CameraColourModel().metadata()
    records = DNGMetadataBuilder().build(
        geometry,
        config,
        cfa_phase=phase,
        as_shot_neutral=neutral,
        color_matrix1=colour["ColorMatrix1"].value,
        color_matrix2=colour["ColorMatrix2"].value,
    )
    y, x = np.mgrid[0:height, 0:width]
    scene = np.stack([0.2 + 0.6 * x / width, 0.3 + 0.4 * y / height, 0.5 - 0.3 * x / width], axis=-1)
    camera = scene * np.asarray(neutral) * scale
    mosaic = np.rint(512 + CFAMosaicer(phase).mosaic(camera) * (16383 - 512)).astype(np.uint16)
    path = DNGWriter().write(tmp_path / "tiny.dng", mosaic, records, config)
    return path, mosaic, records


def test_numeric_health_counts_nonfinite_and_range():
    good = np.linspace(0, 1, 30).reshape(10, 3)
    bad = good.copy()
    bad[0, 0], bad[5, 1], bad[9, 2] = np.nan, -0.5, 2.0
    assert NumericHealthValidator().validate({"good": good}, ranges={"good": (0, 1)}).passed
    check = NumericHealthValidator().validate({"bad": bad}, ranges={"bad": (0, 1)}, strip_rows=3)
    assert not check.passed and check.hard
    assert check.measured["arrays"]["bad"] == {"nonfinite": 1, "below_range": 1, "above_range": 1}


@pytest.mark.parametrize("phase", CFA_PATTERNS)
def test_cfa_phase_gate(phase):
    rgb = np.random.default_rng(1).random((10, 12, 3))
    mosaic = CFAMosaicer(phase).mosaic(rgb)
    assert CFAValidator().validate(mosaic, rgb, phase, strip_rows=3).passed
    other = next(p for p in CFA_PATTERNS if p != phase)
    wrong = CFAValidator().validate(CFAMosaicer(other).mosaic(rgb), rgb, phase)
    assert not wrong.passed and wrong.measured["mismatched_photosites"] > 0
    assert not CFAValidator().validate(rgb, rgb, phase).passed


def test_cfa_gate_checks_dng_pattern(tmp_path):
    _, _, records = tiny_dng(tmp_path)
    rgb = np.zeros((4, 4, 3))
    mosaic = CFAMosaicer("RGGB").mosaic(rgb)
    assert CFAValidator().validate(mosaic, rgb, "RGGB", records).passed
    assert not CFAValidator().validate(CFAMosaicer("BGGR").mosaic(rgb), rgb, "BGGR", records).passed


def test_adc_contract_gate():
    codes = np.array([[0, 512, 16383]], dtype=np.uint16)
    assert ADCContractValidator().validate(codes, black_level=512, white_level=16383, adc_bits=14).passed
    assert not ADCContractValidator().validate(codes, black_level=512, white_level=4095, adc_bits=14).passed
    assert (
        not ADCContractValidator()
        .validate(codes.astype(np.int32), black_level=512, white_level=16383, adc_bits=14)
        .passed
    )
    assert not ADCContractValidator().validate(codes, black_level=512, white_level=16383, adc_bits=12).passed


def test_colour_validator_passes_profile_and_reports_metrics(tmp_path):
    _, _, records = tiny_dng(tmp_path)
    checks = {check.name: check for check in ColourValidator().validate(records, np.array([[0.1, 0.5, 0.9]]))}
    assert all(check.passed for check in checks.values()), {k: c.to_dict() for k, c in checks.items()}
    assert checks["color_matrix_invertible"].measured["condition_number"]["ColorMatrix2"] > 1
    x, y = checks["white_point_near_neutral"].measured["as_shot_white_xy"]
    assert 0.2 < x < 0.6 and 0.2 < y < 0.5


def test_colour_validator_rejects_singular_matrix(tmp_path):
    _, _, records = tiny_dng(tmp_path)
    records["ColorMatrix1"].value = (1, 0, 0, 1, 0, 0, 0, 0, 1)
    checks = {check.name: check for check in ColourValidator().validate(records)}
    assert not checks["color_matrix_invertible"].passed
    assert hard_failures(checks.values())


def flat_kwargs(**overrides):
    return {
        "full_well_electrons": 52000.0,
        "conversion_gain_e_per_dn": 52000.0 / 15871,
        "read_noise_e": 3.0,
        "black_level": 512,
        "white_level": 16383,
        "patch": 16,
    } | overrides


def test_noise_flat_field_fit_matches_noise_profile():
    validator = NoiseValidator()
    fit, profile = validator.measure(ExperimentConfig(PRNU=True, DSNU=True, HOT_PIXELS=True), **flat_kwargs())
    assert len(fit.levels) >= 5 and len(fit.residuals) == len(fit.levels)
    profile_check, r2_check = validator.validate(fit, profile)
    assert profile_check.passed, profile_check.measured
    assert profile_check.measured["S_relative_error"] < 0.05
    assert profile_check.measured["O_relative_error"] < 0.05
    assert r2_check.passed and r2_check.measured["r2"] >= 0.95


@pytest.mark.parametrize("analogue_gain", [4.0, 16.0])
def test_noise_flats_follow_the_applied_analogue_gain(analogue_gain):
    """The flats reproduce a non-unity ISO gain's NoiseProfile; agreement is a hard gate."""
    validator = NoiseValidator()
    fit, profile = validator.measure(ExperimentConfig(), analogue_gain=analogue_gain, **flat_kwargs())
    unity = ADCSimulator().simulate(
        np.zeros((2, 2)), ExperimentConfig(), conversion_gain_e_per_dn=52000.0 / 15871, read_noise_e=3.0
    )
    assert profile[0] == pytest.approx(unity.noise_profile[0] * analogue_gain)
    profile_check, _ = validator.validate(fit, profile)
    assert profile_check.hard and profile_check.passed, profile_check.measured
    wrong = validator.validate(fit, unity.noise_profile)[0]
    assert not wrong.passed and hard_failures([wrong]) == [wrong]


def test_noise_profile_without_quantisation_is_detected():
    """Regression: the pre-fix NoiseProfile omitted 1/12 DN^2 rounding variance (~10% O error)."""
    validator = NoiseValidator()
    fit, profile = validator.measure(ExperimentConfig(), **flat_kwargs())
    span = 16383 - 512
    stale = tuple(value - 1 / (12 * span**2) if index % 2 else value for index, value in enumerate(profile))
    check = validator.validate(fit, stale)[0]
    assert not check.passed and check.measured["O_relative_error"] > 0.05


def test_noise_validator_rejects_wrong_profile_and_short_runs():
    validator = NoiseValidator()
    fit, profile = validator.measure(ExperimentConfig(), **flat_kwargs())
    assert not validator.validate(fit, tuple(value * 1.2 for value in profile))[0].passed
    # Without shot noise the DNG omits NoiseProfile: nothing to compare, so the row is report-only.
    omitted = validator.validate(fit, None)[0]
    assert omitted.passed is None and not hard_failures([omitted])
    with pytest.raises(ValueError, match="N >= 32"):
        validator.measure(ExperimentConfig(), realisations=8, **flat_kwargs())


def test_demosaic_validator_develops_ahd_and_records_amaze(tmp_path):
    path, _, _ = tiny_dng(tmp_path)
    checks, thumbs = DemosaicValidator().develop(path, thumbnail_side=32)
    by_name = {check.name: check for check in checks}
    assert by_name["demosaic_ahd"].passed and by_name["demosaic_ahd"].hard
    assert by_name["demosaic_additional"].passed and by_name["demosaic_additional"].hard
    algorithms = by_name["demosaic_additional"].measured["algorithms"]
    assert algorithms["AMAZE"]["status"] in ("ok", "unsupported")
    assert {"AHD", "DCB", "VNG", "PPG", "LINEAR"} <= set(thumbs)
    assert max(thumb.shape[0] for thumb in thumbs.values()) <= 32
    # Smooth fixture: every algorithm agrees closely with AHD.
    assert all(algorithms[name]["mean_abs_diff_vs_AHD"] < 0.02 for name in thumbs)


def test_editability_exposure_and_white_balance(tmp_path):
    path, _, _ = tiny_dng(tmp_path)
    checks, sheets = EditabilityValidator().validate(path, thumbnail_side=24)
    by_name = {check.name: check for check in checks}
    assert by_name["exposure_range"].passed, by_name["exposure_range"].measured
    assert by_name["white_balance_response"].passed, by_name["white_balance_response"].measured
    assert by_name["exposure_range"].hard and by_name["white_balance_response"].hard
    assert list(sheets["exposure"]) == [f"{ev:+d} EV" for ev in range(-4, 5)]
    assert list(sheets["white_balance"]) == ["cool", "neutral", "warm"]
    means = by_name["exposure_range"].measured["mean_display_value"]
    assert means == sorted(means) and means[0] < means[-1]


def test_editability_flags_wb_blow_up(tmp_path):
    path, _, _ = tiny_dng(tmp_path, scale=1.2)
    strict = EditabilityValidator(ValidationThresholds(wb_max_clipped_increase=0.0))
    by_name = {check.name: check for check in strict.validate(path, thumbnail_side=16)[0]}
    assert by_name["white_balance_response"].measured["clipped_increase"] > 0
    assert not by_name["white_balance_response"].passed
    assert hard_failures(by_name.values()) == [by_name["white_balance_response"]]


def test_libraw_reopen_check_converts_hard_failure(tmp_path):
    path, mosaic, records = tiny_dng(tmp_path)
    assert libraw_reopen_check(dng_path=path, mosaic=mosaic, records=records)[0].passed
    mosaic = mosaic.copy()
    mosaic[0, 0] += 1
    failed = libraw_reopen_check(dng_path=path, mosaic=mosaic, records=records)
    assert hard_failures(failed) == failed


def test_check_serialises_non_finite_values():
    check = ValidationCheck("x", "Domain", None, measured={"a": np.float64(np.inf), "b": np.arange(2)})
    assert check.to_dict()["measured"] == {"a": "inf", "b": [0, 1]}
