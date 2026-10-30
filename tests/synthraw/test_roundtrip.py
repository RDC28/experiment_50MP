"""PRD_v3 round-trip, SR seam and HDR validation rows, plus carry-over C on the pipeline."""

from datetime import datetime

import numpy as np
import pytest
import tifffile
from PIL import Image

from synthraw.colour_space import rgb_to_xyz_matrix
from synthraw.config import ExperimentConfig
from synthraw.exporters import ADOBE_GAMMA
from synthraw.hdr import inverse_smoothstep, inverse_transfer
from synthraw.pipeline import run_pipeline
from synthraw.validation import (
    HDRValidator,
    RoundTripValidator,
    SeamValidator,
    ValidationThresholds,
    block_mean,
    hard_failures,
)


def _reference(height=40, width=64) -> np.ndarray:
    rows, cols = np.mgrid[0:height, 0:width]
    return np.stack(
        [0.1 + 0.8 * cols / (width - 1), 0.1 + 0.8 * rows / (height - 1), 0.5 + 0.3 * np.sin(cols / 5)], axis=-1
    ).astype(np.float32)


def _develop_linear(reference: np.ndarray, gain: float = 0.8) -> np.ndarray:
    """What a faithful linear RAW development of the pipeline's scene target looks like."""
    scene = inverse_smoothstep(inverse_transfer(reference)) * gain
    to_adobe = np.linalg.solve(rgb_to_xyz_matrix("ADOBE_RGB"), rgb_to_xyz_matrix("SRGB"))
    adobe = np.clip(scene @ to_adobe.T, 0, 1) ** (1 / ADOBE_GAMMA)
    return np.round(adobe * 65535).astype(np.uint16)


def _by_name(checks):
    return {check.name: check for check in checks}


def test_block_mean_area_averages_and_drops_partial_blocks():
    array = np.arange(5 * 7, dtype=float).reshape(5, 7)
    result = block_mean(array, 2, strip_blocks=1)
    assert result.shape == (2, 3)
    assert result[0, 0] == pytest.approx(np.mean([0, 1, 7, 8]))
    with pytest.raises(ValueError):
        block_mean(array, 0)


def test_faithful_linear_development_passes_after_tone_normalisation():
    reference = _reference()
    supported = np.ones(reference.shape[:2], bool)
    result = RoundTripValidator().validate(reference, _develop_linear(reference), supported)
    checks = _by_name(result.checks)
    assert checks["roundtrip_supported_ssim"].measured["ssim"] > 0.99
    assert checks["colour_delta_e00"].measured["median_delta_e00"] < 0.5
    # The global alignment only has to absorb the exposure gain.
    np.testing.assert_allclose(checks["colour_delta_e00"].measured["alignment_matrix"], np.eye(3) / 0.8, atol=0.02)
    # Without the inverted display tone the linear master is honestly far from the source.
    assert checks["roundtrip_unaligned"].passed is None
    assert checks["roundtrip_unaligned"].measured["unaligned"]["median_delta_e00_supported"] > 2.5
    assert result.difference.shape == reference.shape[:2]
    assert not hard_failures(result.checks)


def test_local_structure_damage_fails_ssim_and_thresholds_are_configuration():
    reference = _reference()
    developed = _develop_linear(reference)
    noise = np.random.default_rng(1).normal(0, 9000, developed.shape)
    damaged = np.clip(developed + noise, 0, 65535).astype(np.uint16)
    supported = np.ones(reference.shape[:2], bool)
    checks = _by_name(RoundTripValidator().validate(reference, damaged, supported).checks)
    assert checks["roundtrip_supported_ssim"].passed is False
    measured = checks["roundtrip_supported_ssim"].measured["ssim"]
    relaxed = ValidationThresholds(roundtrip_supported_ssim=measured - 0.01)
    relaxed_checks = _by_name(RoundTripValidator(relaxed).validate(reference, damaged, supported).checks)
    assert relaxed_checks["roundtrip_supported_ssim"].passed is True
    assert relaxed_checks["roundtrip_supported_ssim"].measured["ssim"] == pytest.approx(measured)


def test_round_trip_downsamples_for_analysis_and_rejects_mismatched_grids():
    reference = _reference(64, 96)
    developed = _develop_linear(reference)
    supported = np.ones(reference.shape[:2], bool)
    checks = _by_name(RoundTripValidator().validate(reference, developed, supported, max_side=48).checks)
    assert checks["roundtrip_whole_ssim"].measured["analysis_factor"] == 2
    assert checks["roundtrip_whole_ssim"].measured["analysed_shape"] == [32, 48]
    with pytest.raises(ValueError, match="one grid"):
        RoundTripValidator().validate(reference[:-1], developed, supported)
    with pytest.raises(ValueError, match="16-bit"):
        RoundTripValidator().validate(reference, developed.astype(np.float32), supported)


def test_seam_validator_gates_ratio_and_reports_single_tile_as_not_applicable():
    validator = SeamValidator()
    single = validator.validate({"seams": None, "seam_unavailable_reason": "Single tile", "tiled": {"tile_count": 1}})
    assert single.passed is None and "Single tile" in single.detail
    seams = {"boundary_mean": 0.012, "interior_mean": 0.01}
    assert validator.validate({"seams": {**seams, "ratio": 1.2}}).passed is True
    assert validator.validate({"seams": {**seams, "ratio": 1.3}}).passed is False


def test_hdr_validator_fails_hard_on_negative_radiance_and_reports_generated_area():
    radiance = np.full((6, 5, 3), 0.5, np.float32)
    report = {"generated_hdr_area_percentage": 12.5, "backend": "DETERMINISTIC"}
    checks = _by_name(HDRValidator().validate(radiance, report, strip_rows=2))
    assert checks["hdr_negative_radiance"].passed is True
    assert checks["hdr_generated_area"].passed is None
    assert checks["hdr_generated_area"].measured["generated_hdr_area_percentage"] == 12.5
    radiance[3, 2, 1] = -1e-4
    failed = HDRValidator().validate(radiance, report, strip_rows=2)
    assert [check.name for check in hard_failures(failed)] == ["hdr_negative_radiance"]
    assert _by_name(failed)["hdr_negative_radiance"].measured["negative_pixels"] == 1


@pytest.mark.parametrize("geometry", ["CINEMA_50MP_VIRTUAL", "SONY_A1II_NATIVE"])
def test_pipeline_round_trip_is_colour_faithful(tmp_path, geometry):
    """Carry-over C: the master is a faithful linear development; only the display tone differs."""
    rows, cols = np.mgrid[0:40, 0:96]
    pixels = np.stack([255 * cols / 95, 255 * rows / 39, 128 + 100 * np.sin(cols / 6)], axis=-1)
    source = tmp_path / "gradient.png"
    Image.fromarray(np.clip(np.round(pixels), 0, 255).astype(np.uint8)).save(source)
    config = ExperimentConfig(OUTPUT_GEOMETRY=geometry, FAST_DEV_RUN=True, SR_BACKEND="LANCZOS")
    result = run_pipeline(
        source, config, tmp_path / "runs", timestamp=datetime(2026, 1, 2, 3, 4, 5), fast_dev_max_side=192
    )
    confidence = result.root / "confidence"
    supported = (np.load(confidence / "sr_source_supported_mask.npy") > 0.5) & np.load(
        confidence / "source_supported_mask.npy"
    )
    reference = np.load(result.root / "intermediates" / "02_sr_rgb.npy")
    checks = _by_name(RoundTripValidator().validate(reference, tifffile.imread(result.tiff_path), supported).checks)
    assert checks["roundtrip_supported_ssim"].passed is True
    assert checks["roundtrip_whole_ssim"].passed is True
    assert checks["colour_delta_e00"].passed is True
    # No double WB, mis-tagged space or exposure error: exposure is placed from the HDR peak actually
    # reconstructed and the development honours BaselineExposure, so alignment is near the identity.
    matrix = np.asarray(checks["colour_delta_e00"].measured["alignment_matrix"])
    np.testing.assert_allclose(matrix, np.eye(3), atol=0.02)
    assert checks["colour_delta_e00"].measured["alignment_gain"] == pytest.approx(1, abs=0.02)
    # The "lifted, desaturated" look is the absent display tone curve, reported not hidden.
    assert checks["roundtrip_unaligned"].measured["unaligned"]["median_delta_e00_supported"] > 2.5
