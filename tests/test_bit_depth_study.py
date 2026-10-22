import numpy as np
import pytest

from synthetic_raw_lab import bit_depth_study as bd
from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import test_images


def ramp(height=16, width=4096) -> np.ndarray:
    """A dark, smooth linear ramp (where quantisation shows first) with a few values above 1."""
    x = np.linspace(0.0, 0.2, width, dtype=np.float32)
    image = np.repeat(np.repeat(x[None, :, None], height, axis=0), 3, axis=2).copy()
    image[:, :, 1] *= 0.8
    image[0, :8] = 1.5
    return image


def rows_by_name(study: dict) -> dict:
    return {row["name"]: row for row in study["representations"]}


def test_representations_rank_by_precision_and_headroom():
    rows = rows_by_name(bd.run_study(ramp(), edit_stops=3.0, cycles=3))
    eight, sixteen, half, single = (
        rows["rgb8_srgb"],
        rows["rgb16_srgb"],
        rows["rgb16_float_linear"],
        rows["rgb32_float_linear"],
    )
    assert eight["quantization_rmse_linear"] > sixteen["quantization_rmse_linear"] > 0
    assert single["quantization_rmse_linear"] == 0 and single["edit_shadow_level_ratio"] == 1.0
    # Integer encodings clip the overshoot; floats keep it.
    assert eight["outside_range_fraction"] == sixteen["outside_range_fraction"] == pytest.approx(24 / (16 * 4096 * 3))
    assert eight["clipping_rmse_linear"] > 0
    assert half["outside_range_fraction"] == single["outside_range_fraction"] == 0
    # Gradient smoothness after the identical push: 8-bit posterises.
    assert eight["edit_shadow_level_ratio"] < 0.2 < sixteen["edit_shadow_level_ratio"] <= 1.0
    assert eight["edit_rmse_encoded"] > sixteen["edit_rmse_encoded"]
    # Repeated transformations: integer 8-bit drifts most, float32 hardly at all.
    assert eight["cycle_drift_rmse_linear"] > half["cycle_drift_rmse_linear"] > single["cycle_drift_rmse_linear"]
    assert eight["levels_in_unit_range"] == 256 and half["levels_in_unit_range"] == 15361


def test_quantization_error_matches_the_8_bit_step():
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 1, (64, 64, 3)).astype(np.float32)
    rows = rows_by_name(bd.run_study(image, cycles=1))
    encoded = cm.linear_to_srgb(image)
    stored = np.rint(encoded * 255) / 255
    expected = np.sqrt(np.mean((cm.srgb_to_linear(stored.astype(np.float32)) - image) ** 2))
    assert rows["rgb8_srgb"]["quantization_rmse_linear"] == pytest.approx(expected, rel=1e-3)
    assert rows["rgb8_srgb"]["psnr_encoded_db"] == pytest.approx(10 * np.log10(12 * 255**2), abs=0.3)


def test_files_are_written_with_real_sizes(tmp_path):
    study = bd.run_study(ramp(32, 64), tmp_path, cycles=1, file_metadata={"notice": "test sample"})
    files = {row["name"]: row["file"] for row in study["representations"]}
    assert {path.name for path in tmp_path.iterdir()} == {
        "rgb8_srgb.png",
        "rgb16_srgb.tif",
        "rgb16_float_linear.exr",
        "rgb32_float_linear.exr",
    }
    assert all(f["size_bytes"] > 0 for f in files.values())
    assert files["rgb32_float_linear"]["size_bytes"] > files["rgb16_float_linear"]["size_bytes"]


def test_mosaic_quantization_matches_a_uniform_quantizer():
    rng = np.random.default_rng(1)
    signal = rng.uniform(0, 1, (256, 256)).astype(np.float32)
    signal[0, :16] = 1.2
    row = bd.mosaic_quantization(signal, 512, 16383, 14)
    assert row["levels_in_unit_range"] == 16383 - 512 + 1
    assert row["rmse"] == pytest.approx(row["expected_rmse"], rel=0.02)
    assert row["max_abs_error"] <= 0.5 / (16383 - 512) + 1e-9
    assert row["outside_range_fraction"] == pytest.approx(16 / signal.size)


def test_expanding_bit_depth_adds_no_information():
    codes = test_images.make_test_image(96, 64)
    result = bd.expansion_information(codes)
    assert not result["information_added"]
    assert all(row["recovers_source_exactly"] for row in result["rows"])
    distinct = {tuple(row["distinct_values_per_channel"]) for row in result["rows"]}
    assert len(distinct) == 1
    assert [row["bits_per_sample"] for row in result["rows"]] == [8, 16, 16, 32]
    assert not result["lossy_representations"]
    assert "not information" in result["finding"]


def test_expansion_of_16_bit_sources_and_bad_input():
    codes = (np.arange(64 * 64 * 3, dtype=np.uint32) * 16 % 65536).astype(np.uint16).reshape(64, 64, 3)
    result = bd.expansion_information(codes)
    assert not result["information_added"]
    # Half floats have an 11-bit significand, so they cannot hold every 16-bit code.
    assert result["lossy_representations"] == ["16-bit float expansion (normalised)"]
    with pytest.raises(ValueError):
        bd.expansion_information(codes.astype(np.float32))


def test_study_rejects_bad_parameters():
    with pytest.raises(ValueError):
        bd.run_study(ramp(), edit_stops=0)
    with pytest.raises(ValueError):
        bd.run_study(np.zeros((4, 4), dtype=np.float32))
