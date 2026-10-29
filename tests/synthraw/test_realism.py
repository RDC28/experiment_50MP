"""Realism statistics of a Bayer mosaic, compared against a reference raw (report only)."""

import numpy as np
import pytest

from synthraw.realism import compare, measure_mosaic

RGGB = np.array([[0, 1], [3, 2]])
BLACK, WHITE = 512.0, 16383.0
GAIN_DN2_PER_DN, READ_DN = 0.25, 3.0


def _mosaic(*, row_banding=0.0, hot=(), seed=0):
    """Horizontal ramp with shot plus read noise, optional row banding and hot photosites."""
    rng = np.random.default_rng(seed)
    rows, cols = 512, 1024
    signal = np.broadcast_to(np.geomspace(2, 0.6 * (WHITE - BLACK), cols), (rows, cols))
    noisy = signal + rng.normal(size=(rows, cols)) * np.sqrt(GAIN_DN2_PER_DN * signal + READ_DN**2)
    noisy += row_banding * rng.normal(size=(rows, 1))
    mosaic = np.clip(np.round(noisy + BLACK), 0, WHITE)
    for row, column in hot:
        mosaic[row, column] = WHITE
    return mosaic.astype(np.uint16)


def test_photon_transfer_recovers_the_simulated_noise_model():
    planes = measure_mosaic(_mosaic(), black=BLACK, white=WHITE, pattern=RGGB)["planes"]
    assert set(planes) == {"R", "G1", "G2", "B"}
    fit = planes["G1"]["photon_transfer_fit"]
    assert fit["dn2_per_dn"] == pytest.approx(GAIN_DN2_PER_DN, rel=0.25)
    assert fit["full_well_e"] == pytest.approx((WHITE - BLACK) / GAIN_DN2_PER_DN, rel=0.25)
    assert planes["G1"]["defects"]["hot_count"] == 0
    assert planes["G1"]["fraction_clipped"] == 0


def test_row_banding_and_hot_photosites_are_detected():
    clean = measure_mosaic(_mosaic(), black=BLACK, white=WHITE, pattern=RGGB)["planes"]["R"]
    banded = measure_mosaic(
        _mosaic(row_banding=4.0, hot=[(100, 200), (300, 600)]), black=BLACK, white=WHITE, pattern=RGGB
    )["planes"]["R"]
    assert banded["darkest_blocks"]["row_banding_rms_dn"] > 2 > clean["darkest_blocks"]["row_banding_rms_dn"]
    psd = banded["banding_psd"]
    assert psd["row_axis_power_over_offaxis"] > 3 * clean["banding_psd"]["row_axis_power_over_offaxis"]
    assert banded["defects"]["hot_count"] == 2


def test_comparison_ratios_are_one_for_identical_raws_and_reject_non_bayer():
    measured = measure_mosaic(_mosaic(), black=BLACK, white=WHITE, pattern=RGGB)
    reference = {**measured, "as_shot_neutral": [0.5, 1, 0.8]}
    rows = compare(reference, reference)
    assert rows and all(row["ratio"] == pytest.approx(1) for row in rows if row["ratio"] is not None)
    with pytest.raises(ValueError, match="RGGB-family"):
        measure_mosaic(_mosaic(), black=BLACK, white=WHITE, pattern=np.array([[0, 1], [1, 0]]))
