import math
from fractions import Fraction

import pytest

from synthetic_raw_lab.resolution import compute_target_resolution, nominal_scale

SIZES = [
    (3840, 2160),
    (4096, 2160),
    (4096, 2304),
    (2160, 3840),
    (3000, 3000),
    (4000, 1714),
    (3833, 2157),
    (5000, 1000),
    (1024, 768),
]


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [(3840, 2160, 2.46), (4096, 2160, 2.38), (4096, 2304, 2.30)],
)
def test_prd_table_scale_factors(width, height, expected):
    assert round(nominal_scale(width, height), 2) == expected


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_tolerance_mode_meets_gate_with_even_dimensions(width, height):
    target = compute_target_resolution(width, height)
    assert abs(target.pixel_count_error) <= 0.001
    assert target.width % 2 == 0 and target.height % 2 == 0
    assert abs(target.aspect_ratio_error) < 0.001


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_tolerance_mode_picks_best_aspect_ratio(width, height):
    """Brute force: no even-dimension candidate within tolerance has a smaller aspect error."""
    target = compute_target_resolution(width, height)
    best_error = abs(Fraction(target.width * height, target.height * width) - 1)
    for w in range(max(2, target.width - 600), target.width + 602, 2):
        low, high = math.ceil(49_950_000 / w), math.floor(50_050_000 / w)
        for h in range(low + low % 2, high + 1, 2):
            assert abs(Fraction(w * height, h * width) - 1) >= best_error


def test_tolerance_mode_is_deterministic():
    assert compute_target_resolution(3840, 2160) == compute_target_resolution(3840, 2160)


def test_3840x2160_result():
    target = compute_target_resolution(3840, 2160)
    assert (target.width, target.height) == (9426, 5302)
    assert target.to_dict()["pixels"] == 9426 * 5302


def test_dimension_multiple_one_allows_odd_sizes():
    target = compute_target_resolution(3840, 2160, dimension_multiple=1)
    assert abs(target.pixel_count_error) <= 0.001
    assert abs(target.aspect_ratio_error) <= abs(compute_target_resolution(3840, 2160).aspect_ratio_error)


def test_exact_mode_hits_exact_pixel_count():
    target = compute_target_resolution(4000, 2000, mode="exact")
    assert (target.width, target.height) == (10000, 5000)
    assert target.pixels == 50_000_000
    assert target.pixel_count_error == 0


def test_exact_mode_refuses_distortion():
    with pytest.raises(ValueError, match="aspect"):
        compute_target_resolution(3840, 2160, mode="exact")


def test_exact_mode_without_aspect_preservation_returns_closest_divisor_pair():
    target = compute_target_resolution(3840, 2160, mode="exact", preserve_aspect_ratio=False)
    assert target.pixels == 50_000_000
    assert (target.width, target.height) == (10000, 5000)


def test_downscale_when_source_exceeds_target():
    target = compute_target_resolution(10000, 6000)
    assert target.nominal_scale < 1
    assert abs(target.pixel_count_error) <= 0.001


def test_other_targets_and_tolerances():
    target = compute_target_resolution(3840, 2160, 2.0, pixel_count_tolerance=0.01)
    assert abs(target.pixels - 2_000_000) <= 20_000


@pytest.mark.parametrize(
    "kwargs",
    [{"mode": "nearest"}, {"dimension_multiple": 0}, {"pixel_count_tolerance": -1}, {"target_megapixels": 0}],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        compute_target_resolution(3840, 2160, **kwargs)


def test_invalid_source():
    with pytest.raises(ValueError):
        compute_target_resolution(0, 2160)


def test_impossible_tolerance_explains_alternative():
    with pytest.raises(ValueError, match="exact"):
        compute_target_resolution(3840, 2160, pixel_count_tolerance=0, dimension_multiple=16)
