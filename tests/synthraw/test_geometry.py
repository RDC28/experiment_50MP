"""Tiny source geometry checks; no full-resolution images are allocated."""

import math

import pytest

from synthraw.config import ExperimentConfig
from synthraw.geometry import TargetGeometryPlanner


def test_native_outpaint_and_memory():
    plan = TargetGeometryPlanner().plan(2048, 868)
    assert (plan.width, plan.height) == (8640, 5760)
    assert plan.scale_factor == 8640 / 2048
    assert plan.crop_box == (0, 0, 2048, 868)
    assert len(plan.outpaint_bands) == 2
    assert not plan.margin_bands
    assert plan.active_area == (0, 0, 5760, 8640)
    assert plan.default_crop_origin == (0, 0)
    assert plan.default_crop_size == (8640, 5760)
    pixels = 8640 * 5760
    assert plan.memory_bytes == dict(
        rgb_float32=pixels * 12,
        rgb_float16=pixels * 6,
        rgb_uint16=pixels * 6,
        cfa_uint16=pixels * 2,
        scalar_float32=pixels * 4,
    )


def test_reference_cinema_geometry():
    plan = TargetGeometryPlanner().plan(2048, 868, ExperimentConfig(OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL"))
    assert (plan.width, plan.height) == (10872, 4608)
    assert plan.width * plan.height == 50_098_176


def test_horizontal_crop():
    plan = TargetGeometryPlanner().plan(2048, 868, ExperimentConfig(SENSOR_FIT_MODE="CROP_HORIZONTAL"))
    assert plan.scale_factor == 5760 / 868
    assert plan.crop_box[0] > 0
    assert plan.crop_box[2] < 2048
    assert plan.content_box == (0, 0, 8640, 5760)
    assert not plan.outpaint_bands


def test_margins_crop_is_inside_source_and_cfa_aligned():
    plan = TargetGeometryPlanner().plan(2048, 868, ExperimentConfig(SENSOR_FIT_MODE="FIT_WITH_SENSOR_MARGINS"))
    assert len(plan.margin_bands) == 2
    assert not plan.outpaint_bands
    x, y = plan.default_crop_origin
    w, h = plan.default_crop_size
    assert all(value % 2 == 0 for value in (x, y, w, h))
    assert x >= plan.content_box[0] and y >= plan.content_box[1]
    assert x + w <= plan.content_box[2] and y + h <= plan.content_box[3]


@pytest.mark.parametrize("source", [(2048, 868), (100, 200), (60, 40), (17, 19)])
@pytest.mark.parametrize("mode", ["OUTPAINT_VERTICAL", "CROP_HORIZONTAL", "FIT_WITH_SENSOR_MARGINS"])
@pytest.mark.parametrize("geometry", ["SONY_A1II_NATIVE", "CINEMA_50MP_VIRTUAL"])
def test_uniform_fit_and_fast_dev(source, mode, geometry):
    config = ExperimentConfig(SENSOR_FIT_MODE=mode, OUTPUT_GEOMETRY=geometry, FAST_DEV_RUN=True)
    plan = TargetGeometryPlanner().plan(*source, config)
    assert max(plan.width, plan.height) <= 512
    assert plan.width % 2 == plan.height % 2 == 0
    left, top, right, bottom = plan.crop_box
    x0, y0, x1, y1 = plan.content_box
    assert math.isclose((right - left) / (bottom - top), (x1 - x0) / (y1 - y0))
    assert math.isclose((right - left) * plan.scale_factor, x1 - x0)
    assert 0 <= left < right <= source[0]
    assert 0 <= top < bottom <= source[1]


@pytest.mark.parametrize("source", [(6336, 2688), (3168, 1344)])
def test_default_crop_box_stays_inside_the_source(source):
    """Regression: float rounding put the 6336 x 2688 right edge at 6336.000000000001."""
    left, top, right, bottom = TargetGeometryPlanner().plan(*source, ExperimentConfig()).crop_box
    assert 0 <= left < right <= source[0]
    assert 0 <= top < bottom <= source[1]


SWEEP_SOURCES = [(width, height) for width in range(1, 41) for height in range(1, 41)] + [
    (6336, 2688),
    (3168, 1344),
    (4096, 1716),
    (2048, 858),
    (3840, 2160),
    (1920, 1080),
    (7680, 3200),
    (2688, 6336),
    (1344, 3168),
    (2160, 3840),
    (1080, 1920),
    (4000, 6000),
]


@pytest.mark.parametrize("mode", ["OUTPAINT_VERTICAL", "CROP_HORIZONTAL", "FIT_WITH_SENSOR_MARGINS"])
@pytest.mark.parametrize("geometry", ["SONY_A1II_NATIVE", "CINEMA_50MP_VIRTUAL"])
@pytest.mark.parametrize("fast_dev", [False, True])
def test_crop_and_content_boxes_stay_inside_source_and_sensor(mode, geometry, fast_dev):
    config = ExperimentConfig(SENSOR_FIT_MODE=mode, OUTPUT_GEOMETRY=geometry, FAST_DEV_RUN=fast_dev)
    planner = TargetGeometryPlanner()
    for source_width, source_height in SWEEP_SOURCES:
        try:
            plan = planner.plan(source_width, source_height, config)
        except ValueError as error:
            # Extreme aspect ratios on a fast-dev sensor can leave no CFA-aligned margin crop.
            assert "no CFA-aligned default crop" in str(error)
            continue
        left, top, right, bottom = plan.crop_box
        assert 0 <= left < right <= source_width, (source_width, source_height, plan.crop_box)
        assert 0 <= top < bottom <= source_height, (source_width, source_height, plan.crop_box)
        x0, y0, x1, y1 = plan.content_box
        assert 0 <= x0 < x1 <= plan.width, (source_width, source_height, plan.content_box)
        assert 0 <= y0 < y1 <= plan.height, (source_width, source_height, plan.content_box)


@pytest.mark.parametrize("source", [(0, 2), (-2, 2), (True, 2), (2.5, 2), (2, 0)])
def test_invalid_dimensions(source):
    with pytest.raises(ValueError, match="positive integers"):
        TargetGeometryPlanner().plan(*source)


def test_fast_dev_max_side_bounds_both_geometries():
    planner = TargetGeometryPlanner()
    native = planner.plan(48, 20, ExperimentConfig(FAST_DEV_RUN=True), fast_dev_max_side=96)
    cinema = planner.plan(
        48, 20, ExperimentConfig(FAST_DEV_RUN=True, OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL"), fast_dev_max_side=96
    )
    assert (native.width, native.height) == (96, 64)
    assert (cinema.width, cinema.height) == (96, 40)
    with pytest.raises(ValueError, match="fast_dev_max_side"):
        planner.plan(48, 20, ExperimentConfig(FAST_DEV_RUN=True), fast_dev_max_side=1)
