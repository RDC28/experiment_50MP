"""Tiny, allocation-free geometry and metadata contracts."""

import json

import pytest

from prd_v5.geometry import GeometryPlanner


@pytest.mark.parametrize("size", [(2048, 1152), (4096, 2304), (3000, 4000), (6336, 2688), (100, 100)])
def test_default_preserves_uniform_mapping_and_reports_real_pixel_budget(size):
    plan = GeometryPlanner().plan(*size)
    assert plan.width % 2 == plan.height % 2 == 0
    assert plan.actual_megapixels == pytest.approx(50, abs=0.02)
    assert not plan.generated_bands
    assert plan.width / plan.height == pytest.approx(size[0] / size[1], rel=0.001)
    # The crop transform must map source coordinates to supported output edges.
    for i, edge in enumerate(plan.source_crop):
        assert edge * plan.scale + plan.offset_xy[i % 2] == pytest.approx(plan.content_box[i], abs=1e-9)
    x, y = plan.default_crop_origin
    w, h = plan.default_crop_size
    assert all(v % 2 == 0 for v in (x, y, w, h))
    assert x >= plan.content_box[0] and y >= plan.content_box[1]
    assert x + w <= plan.content_box[2] and y + h <= plan.content_box[3]
    record = json.loads(json.dumps(plan.to_dict(), allow_nan=False))
    assert record["buffer_bytes"]["bayer_uint16"] == plan.width * plan.height * 2
    assert record["active_area"] == [0, 0, plan.height, plan.width]
    assert record["requested_megapixels"] == 50


def test_two_input_resolutions_have_identical_output_geometry_and_normalized_mapping():
    small = GeometryPlanner().plan(2048, 1152)
    large = GeometryPlanner().plan(4096, 2304)
    assert (small.width, small.height, small.content_box, small.default_crop_size) == (
        large.width,
        large.height,
        large.content_box,
        large.default_crop_size,
    )
    assert small.scale == large.scale * 2
    assert small.offset_xy == large.offset_xy


@pytest.mark.parametrize("size", [(2048, 1024), (1024, 2048)])
def test_native_letterbox_and_crop_keep_source_support_explicit(size):
    planner = GeometryPlanner()
    margins = planner.plan(*size, mode="SONY_A1II_NATIVE")
    assert (margins.width, margins.height) == (8640, 5760)
    assert margins.actual_megapixels == 49.7664
    assert margins.margin_bands and not margins.generated_bands
    assert margins.source_crop == pytest.approx((0, 0, *size))
    cropped = planner.plan(*size, mode="SONY_A1II_NATIVE", native_fit_policy="CROP_HORIZONTAL")
    assert not cropped.margin_bands and not cropped.generated_bands
    assert cropped.content_box == (0, 0, 8640, 5760)
    assert cropped.default_crop_origin == (0, 0)
    assert cropped.default_crop_size == (8640, 5760)
    assert (cropped.source_crop[2] - cropped.source_crop[0]) * cropped.scale == pytest.approx(8640)
    assert (cropped.source_crop[3] - cropped.source_crop[1]) * cropped.scale == pytest.approx(5760)


def test_outpainting_plan_does_not_claim_generation_and_matches_margin_area():
    planner = GeometryPlanner()
    plan = planner.plan(2048, 1024, mode="SONY_A1II_NATIVE", native_fit_policy="OUTPAINT_VERTICAL")
    assert not plan.margin_bands and len(plan.generated_bands) == 2
    assert plan.to_dict()["outpainting_status"] == "required_not_executed"
    generated = sum((right - left) * (bottom - top) for left, top, right, bottom in plan.generated_bands)
    left, top, right, bottom = plan.content_box
    assert generated + (right - left) * (bottom - top) == pytest.approx(plan.width * plan.height)
    assert plan.default_crop_size == (8640, 5760)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_width": True},
        {"source_height": 0},
        {"source_width": 2.5},
        {"target_megapixels": float("nan")},
        {"target_megapixels": float("inf")},
        {"target_megapixels": True},
        {"target_megapixels": 0},
        {"target_megapixels": 1001},
        {"mode": "FAST"},
        {"native_fit_policy": "STRETCH"},
    ],
)
def test_invalid_geometry_fails_before_any_processing(kwargs):
    parameters = {"source_width": 2048, "source_height": 1152, **kwargs}
    with pytest.raises(ValueError):
        GeometryPlanner().plan(**parameters)


def test_impossible_cfa_crop_is_rejected():
    with pytest.raises(ValueError, match="CFA-aligned"):
        GeometryPlanner().plan(1000000000, 1, mode="SONY_A1II_NATIVE")
