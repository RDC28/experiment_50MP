"""Tiny baseline geometry and analytic-to-ADC checks, with no inference."""

from dataclasses import replace

import numpy as np
import pytest

from prd_v5.analytic import AnalyticISP
from prd_v5.geometry import GeometryPlanner
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.raster import rasterize, rasterize_lanczos
from prd_v5.sensor import BayerSensor


def buffers(tmp_path, plan):
    rgb = np.memmap(tmp_path / "rgb", mode="w+", dtype=np.float32, shape=(plan.height, plan.width, 3))
    mask = np.memmap(tmp_path / "mask", mode="w+", dtype=np.bool_, shape=(plan.height, plan.width))
    return rgb, mask


def test_identity_and_analytic_sensor(tmp_path):
    from PIL import Image

    from prd_v5.decode import decode_source

    plan = GeometryPlanner().plan(4, 4, target_megapixels=16 / 1e6)
    pixels = np.arange(48, dtype=np.uint8).reshape(4, 4, 3) * 5
    path = tmp_path / "input.png"
    Image.fromarray(pixels).save(path)
    source = np.memmap(tmp_path / "decoded", mode="w+", dtype=np.float32, shape=(4, 4, 3))
    decode_source(path, source, assume_unprofiled_srgb=True)
    rgb, mask = buffers(tmp_path, plan)
    report = rasterize(source, rgb, mask, plan)
    np.testing.assert_array_equal(rgb, source)
    assert mask.all() and report["sensor_margin_pixels"] == 0
    clean = np.memmap(tmp_path / "clean", mode="w+", dtype=np.float32, shape=rgb.shape)
    adc = np.memmap(tmp_path / "adc", mode="w+", dtype=np.uint16, shape=mask.shape)
    AnalyticISP().process(rgb, clean)
    camera = CameraHypothesis(shot_noise=False, read_noise=False)
    BayerSensor(camera).process(clean, adc)
    expected = np.empty(mask.shape, dtype=np.uint16)
    for y in range(4):
        for x in range(4):
            c = ((0, 1), (1, 2))[y % 2][x % 2]
            v = float(source[y, x, c])
            linear = v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
            expected[y, x] = round(camera.black_level + linear * (camera.white_level - camera.black_level))
    np.testing.assert_array_equal(adc, expected)


def test_pixel_centers_and_margin_mask(tmp_path):
    plan = GeometryPlanner().plan(2, 2, target_megapixels=16 / 1e6)
    source = np.repeat(np.array([[0, 1], [0, 1]], dtype=np.float32)[..., None], 3, axis=2)
    rgb, mask = buffers(tmp_path, plan)
    rasterize(source, rgb, mask, plan)
    np.testing.assert_allclose(rgb[0, :, 0], [0, 0.25, 0.75, 1])
    margin_plan = replace(plan, offset_xy=(0, 1), content_box=(0, 1, 4, 3))
    report = rasterize(source, rgb, mask, margin_plan)
    assert report["sensor_margin_pixels"] == 8
    assert not mask[[0, 3]].any()
    assert not rgb[[0, 3]].any()
    assert mask[1:3].all()


@pytest.mark.parametrize("raster", [rasterize, rasterize_lanczos])
def test_outpaint_and_invalid_buffers_rejected_before_write(tmp_path, raster):
    plan = GeometryPlanner().plan(2, 2, target_megapixels=16 / 1e6)
    source = np.zeros((2, 2, 3), dtype=np.float32)
    rgb, mask = buffers(tmp_path, plan)
    rgb[:] = 0.75
    with pytest.raises(ValueError, match="Outpainting"):
        raster(source, rgb, mask, replace(plan, generated_bands=((0, 0, 4, 1),)))
    with pytest.raises(ValueError, match="matching the geometry plan"):
        raster(np.zeros((3, 2, 3), dtype=np.float32), rgb, mask, plan)
    assert np.all(rgb == 0.75)
    with pytest.raises(ValueError, match="disk-backed"):
        raster(source, np.zeros(rgb.shape, dtype=np.float32), mask, plan)
    with pytest.raises(ValueError, match="separate"):
        raster(rgb, rgb, mask, replace(plan, source_size=(4, 4)))


def test_lanczos_crop_follows_bilinear_geometry(tmp_path):
    plan = GeometryPlanner().plan(50, 20, target_megapixels=0.0004)
    x, y = plan.default_crop_origin
    width, height = plan.default_crop_size
    assert width < plan.content_box[2] - plan.content_box[0]
    source = np.broadcast_to(np.linspace(0, 1, 50, dtype=np.float32)[None, :, None], (20, 50, 3)).copy()
    outputs = []
    for index, raster in enumerate((rasterize, rasterize_lanczos)):
        (tmp_path / str(index)).mkdir()
        rgb, mask = buffers(tmp_path / str(index), plan)
        raster(source, rgb, mask, plan)
        outputs.append(rgb[y : y + height, x : x + width].copy())
    np.testing.assert_allclose(outputs[1], outputs[0], atol=1e-3)


@pytest.mark.parametrize("value", [np.nan, np.inf, -0.1, 1.1])
def test_bad_pixels(tmp_path, value):
    plan = GeometryPlanner().plan(2, 2, target_megapixels=16 / 1e6)
    source = np.full((2, 2, 3), value, dtype=np.float32)
    rgb, mask = buffers(tmp_path, plan)
    with pytest.raises(ValueError, match="normalized"):
        rasterize(source, rgb, mask, plan)
