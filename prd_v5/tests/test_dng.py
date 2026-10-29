"""Tiny real container exports, without independent decoder acceptance claims."""

import json

import numpy as np
import pytest
import tifffile

from prd_v5.analytic import AnalyticISP
from prd_v5.dng import SRGB_TO_XYZ_D65, build_metadata, write_dng
from prd_v5.geometry import GeometryPlanner
from prd_v5.hypothesis import CameraHypothesis


def geometry():
    return GeometryPlanner().plan(32, 24, target_megapixels=0.000768)


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_container_exact_samples_and_shared_tags(tmp_path, pattern):
    camera = CameraHypothesis(cfa_pattern=pattern, iso=400, black_level=600)
    plan = geometry()
    mosaic = np.arange(plan.width * plan.height, dtype=np.uint16).reshape(plan.height, plan.width) + 600
    report = write_dng(tmp_path / "tiny.dng", mosaic, camera, plan, AnalyticISP(), strip_rows=3)
    json.dumps(report, allow_nan=False)
    with tifffile.TiffFile(report["path"]) as container:
        page = container.pages[0]
        np.testing.assert_array_equal(page.asarray(), mosaic)
        assert page.tags["WhiteLevel"].value == camera.white_level
        assert page.tags["BitsPerSample"].value == 16
        assert page.tags["NoiseProfile"].value == camera.noise_profile
        assert page.tags[tifffile.TIFF.TAGS["PhotographicSensitivity"]].value == camera.iso
        assert page.tags["DefaultCropSize"].value == plan.default_crop_size
        for forbidden in ("GPSInfo", "DateTime", "CameraSerialNumber", "MakerNote"):
            assert forbidden not in page.tags
    assert report["independent_rawpy_validation"] == "not measured"


def test_profile_is_derived_from_analytic_transform():
    isp = AnalyticISP(camera_to_srgb=((0.9, 0.1, 0), (0, 1, 0), (0, 0.1, 0.9)), white_balance=(2, 1, 1.5))
    tags = build_metadata(CameraHypothesis(), geometry(), isp)
    matrix = np.asarray(tags["ColorMatrix1"].value).reshape(3, 3)
    np.testing.assert_allclose(matrix @ np.asarray(SRGB_TO_XYZ_D65) @ isp.camera_to_srgb, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(tags["AsShotNeutral"].value, (0.5, 1, 2 / 3))
    assert tags["CalibrationIlluminant1"].value == tags["CalibrationIlluminant2"].value == 21


def test_disabled_shot_noise_omits_profile():
    assert "NoiseProfile" not in build_metadata(CameraHypothesis(shot_noise=False), geometry(), AnalyticISP())


def test_out_of_adc_range_rejected(tmp_path):
    plan = geometry()
    with pytest.raises(ValueError, match="WhiteLevel"):
        write_dng(
            tmp_path / "invalid.dng",
            np.full((plan.height, plan.width), 16384, dtype=np.uint16),
            CameraHypothesis(),
            plan,
            AnalyticISP(),
        )
    assert not (tmp_path / "invalid.dng").exists()


def test_analytic_sensor_to_dng(tmp_path):
    from prd_v5.sensor import BayerSensor

    plan = geometry()
    camera = CameraHypothesis(shot_noise=False, read_noise=False)
    isp = AnalyticISP()
    clean = np.memmap(tmp_path / "clean.bin", mode="w+", dtype=np.float32, shape=(plan.height, plan.width, 3))
    mosaic = np.memmap(tmp_path / "adc.bin", mode="w+", dtype=np.uint16, shape=(plan.height, plan.width))
    try:
        isp.process(np.full(clean.shape, 0.5, dtype=np.float32), clean, strip_rows=3)
        BayerSensor(camera).process(clean, mosaic, strip_rows=3)
        report = write_dng(tmp_path / "analytic.dng", mosaic, camera, plan, isp, strip_rows=3)
        linear = ((0.5 + 0.055) / 1.055) ** 2.4
        expected = round(camera.black_level + linear * (camera.white_level - camera.black_level))
        with tifffile.TiffFile(report["path"]) as container:
            assert np.all(container.pages[0].asarray() == expected)
    finally:
        clean._mmap.close()
        mosaic._mmap.close()
