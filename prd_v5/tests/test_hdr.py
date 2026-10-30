"""Experimental analytic HDR arithmetic and actual independent RAW output."""

import json

import numpy as np
import pytest
import rawpy
from PIL import Image

from prd_v5.analytic import AnalyticISP
from prd_v5.config import AnalyticRunConfig
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.paired import run_analytic_pair


def test_inverse_hdr_known_curve_and_strip_invariance(tmp_path):
    radiance = np.array([0, 0.01, 0.5, 1, 4, 16], dtype=np.float64)
    linear = radiance * (1 + 1 / 16) / (1 + radiance)
    encoded = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    source = np.broadcast_to(encoded[:, None, None], (6, 2, 3)).astype(np.float32)
    destination = np.memmap(tmp_path / "hdr.bin", mode="w+", dtype=np.float32, shape=source.shape)
    try:
        isp = AnalyticISP(hdr_max_radiance=16)
        report = isp.process(source, destination, strip_rows=1)
        np.testing.assert_allclose(destination[:, 0, 0], radiance / 16, rtol=2e-5, atol=1e-7)
        assert report["hdr"]["expanded_pixels"] == 4
        assert report["source_clipped_pixels"] == 2
        assert report["forward_consistency"]["status"] == "passed"
        first = destination.copy()
        isp.process(source, destination, strip_rows=6)
        np.testing.assert_array_equal(first, destination)
        json.dumps(report, allow_nan=False)
    finally:
        destination._mmap.close()


@pytest.mark.parametrize("value", [True, None, 0, -1, float("nan"), float("inf"), 33, "16"])
def test_invalid_hdr(value):
    with pytest.raises(ValueError, match="hdr_max_radiance"):
        AnalyticISP(hdr_max_radiance=value)


def test_replayed_hdr_above_cap_explains_grey_floor():
    config = json.loads(AnalyticRunConfig().to_json())
    config["isp"]["hdr_max_radiance"] = 64
    with pytest.raises(ValueError, match="18% grey"):
        AnalyticRunConfig.from_json(json.dumps(config))


def test_hdr_pair_pixels_replay_and_uncertainty(tmp_path):
    source = tmp_path / "source.png"
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(source)
    camera = CameraHypothesis(shot_noise=False, read_noise=False)
    config = AnalyticRunConfig(
        camera=camera,
        isp=AnalyticISP(hdr_max_radiance=16),
        target_megapixels=0.001024,
        assume_unprofiled_srgb=True,
    )
    assert AnalyticRunConfig.from_json(config.to_json()) == config
    run_analytic_pair(source, source, tmp_path / "pair", config)
    # Locate child bundles independently of label spelling.
    children = sorted((tmp_path / "pair").glob("*/capture.dng"))
    assert len(children) == 2
    encoded = 128 / 255
    linear = ((encoded + 0.055) / 1.055) ** 2.4
    radiance = linear / (1 + 1 / 16 - linear)
    expected = round(camera.black_level + radiance / 16 * camera.full_well_e / camera.conversion_gain_e_per_dn)
    for dng in children:
        with rawpy.imread(str(dng)) as raw:
            assert np.median(raw.raw_image) == pytest.approx(expected, abs=1)
            assert raw.raw_image.dtype == np.uint16
        manifest = json.loads((dng.parent / "manifest.json").read_text())
        hdr = manifest["stages"]["analytic"]["hdr"]
        assert hdr["effective"] and hdr["max_linear_srgb_radiance"] == 16
        assert "unknown" in hdr["uncertainty"]
        assert manifest["stages"]["roundtrip"]["tiff_roundtrip"].startswith("passed")
        assert json.loads((dng.parent / "config.json").read_text())["isp"]["hdr_max_radiance"] == 16


def test_hdr_headroom_keeps_midtones_anchored_and_highlights_unclipped(tmp_path):
    import tifffile

    source = tmp_path / "source.png"
    values = np.array([118, 250], dtype=np.uint8)
    Image.fromarray(np.broadcast_to(values[:, None, None], (2, 32, 3)).repeat(16, axis=0).copy()).save(source)
    camera = CameraHypothesis(shot_noise=False, read_noise=False)
    isp = AnalyticISP(hdr_max_radiance=16)
    config = AnalyticRunConfig(camera=camera, isp=isp, target_megapixels=0.001024, assume_unprofiled_srgb=True)
    output = tmp_path / "run"
    config.run(source, output)
    with rawpy.imread(str(output / "capture.dng")) as raw:
        highlights = raw.raw_image[20:28]
        assert np.all(highlights < camera.white_level)
    with tifffile.TiffFile(next(output.glob("*.dng"))) as handle:
        baseline = np.asarray(handle.pages[0].tags["BaselineExposure"].value, dtype=float)
        assert baseline[0] / baseline[1] == pytest.approx(isp.baseline_exposure_ev, abs=1e-6)
    developed = tifffile.imread(next(output.glob("*.tif*")))
    assert np.median(developed[4:12]) / 65535 == pytest.approx(118 / 255, abs=0.01)
