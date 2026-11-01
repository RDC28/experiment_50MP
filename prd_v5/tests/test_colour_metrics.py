"""Real TIFF fixtures for the calibrated colour-reference contract."""

import json

import numpy as np
import pytest
import tifffile
from skimage.color import deltaE_ciede2000

from prd_v5.colour_metrics import calibrated_delta_e

CONTRACT = {
    "camera_profile": "fixture camera / calibrated profile record",
    "trusted_development": "fixture fixed sRGB development",
    "alignment": "identical native coordinates",
    "colour_space": "sRGB_D65_2deg",
}


def image(tmp_path, name, rgb, shape=(3, 4, 3)):
    path = tmp_path / name
    tifffile.imwrite(path, np.broadcast_to(np.array(rgb, dtype=np.uint16), shape), photometric="rgb")
    return path


def test_missing_truth_never_reads_candidate():
    assert calibrated_delta_e("absent", None)["status"] == "not measured"
    assert calibrated_delta_e("absent", "absent")["status"] == "not measured"


def test_known_black_white_and_strip_crop(tmp_path):
    black = image(tmp_path, "black.tif", [0, 0, 0])
    white = image(tmp_path, "white.tif", [65535] * 3)
    report = calibrated_delta_e(black, white, calibration=CONTRACT, strip_rows=1)
    assert report["mean_delta_e2000"] == pytest.approx(100, abs=1e-4)
    assert report["pixels"] == 12
    crop = calibrated_delta_e(black, white, calibration=CONTRACT, region=(1, 1, 3, 3))
    assert crop["pixels"] == 4
    assert crop["mean_delta_e2000"] == pytest.approx(report["mean_delta_e2000"])
    same = calibrated_delta_e(white, white, calibration=CONTRACT)
    assert same["max_delta_e2000"] == 0
    json.dumps(report, allow_nan=False)


def test_ciede2000_published_sharma_pair():
    # Sharma et al. supplementary pair 1, exercises hue/chroma weighting.
    assert deltaE_ciede2000([50, 2.6772, -79.7751], [50, 0, -82.7485]) == pytest.approx(2.0425, abs=5e-5)


@pytest.mark.parametrize("contract", [{}, {**CONTRACT, "alignment": " "}, {**CONTRACT, "colour_space": "camera RGB"}])
def test_invalid_calibration_rejected(contract):
    with pytest.raises(ValueError, match="calibration|sRGB"):
        calibrated_delta_e("absent", "absent", calibration=contract)


def test_shape_and_bounds(tmp_path):
    a = image(tmp_path, "a.tif", [0] * 3)
    b = image(tmp_path, "b.tif", [0] * 3, (4, 4, 3))
    assert calibrated_delta_e(a, b, calibration=CONTRACT)["status"] == "not measured"
    with pytest.raises(ValueError, match="exceeds"):
        calibrated_delta_e(a, a, calibration=CONTRACT, region=(0, 0, 8, 8))


@pytest.mark.parametrize("rows", [True, 0, -1, 1.5])
def test_invalid_rows(rows):
    with pytest.raises(ValueError, match="strip_rows"):
        calibrated_delta_e("absent", None, strip_rows=rows)


def test_colour_conversion_is_bounded_and_rejects_wrong_dtype(tmp_path, monkeypatch):
    import prd_v5.colour_metrics as metrics

    a = image(tmp_path, "red.tif", [65535, 0, 0], (9, 4, 3))
    b = image(tmp_path, "white.tif", [65535] * 3, (9, 4, 3))
    original = metrics.rgb2lab
    shapes = []

    def bounded(array, **kwargs):
        shapes.append(array.shape)
        assert array.shape[0] <= 2
        return original(array, **kwargs)

    monkeypatch.setattr(metrics, "rgb2lab", bounded)
    report = calibrated_delta_e(a, b, calibration=CONTRACT, strip_rows=2)
    # Standard D65 sRGB Lab coordinates; allow 0.005 for differing RGB/XYZ coefficients.
    expected = deltaE_ciede2000([53.2408, 80.0925, 67.2032], [100, 0, 0])
    assert report["mean_delta_e2000"] == pytest.approx(expected, abs=0.005)
    assert len(shapes) == 10
    wrong = tmp_path / "wrong.tif"
    tifffile.imwrite(wrong, np.zeros((3, 4, 3), dtype=np.uint8), photometric="rgb")
    with pytest.raises(ValueError, match="uint16"):
        calibrated_delta_e(wrong, wrong, calibration=CONTRACT)
