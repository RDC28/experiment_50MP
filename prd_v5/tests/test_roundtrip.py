"""Real tiny LibRaw reads and TIFF exports, with no model or full-frame work."""

import json

import numpy as np
import pytest
import tifffile

from prd_v5.analytic import AnalyticISP
from prd_v5.dng import write_dng
from prd_v5.geometry import GeometryPlanner
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.roundtrip import develop_roundtrip
from synthraw.libraw_validator import HardValidationError


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_real_roundtrip(tmp_path, pattern):
    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    camera = CameraHypothesis(cfa_pattern=pattern)
    isp = AnalyticISP(white_balance=(2, 1, 1.5))
    mosaic = (np.arange(64 * 48).reshape(48, 64) + 800).astype(np.uint16)
    source, destination = tmp_path / "source.dng", tmp_path / "render.tiff"
    write_dng(source, mosaic, camera, plan, isp)
    report = develop_roundtrip(source, destination, mosaic, camera, plan, isp)
    json.dumps(report, allow_nan=False)
    rgb = tifffile.imread(destination)
    assert rgb.shape == (48, 64, 3)
    assert rgb.dtype == np.uint16
    assert np.ptp(rgb) > 0
    assert report["tiff_roundtrip"].startswith("passed")
    with pytest.raises(HardValidationError, match="pixels differ"):
        develop_roundtrip(source, tmp_path / "bad.tiff", mosaic + 1, camera, plan, isp)
    assert not (tmp_path / "bad.tiff").exists()


def test_budget_and_alias_rejected_before_decode(tmp_path):
    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    args = (tmp_path / "absent.dng", tmp_path / "out.tiff", None, CameraHypothesis(), plan, AnalyticISP())
    with pytest.raises(ValueError, match="memory budget"):
        develop_roundtrip(*args, max_working_bytes=1)
    with pytest.raises(ValueError, match="positive integer"):
        develop_roundtrip(*args, max_working_bytes=True)
    with pytest.raises(ValueError, match="differ"):
        develop_roundtrip(args[0], args[0], *args[2:])


def test_exposure_sweep_real_dng_linear_response(tmp_path):
    from prd_v5.roundtrip import develop_exposure_sweep

    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    camera, isp = CameraHypothesis(), AnalyticISP()
    mosaic = np.full((48, 64), camera.black_level + 100, dtype=np.uint16)
    source = tmp_path / "source.dng"
    write_dng(source, mosaic, camera, plan, isp)
    original = source.read_bytes()
    report = develop_exposure_sweep(source, tmp_path / "edits")
    json.dumps(report, allow_nan=False)
    assert [item["ev"] for item in report["renders"]] == [-3, -2, 0, 2, 3]
    images = [tifffile.imread(item["path"]) for item in report["renders"]]
    baseline = images[2].astype(float)
    for item, rgb in zip(report["renders"], images, strict=True):
        assert rgb.dtype == np.uint16 and rgb.shape == (48, 64, 3)
        np.testing.assert_allclose(rgb, baseline * 2.0 ** item["ev"], atol=8)
        assert item["saturated_samples"] == 0
    assert source.read_bytes() == original
    assert report["highlight_recovery"] == report["physical_dynamic_range"] == "not measured"
    with pytest.raises(ValueError, match="memory budget"):
        develop_exposure_sweep(source, tmp_path / "budget", max_working_bytes=1)
    assert not (tmp_path / "budget").exists()
    with pytest.raises(FileExistsError):
        develop_exposure_sweep(source, tmp_path / "edits")
    assert len(list((tmp_path / "edits").iterdir())) == 5


def test_sweep_failure_removes_only_its_new_outputs(tmp_path, monkeypatch):
    from prd_v5.roundtrip import develop_exposure_sweep

    plan = GeometryPlanner().plan(64, 48, target_megapixels=0.003072)
    source = tmp_path / "source.dng"
    write_dng(source, np.full((48, 64), 1000, np.uint16), CameraHypothesis(), plan, AnalyticISP())
    original_write = tifffile.imwrite
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(args[0])
        if len(calls) == 2:
            raise OSError("fixture disk failure")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(tifffile, "imwrite", fail_second)
    with pytest.raises(OSError, match="fixture disk failure"):
        develop_exposure_sweep(source, tmp_path / "failed")
    assert not (tmp_path / "failed").exists()
    assert source.is_file()


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), 17, "0"])
def test_sweep_invalid_baseline_fails_before_io(tmp_path, value):
    from prd_v5.roundtrip import develop_exposure_sweep

    with pytest.raises(ValueError, match="baseline_exposure_ev"):
        develop_exposure_sweep(tmp_path / "absent", tmp_path / "edits", baseline_exposure_ev=value)
    assert not (tmp_path / "edits").exists()
