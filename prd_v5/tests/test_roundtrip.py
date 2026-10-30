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
