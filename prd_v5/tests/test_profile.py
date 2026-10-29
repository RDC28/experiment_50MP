"""Independent matrix convention, pixel effect and provenance checks."""

import json

import numpy as np
import pytest

from prd_v5.analytic import AnalyticISP
from prd_v5.dng import SRGB_TO_XYZ_D65, build_metadata
from prd_v5.fields import validate_selections
from prd_v5.geometry import GeometryPlanner
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.profile import resolve_d65_profile


def test_profile_conversion_changes_pixels_and_preserves_dng_convention(tmp_path):
    expected = np.diag([0.8, 1.0, 1.2])
    supplied = np.linalg.inv(np.asarray(SRGB_TO_XYZ_D65) @ expected)
    base = AnalyticISP(white_balance=(2, 1, 1.5), exposure_gain=2)
    isp, records, report = resolve_d65_profile(base, supplied, label="User test hypothesis")
    np.testing.assert_allclose(isp.camera_to_srgb, expected, atol=1e-14)
    assert isp.white_balance == base.white_balance and isp.exposure_gain == base.exposure_gain
    evidence = validate_selections(isp, records, isp=True)["isp.camera_to_srgb"]
    assert evidence["provenance"] == "USER_SET" and evidence["confidence"] is None
    json.dumps(report, allow_nan=False)
    supplied[:] = 0
    assert np.any(report["xyz_to_camera"])
    source = np.full((2, 2, 3), 0.4, dtype=np.float32)
    outputs = []
    for index, selected in enumerate((base, isp)):
        dest = np.memmap(tmp_path / str(index), mode="w+", shape=source.shape, dtype=np.float32)
        try:
            assert selected.process(source, dest)["forward_consistency"]["status"] == "passed"
            outputs.append(dest.copy())
        finally:
            dest._mmap.close()
    np.testing.assert_allclose(outputs[1], outputs[0] / np.diag(expected), rtol=1e-6)
    geometry = GeometryPlanner().plan(32, 24, target_megapixels=0.000768)
    tags = build_metadata(CameraHypothesis(), geometry, isp)
    for index in (1, 2):
        np.testing.assert_allclose(
            np.asarray(tags[f"ColorMatrix{index}"].value).reshape(3, 3), report["xyz_to_camera"], atol=1e-14
        )
        assert tags[f"CalibrationIlluminant{index}"].value == 21


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.eye(2), np.full((3, 3), np.nan), np.diag([1, 1, 1e-8])])
def test_invalid_profile_fails(matrix):
    with pytest.raises(ValueError, match="well-conditioned"):
        resolve_d65_profile(AnalyticISP(), matrix, label="invalid")


@pytest.mark.parametrize("label", [None, "", "  "])
def test_profile_requires_explicit_description(label):
    with pytest.raises(ValueError, match="label"):
        resolve_d65_profile(AnalyticISP(), np.eye(3), label=label)
