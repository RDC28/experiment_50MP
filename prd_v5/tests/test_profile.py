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


def test_catalogue_profile_reaches_paired_dng_and_selection_evidence(tmp_path):
    import tifffile
    from PIL import Image

    from prd_v5.config import AnalyticRunConfig
    from prd_v5.paired import run_analytic_pair
    from prd_v5.panel import resolve_analytic_panel_from_source

    matrix = np.linalg.inv(np.asarray(SRGB_TO_XYZ_D65) @ np.diag([0.8, 1, 1.2]))
    camera, isp, camera_records, isp_records = resolve_analytic_panel_from_source(
        CameraHypothesis(shot_noise=False, read_noise=False),
        AnalyticISP(),
        [
            {"name": "ColorMatrix1/2", "mode": "AUTO", "value": matrix.tolist()},
            {"name": "AsShotNeutral", "mode": "AUTO", "value": [0.5, 1, 0.25]},
        ],
        {"ColorMatrix1/2": "USER", "AsShotNeutral": "USER"},
    )
    assert isp.white_balance == (2, 1, 4)
    evidence = validate_selections(isp, isp_records, isp=True)["isp.camera_to_srgb"]
    assert evidence["provenance"] == "USER_SET"
    assert evidence["posterior"]["illuminant"] == "D65"
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(source)
    output = tmp_path / "pair"
    run_analytic_pair(
        source,
        source,
        output,
        AnalyticRunConfig(camera=camera, isp=isp, target_megapixels=0.001024, assume_unprofiled_srgb=True),
        camera_selections=camera_records,
        isp_selections=isp_records,
    )
    dngs = list(output.rglob("*.dng"))
    assert len(dngs) == 2
    for path in dngs:
        with tifffile.TiffFile(path) as handle:
            tags = handle.pages[0].tags
            for name in ("ColorMatrix1", "ColorMatrix2"):
                rational = np.asarray(tags[name].value).reshape(9, 2)
                np.testing.assert_allclose((rational[:, 0] / rational[:, 1]).reshape(3, 3), matrix, atol=1e-6)
    manifests = [json.loads(path.read_text()) for path in output.rglob("manifest.json")]
    assert len(manifests) == 2
    for manifest in manifests:
        assert "user_supplied_d65_xyz_to_camera" in json.dumps(manifest)


@pytest.mark.parametrize(
    "mode,value,error",
    [
        ("AUTO", np.eye(3).tolist(), "requires USER"),
        ("FABRICATE", np.eye(3).tolist(), "requires USER"),
        ("USER", None, "Unresolved ISP"),
        ("ESTIMATE", None, "Unresolved ISP"),
        ("USER", np.zeros((3, 3)).tolist(), "well-conditioned"),
    ],
)
def test_catalogue_profile_rejects_unsupported_evidence(mode, value, error):
    from prd_v5.panel import resolve_analytic_panel

    with pytest.raises(ValueError, match=error):
        resolve_analytic_panel(
            CameraHypothesis(),
            AnalyticISP(),
            [{"name": "ColorMatrix1/2", "mode": "AUTO", "value": value}],
            {"ColorMatrix1/2": mode},
        )


@pytest.mark.parametrize("mode", ["AUTO", "FABRICATE"])
def test_catalogue_profile_without_value_preserves_base(mode):
    from prd_v5.panel import resolve_analytic_panel

    base = AnalyticISP()
    _, isp, _, records = resolve_analytic_panel(
        CameraHypothesis(),
        base,
        [{"name": "ColorMatrix1/2", "mode": "AUTO", "value": None}],
        {"ColorMatrix1/2": mode},
    )
    assert isp == base
    assert next(record for record in records if record.name == "camera_to_srgb").mode == mode
