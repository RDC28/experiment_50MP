"""Registry arithmetic and epistemic status without frame processing."""

import json

from PIL import Image

from prd_v5.analytic import AnalyticISP
from prd_v5.geometry import GeometryPlanner
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.inputs import InputInspector
from prd_v5.registry import analytic_registry


def test_nondefault_hypotheses_and_detached_records(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (16, 12)).save(path)
    source = InputInspector().inspect(path, assume_unprofiled_srgb=True)
    geometry = GeometryPlanner().plan(16, 12, target_megapixels=0.000768)
    camera = CameraHypothesis(iso=800)
    isp = AnalyticISP(white_balance=(2, 1, 1.5), exposure_gain=0.5, inverse_smoothstep=True)
    registry = analytic_registry(camera, source, geometry, isp)
    saved = json.loads(json.dumps(registry, allow_nan=False))
    assert saved["isp.white_balance"]["value"] == [2, 1, 1.5]
    assert saved["isp.exposure_gain"]["value"] == 0.5
    assert saved["isp.inverse_smoothstep"]["value"] is True
    assert saved["geometry.actual_megapixels"]["value"] == geometry.width * geometry.height / 1e6
    assert saved["geometry.source_crop"]["value"] == list(geometry.source_crop)
    assert saved["analogue_gain"]["value"] == 8
    registry["isp.white_balance"]["value"] = [99]
    registry["gps"]["value"] = "invented"
    fresh = analytic_registry(camera, source, geometry, isp)
    assert fresh["isp.white_balance"]["value"] == (2, 1, 1.5)
    assert fresh["gps"]["value"] is None
    assert fresh["gps"]["mode"] == "USER"
