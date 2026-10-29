"""Lightweight verification of joint exposure selection and actual sensor effects."""

import json
import math

import numpy as np
import pytest

from prd_v5.exposure import resolve_exposure
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.sensor import BayerSensor


def test_joint_exposure_is_normalized_reproducible_and_preserves_camera():
    base = CameraHypothesis(seed=123, full_well_e=48000)
    camera, records, report = resolve_exposure(base, scene_ev100=4)
    again = resolve_exposure(base, scene_ev100=4)
    assert (camera, records, report) == again
    assert base.iso == 100
    assert camera.seed == base.seed and camera.full_well_e == base.full_well_e
    assert camera.cfa_pattern == base.cfa_pattern
    assert math.isclose(sum(c["weight"] for c in report["joint_candidates"]), 1)
    assert camera.iso == report["selected"]["iso"]
    assert camera.shutter_s == report["selected"]["exposure_time"]
    assert camera.aperture == report["selected"]["f_number"]
    selected = {r.name: r for r in records}
    for name in ("iso", "shutter_s", "aperture"):
        assert selected[name].provenance == "ESTIMATED"
        assert selected[name].confidence is None
        assert selected[name].posterior["joint_candidates"] == report["joint_candidates"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scene_ev100": float("nan")},
        {"scene_ev100": True},
        {"scene_ev100": 4, "candidate_count": 1},
        {"scene_ev100": 4, "electronic_shutter": 1},
    ],
)
def test_invalid_exposure_requests_fail(kwargs):
    with pytest.raises(ValueError):
        resolve_exposure(CameraHypothesis(), **kwargs)


def test_scene_ev_moves_joint_exposure():
    dark, _, _ = resolve_exposure(CameraHypothesis(), scene_ev100=0)
    bright, _, _ = resolve_exposure(CameraHypothesis(), scene_ev100=16)

    def ev(camera):
        return math.log2(camera.aperture**2 / camera.shutter_s) - math.log2(camera.iso / 100)

    assert ev(bright) > ev(dark) + 10


def test_selected_iso_changes_actual_adc(tmp_path):
    base = CameraHypothesis(shot_noise=False, read_noise=False)
    camera, _, _ = resolve_exposure(base, scene_ev100=-8)
    assert camera.iso > base.iso
    source = np.full((4, 4, 3), 0.001, dtype=np.float32)
    outputs = []
    for index, hypothesis in enumerate((base, camera)):
        dest = np.memmap(tmp_path / str(index), mode="w+", shape=(4, 4), dtype=np.uint16)
        try:
            BayerSensor(hypothesis).process(source, dest)
            outputs.append(dest.copy())
        finally:
            dest._mmap.close()
    assert np.all(outputs[1] > outputs[0])
    assert camera.analogue_gain > base.analogue_gain


def test_exposure_selection_survives_real_dng_tiff_bundle(tmp_path):
    from PIL import Image

    from prd_v5.pipeline import run_analytic

    camera, records, report = resolve_exposure(CameraHypothesis(), scene_ev100=4)
    source = tmp_path / "source.png"
    Image.fromarray(np.full((32, 32, 3), 64, dtype=np.uint8)).save(source)
    result = run_analytic(
        source,
        tmp_path / "run",
        camera,
        target_megapixels=0.001024,
        assume_unprofiled_srgb=True,
        camera_selections=records,
    )
    assert result["stages"]["roundtrip"]["tiff_roundtrip"].startswith("passed")
    for name in ("iso", "shutter_s", "aperture"):
        field = result["registry"][name]
        assert field["value"] == getattr(camera, name)
        assert field["provenance"] == "ESTIMATED"
        assert field["posterior"]["joint_candidates"] == report["joint_candidates"]
    assert result["registry"]["iso"]["confidence"] is None
