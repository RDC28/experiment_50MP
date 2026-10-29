"""Tiny executable residual dispatch, confidence gates and independent outputs."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from prd_v5.config import AnalyticRunConfig
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.learned import ResidualAdapter, apply_residual, camera_domain, resolve_mode
from prd_v5.paired import run_analytic_pair


def adapter_for(tmp_path, camera, predict=None):
    checkpoint = tmp_path / "double.bin"
    checkpoint.write_bytes(b"test double, not a research checkpoint")
    return ResidualAdapter(
        predict=predict or (lambda rgb, clean, origin, camera: (np.full_like(clean, 0.1), np.ones(clean.shape[:2]))),
        checkpoint_path=str(checkpoint),
        sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        camera_sha256=camera_domain(camera),
        source="test double",
        licence="test only",
        licence_accepted=True,
        experimental_accepted=True,
    )


def test_gating_tiles_and_units(tmp_path):
    camera = CameraHypothesis()
    origins = []

    def predict(rgb, clean, origin, supplied_camera):
        assert supplied_camera is camera
        assert not clean.flags.writeable and not rgb.flags.writeable
        origins.append(origin)
        confidence = np.full(clean.shape[:2], 0.75)
        confidence[:, 0] = 0.25
        return np.full_like(clean, 0.2), confidence

    adapter = adapter_for(tmp_path, camera, predict)
    clean = np.memmap(tmp_path / "clean", mode="w+", dtype="float32", shape=(3, 4, 3))
    try:
        clean[:] = 0.1
        report = apply_residual(np.zeros_like(clean), clean, camera, adapter, tile=2)
        expected = np.full((3, 4, 3), 0.25)
        expected[:, ::2] = 0.1
        np.testing.assert_allclose(clean, expected)
        assert origins == [(0, 0), (0, 2), (2, 0), (2, 2)]
        assert report["accepted_pixels"] == 6
        assert report["changed_samples"] == 18
        assert report["sensor_noise_in_network"] is False
    finally:
        clean._mmap.close()


@pytest.mark.parametrize(
    "change",
    [
        {"sha256": "0" * 64},
        {"camera_sha256": "0" * 64},
        {"normalization": "electrons"},
        {"experimental_accepted": False},
        {"licence_accepted": False},
    ],
)
def test_contract_failures(tmp_path, change):
    camera = CameraHypothesis()
    with pytest.raises(ValueError):
        resolve_mode("HYBRID", replace(adapter_for(tmp_path, camera), **change), camera)


@pytest.mark.parametrize("bad", ["shape", "nan", "confidence"])
def test_bad_predictions(tmp_path, bad):
    camera = CameraHypothesis()

    def predict(rgb, clean, origin, camera):
        residual = np.zeros_like(clean)
        confidence = np.ones(clean.shape[:2])
        if bad == "shape":
            residual = residual[..., 0]
        elif bad == "nan":
            residual[:] = np.nan
        else:
            confidence[:] = 2
        return residual, confidence

    clean = np.memmap(tmp_path / "clean", mode="w+", dtype="float32", shape=(2, 2, 3))
    try:
        clean[:] = 0.1
        with pytest.raises(ValueError):
            apply_residual(np.zeros_like(clean), clean, camera, adapter_for(tmp_path, camera, predict))
        np.testing.assert_allclose(clean, 0.1)
    finally:
        clean._mmap.close()


def test_strict_failure_before_io_and_hybrid_skip(tmp_path):
    config = AnalyticRunConfig()
    with pytest.raises(ValueError, match="Strict learned"):
        run_analytic_pair("missing", "missing", tmp_path / "out", config, inverse_mode="LEARNED_RGB2RAW")
    assert not (tmp_path / "out").exists()
    state = resolve_mode("HYBRID", None, config.camera)
    assert state["effective"] == "ANALYTIC" and "skipped" in state["status"]


def test_paired_residual_adc_effect_and_provenance(tmp_path):
    camera = CameraHypothesis(shot_noise=False, read_noise=False)
    source = tmp_path / "input.png"
    Image.fromarray(np.full((32, 32, 3), 100, dtype=np.uint8)).save(source)
    config = AnalyticRunConfig(camera=camera, target_megapixels=0.001024, assume_unprofiled_srgb=True)
    baseline = tmp_path / "baseline"
    output = tmp_path / "learned"
    run_analytic_pair(source, source, baseline, config)
    adapter = adapter_for(tmp_path, camera)
    result = run_analytic_pair(source, source, output, config, inverse_mode="LEARNED_RGB2RAW", residual_adapter=adapter)
    import rawpy

    for label in ("2k", "4k"):
        with rawpy.imread(str(baseline / label / "capture.dng")) as before:
            original = before.raw_image.copy()
        with rawpy.imread(str(output / label / "capture.dng")) as after:
            delta = after.raw_image.astype(float) - original
        expected = 0.1 * camera.full_well_e * camera.analogue_gain / camera.conversion_gain_e_per_dn
        np.testing.assert_allclose(delta, expected, atol=1)
        child = json.loads((output / label / "manifest.json").read_text())
        stage = child["stages"]["learned_raw"]
        assert stage["changed_samples"] == 32 * 32 * 3
        assert stage["checkpoint"]["sha256"] == adapter.sha256
        assert child["stages"]["roundtrip"]["tiff_roundtrip"].startswith("passed")
        assert child["replay"]["additional_checkpoint_required"]
    assert result["inverse_isp"]["requested"] == "LEARNED_RGB2RAW"
    assert result["inverse_isp"]["effective"] == "HYBRID"


def test_failed_adapter_withholds_completion_and_cleans_scratch(tmp_path):
    camera = CameraHypothesis()
    source = tmp_path / "source.png"
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(source)

    def fail(*args):
        raise RuntimeError("adapter failed")

    config = AnalyticRunConfig(camera=camera, target_megapixels=0.001024, assume_unprofiled_srgb=True)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="adapter failed"):
        run_analytic_pair(
            source, source, output, config, inverse_mode="HYBRID", residual_adapter=adapter_for(tmp_path, camera, fail)
        )
    assert not (output / "pair_manifest.json").exists()
    assert not (output / "2k/manifest.json").exists()
    assert not list((output / "2k").glob("working-*"))
    assert not (output / "4k").exists()
