"""Lightweight checkpoint integrity and actual disk-backed tile execution."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from prd_v5 import fidelity


def checkpoint(tmp_path):
    path = tmp_path / "double.weights"
    path.write_bytes(b"model double, not a real checkpoint")
    return fidelity.SRCheckpoint(
        str(path),
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "HAT",
        "a" * 40,
        "test fixture",
        "test only",
        True,
    )


def test_integrity_and_explicit_acceptance(tmp_path):
    spec = checkpoint(tmp_path)
    assert spec.verify()["integrity"] == "verified"
    for altered in (
        replace(spec, licence_accepted=False),
        replace(spec, scale=2),
        replace(spec, sha256="b" * 64),
        replace(spec, repository_revision="main"),
    ):
        with pytest.raises(ValueError):
            altered.verify()


def test_real_loader_is_guarded_locally(tmp_path, monkeypatch):
    import synthraw.model_manifest as manifest

    def reject():
        raise RuntimeError("Colab-only")

    monkeypatch.setattr(manifest, "_require_colab", reject)
    with pytest.raises(RuntimeError, match="Colab-only"):
        fidelity.load_checkpoint(checkpoint(tmp_path))


def test_tiled_double_and_visible_oom_retry(tmp_path, monkeypatch):
    spec = checkpoint(tmp_path)
    source = np.random.default_rng(42).random((7, 9, 3), dtype=np.float32)
    calls = []

    def infer(patch):
        calls.append(patch.shape)
        if len(calls) == 1:
            raise MemoryError("test tile admission")
        return np.repeat(np.repeat(patch, 4, axis=0), 4, axis=1)

    monkeypatch.setattr(fidelity, "load_checkpoint", lambda spec: (infer, spec.verify()))
    output = tmp_path / "sr.npy"
    report = fidelity.reconstruct(source, output, spec, tile=4, overlap=1)
    result = np.load(output, mmap_mode="r")
    np.testing.assert_allclose(result, np.repeat(np.repeat(source, 4, axis=0), 4, axis=1), atol=2e-7)
    assert report["effective_tile"] == 2 and len(report["retry_reasons"]) == 1
    assert report["effective"] == "HAT" and report["fallback"] is None
    assert not output.with_name(output.name + ".weights.npy").exists()
    result._mmap.close()


def test_model_overshoot_is_clamped_and_counted(tmp_path, monkeypatch):
    def infer(patch):
        prediction = np.repeat(np.repeat(patch, 4, axis=0), 4, axis=1)
        prediction[0, 0] = [-0.25, 1.5, 0.5]
        return prediction

    monkeypatch.setattr(fidelity, "load_checkpoint", lambda spec: (infer, {}))
    output = tmp_path / "sr.npy"
    report = fidelity.reconstruct(
        np.full((2, 2, 3), 0.5, dtype=np.float32), output, checkpoint(tmp_path), tile=4, overlap=0
    )
    result = np.load(output, mmap_mode="r")
    try:
        assert result.min() >= 0 and result.max() <= 1
        np.testing.assert_allclose(result[0, 0], [0, 1, 0.5], atol=1e-7)
        assert report["clamped_samples"] == 2
    finally:
        result._mmap.close()


def test_invalid_model_output_cleans_partial_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(fidelity, "load_checkpoint", lambda spec: (lambda patch: patch, {}))
    output = tmp_path / "sr.npy"
    with pytest.raises(ValueError, match="incompatible shape"):
        fidelity.reconstruct(np.zeros((3, 3, 3), dtype=np.float32), output, checkpoint(tmp_path), tile=2, overlap=1)
    assert not output.exists()
    assert not output.with_name(output.name + ".weights.npy").exists()


def fake_loader(monkeypatch, architecture, scale, channels):
    from types import SimpleNamespace

    import synthraw.model_manifest as manifest

    class Descriptor:
        supports_half = False

        def eval(self):
            return self

        def to(self, **kwargs):
            return self

    descriptor = Descriptor()
    descriptor.architecture = SimpleNamespace(id=architecture)
    descriptor.scale = scale
    descriptor.input_channels = channels
    descriptor.output_channels = 3
    loads = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: False),
        load=lambda *args, **kwargs: loads.append(kwargs) or {},
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
    )
    spandrel = SimpleNamespace(
        ImageModelDescriptor=Descriptor,
        ModelLoader=lambda: SimpleNamespace(load_from_state_dict=lambda state: descriptor),
    )
    monkeypatch.setattr(manifest, "_require_colab", lambda: None)
    monkeypatch.setattr(fidelity.importlib, "import_module", lambda name: {"torch": torch, "spandrel": spandrel}[name])
    return loads


@pytest.mark.parametrize("architecture,scale,channels", [("wrong", 4, 3), ("HAT", 2, 3), ("HAT", 4, 1)])
def test_loader_rejects_incompatible_descriptor(tmp_path, monkeypatch, architecture, scale, channels):
    loads = fake_loader(monkeypatch, architecture, scale, channels)
    with pytest.raises(ValueError, match="contract mismatch"):
        fidelity.load_checkpoint(checkpoint(tmp_path))
    assert loads == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize("backend,architecture", [("HAT", "HAT"), ("REAL_ESRGAN", "ESRGAN")])
def test_loader_accepts_spandrel_architecture_ids(tmp_path, monkeypatch, backend, architecture):
    fake_loader(monkeypatch, architecture, 4, 3)
    _, evidence = fidelity.load_checkpoint(replace(checkpoint(tmp_path), backend=backend))
    assert evidence["architecture"] == architecture


def test_checkpoint_pixels_reach_independent_raw_roundtrip(tmp_path, monkeypatch):
    import rawpy
    from PIL import Image

    from prd_v5.hypothesis import CameraHypothesis
    from prd_v5.pipeline import run_analytic

    source = tmp_path / "source.png"
    Image.fromarray(np.full((16, 24, 3), 128, dtype=np.uint8)).save(source)
    spec = checkpoint(tmp_path)

    def infer(patch):
        return np.repeat(np.repeat(patch * 0.5, 4, axis=0), 4, axis=1)

    monkeypatch.setattr(fidelity, "load_checkpoint", lambda spec: (infer, spec.verify()))
    camera = CameraHypothesis(shot_noise=False, read_noise=False)
    baseline = run_analytic(
        source, tmp_path / "baseline", camera, target_megapixels=0.001536, assume_unprofiled_srgb=True
    )
    report = run_analytic(
        source,
        tmp_path / "sr",
        camera,
        target_megapixels=0.001536,
        assume_unprofiled_srgb=True,
        sr_checkpoint=spec,
    )
    assert report["geometry"] == baseline["geometry"]
    assert report["stages"]["super_resolution"]["checkpoint"]["sha256"] == spec.sha256
    assert report["stages"]["raster"]["learned_super_resolution"] == "HAT"
    assert report["stages"]["roundtrip"]["tiff_roundtrip"].startswith("passed")
    assert report["replay"]["additional_checkpoint_required"] is True
    assert report["scientific_acceptance"].startswith("conditional:")
    with rawpy.imread(str(tmp_path / "baseline/capture.dng")) as original:
        before = original.raw_image.copy()
    with rawpy.imread(str(tmp_path / "sr/capture.dng")) as reconstructed:
        after = reconstructed.raw_image.copy()
    # Independent sRGB inversion, not merely a changed stage label.
    encoded = 128 / 255

    def linear(x):
        return ((x + 0.055) / 1.055) ** 2.4

    expected_ratio = linear(encoded * 0.5) / linear(encoded)
    np.testing.assert_allclose(
        after.astype(float) - camera.black_level,
        (before.astype(float) - camera.black_level) * expected_ratio,
        atol=1,
    )
    assert not list((tmp_path / "sr").glob("working-*"))


def test_pipeline_checkpoint_failure_never_falls_back(tmp_path, monkeypatch):
    from PIL import Image

    from prd_v5.hypothesis import CameraHypothesis
    from prd_v5.pipeline import run_analytic

    source = tmp_path / "source.png"
    Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(source)

    def fail(spec):
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(fidelity, "load_checkpoint", fail)
    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        run_analytic(
            source,
            tmp_path / "out",
            CameraHypothesis(),
            target_megapixels=0.001024,
            assume_unprofiled_srgb=True,
            sr_checkpoint=checkpoint(tmp_path),
        )
    assert list((tmp_path / "out").iterdir()) == []


def test_pair_checkpoint_is_shared_and_pixels_reach_both_bundles(tmp_path, monkeypatch):
    import json

    import rawpy
    from PIL import Image

    from prd_v5.config import AnalyticRunConfig
    from prd_v5.hypothesis import CameraHypothesis
    from prd_v5.paired import run_analytic_pair

    sources = [tmp_path / name for name in ("low.png", "high.png")]
    for source, size in zip(sources, (16, 32), strict=True):
        Image.new("RGB", (size, size), (128, 128, 128)).save(source)
    spec = checkpoint(tmp_path)
    loads = []

    def load(selected):
        loads.append(selected)
        # A deterministic black prediction distinguishes execution from bilinear input.
        return lambda patch: np.zeros((patch.shape[0] * 4, patch.shape[1] * 4, 3), np.float32), selected.verify()

    monkeypatch.setattr(fidelity, "load_checkpoint", load)
    config = AnalyticRunConfig(
        camera=CameraHypothesis(shot_noise=False, read_noise=False),
        target_megapixels=0.001024,
        assume_unprofiled_srgb=True,
    )
    output = tmp_path / "pair"
    report = run_analytic_pair(*sources, output, config, sr_checkpoint=spec)
    assert loads == [spec, spec]
    assert all(selected is spec for selected in loads)
    assert report["shared_checkpoint"] == spec.verify()
    assert report["replay"]["additional_checkpoint_required"] is True
    assert "no learned inference" not in report["pipeline"]
    assert json.loads((output / "pair_manifest.json").read_text()) == report
    for label in ("2k", "4k"):
        child = json.loads((output / label / "manifest.json").read_text())
        assert child["camera"] == report["shared_camera"]
        assert child["stages"]["super_resolution"]["checkpoint"] == report["shared_checkpoint"]
        assert child["stages"]["roundtrip"]["tiff_roundtrip"].startswith("passed")
        with rawpy.imread(str(output / label / "capture.dng")) as raw:
            assert np.all(raw.raw_image == config.camera.black_level)
        assert not list((output / label).glob("working-*"))


def test_pair_invalid_checkpoint_fails_before_output_creation(tmp_path):
    from prd_v5.config import AnalyticRunConfig
    from prd_v5.paired import run_analytic_pair

    output = tmp_path / "pair"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_analytic_pair(
            "missing2k",
            "missing4k",
            output,
            AnalyticRunConfig(),
            sr_checkpoint=replace(checkpoint(tmp_path), sha256="b" * 64),
        )
    assert not output.exists()


def test_pair_second_checkpoint_failure_never_claims_completion(tmp_path, monkeypatch):
    from PIL import Image

    from prd_v5.config import AnalyticRunConfig
    from prd_v5.paired import run_analytic_pair

    source = tmp_path / "source.png"
    Image.new("RGB", (16, 16), (128, 128, 128)).save(source)
    calls = []

    def load(selected):
        calls.append(selected)
        if len(calls) == 2:
            raise RuntimeError("second checkpoint load failed")
        return lambda patch: np.repeat(np.repeat(patch, 4, axis=0), 4, axis=1), selected.verify()

    monkeypatch.setattr(fidelity, "load_checkpoint", load)
    output = tmp_path / "pair"
    config = AnalyticRunConfig(target_megapixels=0.001024, assume_unprofiled_srgb=True)
    with pytest.raises(RuntimeError, match="second checkpoint load failed"):
        run_analytic_pair(source, source, output, config, sr_checkpoint=checkpoint(tmp_path))
    assert (output / "2k/manifest.json").is_file()
    assert not (output / "4k/manifest.json").exists()
    assert not (output / "pair_manifest.json").exists()
    assert list((output / "4k").iterdir()) == []
