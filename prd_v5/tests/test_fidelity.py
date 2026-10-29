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


def test_invalid_model_output_cleans_partial_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(fidelity, "load_checkpoint", lambda spec: (lambda patch: patch, {}))
    output = tmp_path / "sr.npy"
    with pytest.raises(ValueError, match="incompatible shape"):
        fidelity.reconstruct(np.zeros((3, 3, 3), dtype=np.float32), output, checkpoint(tmp_path), tile=2, overlap=1)
    assert not output.exists()
    assert not output.with_name(output.name + ".weights.npy").exists()


@pytest.mark.parametrize("architecture,scale,channels", [("wrong", 4, 3), ("HAT", 2, 3), ("HAT", 4, 1)])
def test_loader_rejects_incompatible_descriptor(tmp_path, monkeypatch, architecture, scale, channels):
    from types import SimpleNamespace

    import synthraw.model_manifest as manifest

    class Descriptor:
        pass

    descriptor = Descriptor()
    descriptor.architecture = SimpleNamespace(id=architecture)
    descriptor.scale = scale
    descriptor.input_channels = channels
    descriptor.output_channels = 3
    loads = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        load=lambda *args, **kwargs: loads.append(kwargs) or {},
    )
    spandrel = SimpleNamespace(
        ImageModelDescriptor=Descriptor,
        ModelLoader=lambda: SimpleNamespace(load_from_state_dict=lambda state: descriptor),
    )
    monkeypatch.setattr(manifest, "_require_colab", lambda: None)
    monkeypatch.setattr(fidelity.importlib, "import_module", lambda name: {"torch": torch, "spandrel": spandrel}[name])
    with pytest.raises(ValueError, match="contract mismatch"):
        fidelity.load_checkpoint(checkpoint(tmp_path))
    assert loads == [{"map_location": "cpu", "weights_only": True}]
