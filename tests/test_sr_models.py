"""Weights registry, checksum-verified downloads, the PyTorch wrapper and a real-weights smoke test.

The download tests use local file:// URLs. The wrapper tests build a tiny random network and skip
without PyTorch and spandrel. The smoke test runs the real SwinIR networks on a small image; it is opt-in
(SYNTHETIC_RAW_LAB_HEAVY_TESTS=1) and also needs the weights cached in ``<repository>/weights`` or in
``$SYNTHETIC_RAW_LAB_WEIGHTS`` (it never downloads).
"""

import dataclasses
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import pipeline, sr_models
from synthetic_raw_lab import super_resolution as sr
from synthetic_raw_lab.test_images import make_test_image

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = Path(os.environ.get("SYNTHETIC_RAW_LAB_WEIGHTS") or sr_models.default_weights_dir(ROOT))


def test_registry_records_provenance_for_every_backend_network():
    for backend, networks in sr_models.BACKENDS.items():
        assert set(networks) == {"fidelity", "perceptual"}, backend
        for role, name in networks.items():
            spec = sr_models.WEIGHTS[name]
            assert spec.role == role
            assert spec.url.startswith("https://github.com/") and spec.filename.endswith(".pth")
            assert len(spec.sha256) == 64 and spec.size_bytes > 1_000_000
            assert spec.license and spec.training_data and "commercial" in spec.commercial_use
    assert {sr.PRESETS[m].network for m in sr.MODES} == {"fidelity", "perceptual"}


def test_weights_are_ignored_by_git():
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert "weights/" in ignored and "*.pth" in ignored
    assert sr_models.default_weights_dir(ROOT) == ROOT / "weights"


@pytest.fixture
def fake_weights(tmp_path, monkeypatch):
    """A registry entry whose 'release' is a local file, so downloads need no network."""
    payload = os.urandom(4096)
    release = tmp_path / "release" / "net.pth"
    release.parent.mkdir()
    release.write_bytes(payload)
    spec = dataclasses.replace(
        sr_models.WEIGHTS["swinir_m_x4_classical"],
        name="fake",
        url=release.as_uri(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    monkeypatch.setitem(sr_models.WEIGHTS, "fake", spec)
    return spec, payload


def test_download_verifies_the_checksum_and_caches(fake_weights, tmp_path):
    spec, payload = fake_weights
    cache = tmp_path / "cache"
    assert not sr_models.cached("fake", cache)
    path = sr_models.ensure_weights("fake", cache, log=lambda _: None)
    assert path == cache / "net.pth" and path.read_bytes() == payload
    assert sr_models.cached("fake", cache)
    assert not list(cache.glob("*.part"))


def test_corrupt_cache_is_downloaded_again(fake_weights, tmp_path):
    spec, payload = fake_weights
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "net.pth").write_bytes(b"truncated")
    messages = []
    path = sr_models.ensure_weights("fake", cache, log=messages.append)
    assert path.read_bytes() == payload
    assert any("checksum mismatch" in m for m in messages)


def test_checksum_mismatch_after_download_is_rejected(fake_weights, tmp_path, monkeypatch):
    spec, _ = fake_weights
    monkeypatch.setitem(sr_models.WEIGHTS, "fake", dataclasses.replace(spec, sha256="0" * 64))
    with pytest.raises(sr_models.WeightsError, match="does not match"):
        sr_models.ensure_weights("fake", tmp_path / "cache", log=lambda _: None)
    assert not list((tmp_path / "cache").iterdir())


def test_missing_weights_without_download_and_unreachable_url_fail_cleanly(fake_weights, tmp_path, monkeypatch):
    with pytest.raises(sr_models.WeightsError, match="downloading is disabled"):
        sr_models.ensure_weights("fake", tmp_path / "cache", download=False)
    spec, _ = fake_weights
    gone = dataclasses.replace(spec, url=(tmp_path / "missing.pth").as_uri())
    monkeypatch.setitem(sr_models.WEIGHTS, "fake", gone)
    with pytest.raises(sr_models.WeightsError, match="could not download"):
        sr_models.ensure_weights("fake", tmp_path / "cache", log=lambda _: None)
    assert not list((tmp_path / "cache").glob("*.part"))


def test_precision_rules():
    assert sr_models.resolve_precision("auto", "cuda", True) == "float16"
    assert sr_models.resolve_precision("auto", "cuda", False) == "float32"
    assert sr_models.resolve_precision("auto", "mps", True) == "float32"
    with pytest.raises(ValueError):
        sr_models.resolve_precision("float16", "cpu", True)
    with pytest.raises(ValueError):
        sr_models.resolve_precision("float16", "cuda", False)


# --------------------------------------------------------------------------- PyTorch wrapper


@pytest.fixture
def tiny_model():
    torch = pytest.importorskip("torch")
    spandrel = pytest.importorskip("spandrel")
    from spandrel.architectures.ESRGAN import ESRGAN

    torch.manual_seed(0)
    network = ESRGAN(in_nc=3, out_nc=3, num_filters=8, num_blocks=1, scale=4)
    descriptor = spandrel.ModelLoader().load_from_state_dict(network.state_dict())
    return sr_models.TorchModel(descriptor, sr_models.WEIGHTS["realesrgan_x4plus"], "cpu", "float32")


def test_wrapper_runs_numpy_tiles_and_pads_tiny_ones(tiny_model):
    tile = np.random.default_rng(0).random((20, 28, 3), dtype=np.float32)
    out = tiny_model(tile)
    assert out.shape == (80, 112, 3) and out.dtype == np.float32
    assert 0 <= out.min() and out.max() <= 1
    assert np.array_equal(out, tiny_model(tile))  # deterministic
    assert tiny_model(tile[:1, :1]).shape == (4, 4, 3)  # below the network's minimum size
    description = tiny_model.describe()
    assert description["weights"]["sha256"] == sr_models.WEIGHTS["realesrgan_x4plus"].sha256
    assert tiny_model.fallback() is None  # already on the CPU


def test_wrapper_turns_out_of_memory_into_a_tiler_signal(tiny_model, monkeypatch):
    def exhausted(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(tiny_model.descriptor.model, "forward", exhausted)
    with pytest.raises(sr_models.DeviceOutOfMemory, match="out of memory"):
        tiny_model(np.zeros((16, 16, 3), np.float32))

    def broken(*args, **kwargs):
        raise RuntimeError("shape mismatch")

    monkeypatch.setattr(tiny_model.descriptor.model, "forward", broken)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        tiny_model(np.zeros((16, 16, 3), np.float32))


def test_auto_tile_size_respects_bounds():
    pytest.importorskip("torch")
    tile, basis = sr_models.auto_tile_size("cpu", "float32", "SwinIR", source_width=3840, scale=4)
    assert sr_models.MIN_AUTO_TILE <= tile <= sr_models.MAX_AUTO_TILE and tile % 32 == 0
    assert "RAM" in basis


# --------------------------------------------------------------------------- real weights


@pytest.mark.parametrize("name", ["swinir_m_x4_classical", "swinir_m_x4_realsr_gan"])
def test_real_network_smoke(name):
    if os.environ.get("SYNTHETIC_RAW_LAB_HEAVY_TESTS") != "1":
        pytest.skip("real-network inference is opt-in: set SYNTHETIC_RAW_LAB_HEAVY_TESTS=1")
    if not sr_models.available():
        pytest.skip("PyTorch and spandrel are not installed (pip install .[sr])")
    if not sr_models.cached(name, WEIGHTS_DIR):
        pytest.skip(f"{sr_models.WEIGHTS[name].filename} is not cached in {WEIGHTS_DIR}")
    model = sr_models.load_model(name, WEIGHTS_DIR, download=False, log=lambda _: None)
    source = cm.srgb_to_linear(make_test_image(384, 320).astype(np.float32) / 255)
    tile = sr_models.MIN_AUTO_TILE
    out, report = sr.super_resolve(source, 922, 768, model, tile_size=tile, tile_overlap=64, log=lambda _: None)
    assert out.shape == (768, 922, 3) and np.isfinite(out).all()
    assert report.tiles == 4  # 2 x 2
    # the Phase 2 seam gates (pipeline.SEAM_*), on the real network
    assert report.to_dict()["overlap_disagreement"]["blend_error_p99_abs_max"] <= pipeline.SEAM_BLEND_ERROR_LIMIT
    assert sr.seam_scores(out, report.boundaries, source)["max_score"] <= pipeline.SEAM_SCORE_LIMIT
    lanczos = sr.resample_box(source, 922, 768, (0.0, 0.0, 384.0, 320.0), Image.Resampling.LANCZOS)
    encoded = [cm.linear_to_srgb(np.clip(x, 0, 1)) for x in (out, lanczos)]
    mse = float(np.mean((encoded[0] - encoded[1]) ** 2))
    assert 10 * np.log10(1 / mse) > 20  # the network stays close to the source structure
