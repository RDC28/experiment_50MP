"""Offline artifact-integrity tests for Notebook 01 inference."""

import hashlib
import io

import pytest

from synthetic_raw_lab import human_models as models


@pytest.fixture
def fake_model(monkeypatch):
    data = b"local fake ONNX artifact"
    monkeypatch.setattr(models, "YUNET_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(models, "YUNET_SIZE", len(data))
    return data


def test_verified_download_and_cache_hit(tmp_path, monkeypatch, fake_model):
    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(fake_model)

    monkeypatch.setattr(models, "urlopen", download)
    path = models.ensure_yunet_model(tmp_path)
    assert path.read_bytes() == fake_model
    assert models.ensure_yunet_model(tmp_path) == path
    assert calls == [(models.YUNET_URL, 30)]
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("cached", [False, True])
def test_checksum_mismatch_never_used(tmp_path, monkeypatch, fake_model, cached):
    path = tmp_path / models.YUNET_FILENAME
    if cached:
        path.write_bytes(b"corrupt")
    monkeypatch.setattr(models, "urlopen", lambda *a, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(RuntimeError, match="checksum/size mismatch"):
        models.ensure_yunet_model(tmp_path)
    assert path.exists() == cached
    assert not list(tmp_path.glob("*.tmp"))


def test_download_unavailable(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("offline test")

    monkeypatch.setattr(models, "urlopen", offline)
    with pytest.raises(RuntimeError, match="runtime weights unavailable: offline test"):
        models.ensure_yunet_model(tmp_path)
    assert not list(tmp_path.iterdir())


def test_metadata_pins_artifact():
    metadata = models.yunet_model_metadata()
    assert metadata["revision"] in metadata["source_url"]
    assert metadata["sha256"] == models.YUNET_SHA256
    assert metadata["size_bytes"] == models.YUNET_SIZE
    assert metadata["license"] == "MIT"
