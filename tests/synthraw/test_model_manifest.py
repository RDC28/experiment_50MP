"""Checkpoint policy tests use byte fixtures, never real weights or network."""

import hashlib
import io
import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from synthraw import model_manifest as manifest


def test_registry_and_log(tmp_path):
    entries = manifest.MODEL_MANIFEST
    assert set(entries) == {
        "SWINIR",
        "HAT",
        "REAL_ESRGAN",
        "SUPIR",
        "EXPANDDIFF",
        "VID2HDR",
        "RAW_FLOW",
        "SPIRALDIFF",
        "GEOCALIB",
    }
    assert entries["SWINIR"].commit == "6545850"
    assert entries["REAL_ESRGAN"].sha256 == "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"
    payload = json.loads(manifest.write_model_manifest(tmp_path).read_text())
    assert len(payload["models"]) == 9
    assert payload["models"][0]["status"] == "PINNED"
    for name, spec in entries.items():
        if name not in ("SWINIR", "REAL_ESRGAN"):
            assert spec.status == manifest.UNPINNED
            assert spec.commit is None and spec.sha256 is None


@pytest.fixture
def download(monkeypatch):
    content = b"tiny mock checkpoint"
    spec = replace(manifest.MODEL_MANIFEST["SWINIR"], sha256=hashlib.sha256(content).hexdigest())
    monkeypatch.setitem(manifest.MODEL_MANIFEST, "SWINIR", spec)
    network = Mock(side_effect=lambda *args, **kwargs: io.BytesIO(content))
    monkeypatch.setattr(manifest, "urlopen", network)
    monkeypatch.setattr(manifest, "_require_colab", lambda: None)
    return network


def test_download_cache_and_revalidation(tmp_path, download):
    target = manifest.download_checkpoint("SWINIR", cache_dir=tmp_path)
    assert target.read_bytes() == b"tiny mock checkpoint"
    assert manifest.download_checkpoint("SWINIR", cache_dir=tmp_path) == target
    assert download.call_count == 1
    assert (tmp_path / ".gitignore").read_text() == "*\n"
    target.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="cached checkpoint"):
        manifest.download_checkpoint("SWINIR", cache_dir=tmp_path)
    assert download.call_count == 1


@pytest.mark.parametrize("failure", ["digest", "network"])
def test_failed_download_leaves_no_checkpoint(tmp_path, download, failure):
    if failure == "digest":
        download.side_effect = lambda *args, **kwargs: io.BytesIO(b"wrong")
    else:
        download.side_effect = OSError("offline")
    with pytest.raises((ValueError, OSError)):
        manifest.download_checkpoint("SWINIR", cache_dir=tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {".gitignore"}


def test_disabled_models_never_download(tmp_path, download):
    for name in ("HAT", "SUPIR", "EXPANDDIFF", "VID2HDR", "RAW_FLOW", "SPIRALDIFF", "GEOCALIB"):
        with pytest.raises(ValueError, match="UNPINNED"):
            manifest.download_checkpoint(name, cache_dir=tmp_path, accept_licence=True)
    download.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_gate_precedes_network(tmp_path, download, monkeypatch):
    monkeypatch.setitem(
        manifest.MODEL_MANIFEST, "SWINIR", replace(manifest.MODEL_MANIFEST["SWINIR"], licence_gate=True)
    )
    with pytest.raises(PermissionError, match="acceptance"):
        manifest.download_checkpoint("SWINIR", cache_dir=tmp_path)
    download.assert_not_called()
    with pytest.warns(UserWarning, match="licence restrictions"):
        manifest.download_checkpoint("SWINIR", cache_dir=tmp_path, accept_licence=True)
    assert download.call_count == 1


def test_mac_runtime_block_precedes_filesystem(tmp_path, monkeypatch):
    def unavailable(name):
        raise ImportError(name)

    monkeypatch.setattr(manifest.importlib, "import_module", unavailable)
    network = Mock()
    monkeypatch.setattr(manifest, "urlopen", network)
    with pytest.raises(RuntimeError, match="only inside Colab"):
        manifest.download_checkpoint("SWINIR", cache_dir=tmp_path)
    network.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("field,value", [("commit", None), ("commit", "main"), ("sha256", "invented")])
def test_incomplete_pins_disabled(field, value):
    assert replace(manifest.MODEL_MANIFEST["SWINIR"], **{field: value}).status == manifest.UNPINNED
