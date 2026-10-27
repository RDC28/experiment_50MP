"""Experimental licence-gated stubs: warn, never download, fall back deterministically."""

import json
import warnings
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from synthraw import model_manifest
from synthraw.camera_colour import CameraColourModel, InverseISPEngine
from synthraw.config import ExperimentConfig
from synthraw.environment import TileParameters
from synthraw.experimental import (
    CAMERA_DOMAIN,
    EXPERIMENTAL_BACKENDS,
    request_experimental,
    spectral_inference_records,
)
from synthraw.hdr import HDRReconstructionEngine, inverse_transfer
from synthraw.pipeline import run_pipeline
from synthraw.super_resolution import SuperResolutionEngine

STUBS = ("SUPIR", "EXPANDDIFF", "VID2HDR", "RAW_FLOW", "SPIRALDIFF", "INFER_SPECTRAL_RESPONSE_FROM_AI_IMAGE")


@pytest.fixture
def network(monkeypatch):
    """Fail loudly if any stub reaches the downloader or the network."""
    download = Mock(side_effect=AssertionError("experimental stubs must never download"))
    monkeypatch.setattr(model_manifest, "download_checkpoint", download)
    monkeypatch.setattr(model_manifest, "urlopen", download)
    return download


def test_every_prd_experimental_option_is_registered_and_unpinned():
    assert set(EXPERIMENTAL_BACKENDS) == set(STUBS)
    for backend in EXPERIMENTAL_BACKENDS.values():
        spec = model_manifest.MODEL_MANIFEST[backend.manifest_name]
        assert spec.status == model_manifest.UNPINNED and spec.licence_gate


@pytest.mark.parametrize("name", STUBS)
@pytest.mark.parametrize("accept", [False, True])
def test_stub_warns_licence_and_records_fallback(name, accept, network):
    with pytest.warns(UserWarning, match=f"{name} licence restrictions"):
        decision = request_experimental(name, accept_licence=accept)
    assert decision.backend == EXPERIMENTAL_BACKENDS[name].fallback != name
    assert decision.status == model_manifest.UNPINNED and decision.licence_accepted is accept
    assert name in decision.fallback_reason and decision.backend in decision.fallback_reason
    assert "not a Sony alpha 1 II" in decision.camera_domain
    network.assert_not_called()


def test_supir_falls_back_to_fidelity_chain_without_running_injected_model(tmp_path, network):
    supir = Mock()
    source = np.linspace(0, 1, 4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
    with pytest.warns(UserWarning, match="SUPIR licence"):
        result = SuperResolutionEngine().reconstruct(
            source,
            tmp_path / "sr.npy",
            backend="SUPIR",
            models={"SUPIR": supir, "SWINIR": lambda patch: np.repeat(np.repeat(patch, 2, 0), 2, 1)},
            scale=2,
            tiles=TileParameters(64, 8, False, True),
        )
    supir.assert_not_called()
    assert result.backend == "SWINIR"
    assert "SUPIR" in result.fallback_reasons[0] and "SWINIR fallback" in result.fallback_reasons[0]
    network.assert_not_called()


@pytest.mark.parametrize("backend", ["EXPANDDIFF", "VID2HDR"])
def test_generative_hdr_matches_deterministic_output(tmp_path, backend, network):
    encoded = np.linspace(0, 1, 27, dtype=np.float32).reshape(3, 3, 3)
    linear = inverse_transfer(encoded)
    engine = HDRReconstructionEngine()
    with pytest.warns(UserWarning, match=f"{backend} licence"):
        stub = engine.expand(encoded, linear, tmp_path / "stub", backend=backend)
    baseline = engine.expand(encoded, linear, tmp_path / "baseline")
    assert stub["backend"] == "DETERMINISTIC" and "DETERMINISTIC fallback" in stub["fallback_reason"]
    for name, array in stub["arrays"].items():
        np.testing.assert_array_equal(array, baseline["arrays"][name])
    network.assert_not_called()


def test_learned_inverse_isp_records_each_requested_prior(network):
    with pytest.warns(UserWarning):
        result = InverseISPEngine().reconstruct(
            np.full((2, 2, 3), 0.25), CameraColourModel(), use_raw_flow=True, use_spiraldiff=True
        )
    reason = result.records["inverse_isp_fallback_reason"].value
    assert "RAW_FLOW" in reason and "SPIRALDIFF" in reason and "PHYSICS fallback" in reason
    assert result.records["inverse_isp_backend"].value == "PHYSICS"
    assert result.records["inverse_isp_camera_domain"].value == CAMERA_DOMAIN


def test_spectral_inference_is_low_confidence_and_never_replaces_matrix():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        disabled = {record.field: record.value for record in spectral_inference_records(False)}
    assert disabled == {
        "spectral_inference_requested": False,
        "spectral_inference_backend": None,
        "spectral_inference_fallback_reason": None,
    }
    with pytest.warns(UserWarning, match="INFER_SPECTRAL_RESPONSE_FROM_AI_IMAGE licence"):
        records = spectral_inference_records(True)
    assert records[1].value == "MATRIX_ONLY" and "LOW CONFIDENCE" in records[0].method
    assert all(record.confidence is None for record in records)
    with pytest.raises(ValueError, match="boolean"):
        spectral_inference_records(1)


def test_tiny_pipeline_with_every_stub_enabled_records_fallbacks(tmp_path, network):
    pixels = (np.linspace(0, 255, 32 * 12 * 3).reshape(12, 32, 3)).astype(np.uint8)
    source = tmp_path / "source.png"
    Image.fromarray(pixels).save(source)
    config = ExperimentConfig(
        FAST_DEV_RUN=True,
        SR_BACKEND="SUPIR",
        HDR_BACKEND="EXPANDDIFF",
        USE_RAW_FLOW=True,
        USE_SPIRALDIFF=True,
        INFER_SPECTRAL_RESPONSE_FROM_AI_IMAGE=True,
    )
    with pytest.warns(UserWarning, match="licence restrictions"):
        result = run_pipeline(
            source, config, tmp_path / "out", timestamp=datetime(2026, 1, 2), fast_dev_max_side=64, strip_rows=16
        )
    fallbacks = " | ".join(result.fallbacks)
    for name in STUBS:
        if name != "VID2HDR":
            assert name in fallbacks
    metadata = (result.root / "metadata").glob("*.json")
    text = "\n".join(path.read_text(encoding="utf-8") for path in metadata)
    assert "Sony alpha 1 II model" not in text.replace("not a Sony alpha 1 II model", "")
    assert json.loads((result.root / "metadata/config.json").read_text())["INFER_SPECTRAL_RESPONSE_FROM_AI_IMAGE"]
    network.assert_not_called()
