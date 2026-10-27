"""Tiny mocked Colab/GPU coverage; never download or execute a network."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from synthraw import learned_sr
from synthraw.environment import TileParameters
from synthraw.super_resolution import SuperResolutionEngine


@pytest.fixture
def runtime(monkeypatch):
    descriptor = MagicMock()
    descriptor.scale = 4
    descriptor.supports_half = True
    descriptor.eval.return_value.to.return_value = descriptor
    output = np.full((8, 12, 3), 0.4, np.float32)
    prediction = descriptor.return_value.squeeze.return_value.permute.return_value
    prediction.float.return_value.cpu.return_value.numpy.return_value = output
    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    torch.inference_mode.side_effect = nullcontext
    spandrel = SimpleNamespace(ModelLoader=MagicMock(), ImageModelDescriptor=type(descriptor))
    spandrel.ModelLoader.return_value.load_from_state_dict.return_value = descriptor
    monkeypatch.setattr(learned_sr.model_manifest, "_require_colab", lambda: None)
    download = MagicMock(return_value="verified.pth")
    monkeypatch.setattr(learned_sr.model_manifest, "download_checkpoint", download)

    def import_module(name):
        return {"torch": torch, "spandrel": spandrel}[name]

    monkeypatch.setattr(learned_sr.importlib, "import_module", import_module)
    return torch, descriptor, download


@pytest.mark.parametrize("half", [True, False])
def test_gpu_tile_callable(runtime, half):
    torch, descriptor, download = runtime
    descriptor.supports_half = half
    model = learned_sr.load_learned_backend("SWINIR", scale=4, cache_dir="cache", accept_licence=True)
    result = model(np.zeros((2, 3, 3), np.float32))
    assert result.shape == (8, 12, 3)
    download.assert_called_once_with("SWINIR", cache_dir="cache", accept_licence=True)
    torch.load.assert_called_once_with("verified.pth", map_location="cpu", weights_only=True)
    descriptor.eval.return_value.to.assert_called_once_with(
        device="cuda", dtype=torch.float16 if half else torch.float32
    )
    tensor = torch.from_numpy.call_args.args[0]
    assert tensor.shape == (3, 2, 3)
    assert tensor.flags.c_contiguous
    torch.inference_mode.assert_called_once()


def test_no_gpu_does_not_download(runtime):
    torch, _, download = runtime
    torch.cuda.is_available.return_value = False
    with pytest.raises(RuntimeError, match="CUDA GPU"):
        learned_sr.load_learned_backend("SWINIR", scale=4)
    download.assert_not_called()


def test_scale_mismatch_is_explicit(runtime):
    with pytest.raises(ValueError, match="scale x4"):
        learned_sr.load_learned_backend("SWINIR", scale=2)


@pytest.mark.parametrize(
    "error",
    [
        ImportError("no torch"),
        RuntimeError("no GPU"),
        ValueError("UNPINNED"),
        ValueError("SHA-256 mismatch"),
        PermissionError("licence acceptance required"),
    ],
)
def test_loading_failures_are_recorded_and_fall_back(monkeypatch, tmp_path, error):
    loader = MagicMock(side_effect=error)
    monkeypatch.setattr(learned_sr, "load_learned_backend", loader)
    result = SuperResolutionEngine().reconstruct(
        np.full((2, 3, 3), 0.4, np.float32),
        tmp_path / "sr.npy",
        scale=4,
        tiles=TileParameters(4, 1, True, False),
    )
    assert result.backend == "LANCZOS"
    assert [call.args[0] for call in loader.call_args_list] == ["SWINIR", "REAL_ESRGAN"]
    assert len(result.fallback_reasons) == 2
    assert all(str(error) in reason for reason in result.fallback_reasons)


def test_loading_stops_after_success(monkeypatch, tmp_path):
    loader = MagicMock(return_value=lambda patch: np.repeat(np.repeat(patch, 4, axis=0), 4, axis=1))
    monkeypatch.setattr(learned_sr, "load_learned_backend", loader)
    result = SuperResolutionEngine().reconstruct(
        np.zeros((2, 3, 3), np.float32),
        tmp_path / "sr.npy",
        scale=4,
        tiles=TileParameters(4, 1, True, False),
    )
    assert result.backend == "SWINIR"
    loader.assert_called_once()


def test_colab_guard_precedes_imports_and_download(runtime, monkeypatch):
    torch, _, download = runtime

    def deny():
        raise RuntimeError("Checkpoint downloads are permitted only inside Colab")

    monkeypatch.setattr(learned_sr.model_manifest, "_require_colab", deny)
    with pytest.raises(RuntimeError, match="only inside Colab"):
        learned_sr.load_learned_backend("SWINIR", scale=4)
    torch.cuda.is_available.assert_not_called()
    download.assert_not_called()


def test_failed_primary_load_uses_learned_fallback(monkeypatch, tmp_path):
    def model(patch):
        return np.repeat(np.repeat(patch, 4, axis=0), 4, axis=1)

    loader = MagicMock(side_effect=[ValueError("UNPINNED"), model])
    monkeypatch.setattr(learned_sr, "load_learned_backend", loader)
    result = SuperResolutionEngine().reconstruct(
        np.zeros((2, 3, 3), np.float32),
        tmp_path / "sr.npy",
        backend="HAT",
        scale=4,
        tiles=TileParameters(4, 1, True, False),
    )
    assert result.backend == "REAL_ESRGAN"
    assert result.fallback_reasons == ("HAT: ValueError: UNPINNED",)
