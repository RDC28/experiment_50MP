"""Colab-only learned fidelity backends for PRD_v3 model selection."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

import numpy as np

from synthraw import model_manifest


def load_learned_backend(
    name: str,
    *,
    scale: int,
    cache_dir: str | Path = "/content/synthraw_cache",
    accept_licence: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a pinned GPU tile callable lazily, with safe tensor-only loading.

    Both runtime and download guards apply before loading any weights. Unsupported
    checkpoint scales fail explicitly so the selector can record the fallback.
    """
    if name not in {"SWINIR", "HAT", "REAL_ESRGAN"}:
        raise ValueError(f"Unsupported fidelity backend: {name}")
    model_manifest._require_colab()
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("Learned SR requires a CUDA GPU")
    spandrel = importlib.import_module("spandrel")
    checkpoint = model_manifest.download_checkpoint(name, cache_dir=cache_dir, accept_licence=accept_licence)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    descriptor = spandrel.ModelLoader().load_from_state_dict(state)
    if not isinstance(descriptor, spandrel.ImageModelDescriptor):
        raise ValueError("Checkpoint is not an image-to-image model")
    if descriptor.scale != scale:
        raise ValueError(f"Checkpoint scale x{descriptor.scale} does not match requested x{scale}")
    half = bool(descriptor.supports_half)
    descriptor = descriptor.eval().to(device="cuda", dtype=torch.float16 if half else torch.float32)

    def infer(patch: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).unsqueeze(0)
        tensor = tensor.to(device="cuda", dtype=torch.float16 if half else torch.float32)
        with torch.inference_mode():
            prediction = descriptor(tensor)
            result = prediction.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        return np.clip(result, 0, 1)

    return infer
