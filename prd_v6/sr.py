"""HAT-L x4 with feathered tiles and the x8 dihedral self-ensemble (sections 8.1, 9.3).

`upscale` works with any callable that maps an (h, w, 3) float32 tile in [0, 1] to (4h, 4w, 3), so
the tiling and ensemble are tested locally with a fake model; `load_hat_l` builds the real one on
the GPU. There is no fallback: if HAT-L cannot be loaded, the max preset fails (Q7).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from prd_v6 import constants

Model = Callable[[np.ndarray], np.ndarray]


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Tile origins with stride tile - overlap; the last tile is shifted to end at the edge."""
    if tile >= length:
        return [0]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than the tile")
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def _ramp(length: int, width: int, at_start: bool, at_end: bool) -> np.ndarray:
    """1-D feather: linear ramps of `width` on interior edges, flat on image edges, never zero."""
    weight = np.ones(length, dtype=np.float32)
    width = min(width, length // 2)
    if width > 0:
        ramp = (np.arange(width, dtype=np.float32) + 0.5) / width
        if not at_start:
            weight[:width] = ramp
        if not at_end:
            weight[-width:] = np.minimum(weight[-width:], ramp[::-1])
    return np.maximum(weight, 1e-4)


def _dihedral(image: np.ndarray, k: int) -> np.ndarray:
    out = np.rot90(image, k % 4, axes=(0, 1))
    return out[:, ::-1] if k >= 4 else out


def _undo_dihedral(image: np.ndarray, k: int) -> np.ndarray:
    out = image[:, ::-1] if k >= 4 else image
    return np.rot90(out, -(k % 4), axes=(0, 1))


@dataclass
class Upscaled:
    """Encoded x4 results: the x8 ensemble mean and its identity member (kept for the ablation)."""

    ensemble: np.ndarray
    identity: np.ndarray
    tiles: int
    passes: int


def upscale(
    encoded: np.ndarray,
    model: Model,
    *,
    scale: int = constants.SR_SCALE,
    tile: int = constants.SR_TILE,
    overlap: int = constants.SR_OVERLAP,
    ensemble: bool = True,
    progress: Callable[[str], None] | None = None,
) -> Upscaled:
    encoded = np.ascontiguousarray(encoded, dtype=np.float32)
    height, width = encoded.shape[:2]
    rows = tile_starts(height, tile, overlap)
    cols = tile_starts(width, tile, overlap)
    out_shape = (height * scale, width * scale, 3)
    acc = np.zeros(out_shape, dtype=np.float32)
    acc_identity = np.zeros(out_shape, dtype=np.float32)
    weights = np.zeros(out_shape[:2], dtype=np.float32)
    members = range(8) if ensemble else range(1)
    total = len(rows) * len(cols)
    done = 0
    for top in rows:
        th = min(tile, height - top)
        wy = _ramp(th * scale, overlap * scale, top == 0, top + th >= height)
        for left in cols:
            tw = min(tile, width - left)
            wx = _ramp(tw * scale, overlap * scale, left == 0, left + tw >= width)
            patch = encoded[top : top + th, left : left + tw]
            mean = np.zeros((th * scale, tw * scale, 3), dtype=np.float32)
            for k in members:
                result = np.asarray(model(np.ascontiguousarray(_dihedral(patch, k))), dtype=np.float32)
                result = _undo_dihedral(result, k)
                if result.shape != mean.shape:
                    raise ValueError(f"model returned {result.shape}, expected {mean.shape}")
                if k == 0:
                    identity = result
                mean += result
            mean /= len(members)
            w = wy[:, None] * wx[None, :]
            ys, xs = slice(top * scale, (top + th) * scale), slice(left * scale, (left + tw) * scale)
            acc[ys, xs] += mean * w[..., None]
            acc_identity[ys, xs] += identity * w[..., None]
            weights[ys, xs] += w
            done += 1
            if progress and (done == total or done % 10 == 0):
                progress(f"SR tiles {done}/{total}")
    acc /= weights[..., None]
    acc_identity /= weights[..., None]
    return Upscaled(acc, acc_identity, total, total * len(members))


def load_hat_l(checkpoint: str | Path, device: str = "cuda") -> tuple[Model, dict]:
    """Load the pinned HAT-L x4 with spandrel, fp32, TF32 off; check it is really HAT-L x4."""
    import torch
    from spandrel import ModelLoader

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    for key in ("params_ema", "params"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    descriptor = ModelLoader().load_from_state_dict(state)
    parameters = sum(p.numel() for p in descriptor.model.parameters())
    info = {
        "architecture": str(descriptor.architecture.id),
        "scale": int(descriptor.scale),
        "input_channels": int(descriptor.input_channels),
        "output_channels": int(descriptor.output_channels),
        "parameters": int(parameters),
        "precision": "fp32",
        "tf32": False,
    }
    if (
        info["architecture"] != "HAT"
        or info["scale"] != constants.SR_SCALE
        or info["input_channels"] != 3
        or info["output_channels"] != 3
        or parameters < constants.HAT_L_MIN_PARAMETERS
    ):
        raise RuntimeError(f"checkpoint is not HAT-L x4: {info}")
    module = descriptor.model.to(device).float().eval()

    def model(tile: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(tile.transpose(2, 0, 1)))[None].to(device)
            out = module(tensor)[0].float().clamp_(0, 1)
            return out.permute(1, 2, 0).cpu().numpy()

    return model, info


def lanczos_model(scale: int = constants.SR_SCALE) -> Model:
    """A stand-in model for local tests and the ablation's Lanczos row."""
    import cv2

    def model(tile: np.ndarray) -> np.ndarray:
        h, w = tile.shape[:2]
        return np.clip(cv2.resize(tile, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4), 0, 1)

    return model
