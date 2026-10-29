"""Opt-in fidelity API: verified user checkpoints and bounded tiled SR.

No downloads or implicit fallback. Real loading/inference is Colab CUDA-only.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from synthraw.environment import TileParameters
from synthraw.super_resolution import SuperResolutionEngine


@dataclass(frozen=True)
class SRCheckpoint:
    path: str
    sha256: str
    backend: str
    repository_revision: str
    source: str
    licence: str
    licence_accepted: bool = False
    scale: int = 4

    def verify(self) -> dict:
        """Verify bytes without loading tensors; provenance is caller supplied."""
        if self.backend not in {"HAT", "REAL_ESRGAN"}:
            raise ValueError("Supported SR backends are HAT and REAL_ESRGAN")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Expected lowercase SHA-256")
        if not re.fullmatch(r"[0-9a-f]{40}", self.repository_revision):
            raise ValueError("Expected full repository revision")
        if not self.source.strip() or not self.licence.strip() or self.licence_accepted is not True:
            raise ValueError("Checkpoint source, licence and explicit acceptance required")
        if type(self.scale) is not int or self.scale != 4:
            raise ValueError("Only x4 checkpoints are supported")
        digest = hashlib.sha256()
        with Path(self.path).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != self.sha256:
            raise ValueError("Checkpoint SHA-256 mismatch")
        return {
            **asdict(self),
            "integrity": "verified",
            "provenance": "user supplied; rights not independently verified",
        }


def load_checkpoint(spec: SRCheckpoint):
    """Load safe tensor state and validate architecture, RGB channels and scale."""
    from synthraw.model_manifest import _require_colab

    _require_colab()
    evidence = spec.verify()
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("Fidelity SR requires CUDA")
    spandrel = importlib.import_module("spandrel")
    state = torch.load(spec.path, map_location="cpu", weights_only=True)
    descriptor = spandrel.ModelLoader().load_from_state_dict(state)
    expected = {"HAT": "HAT", "REAL_ESRGAN": "RRDBNet"}[spec.backend]
    if (
        not isinstance(descriptor, spandrel.ImageModelDescriptor)
        or descriptor.architecture.id != expected
        or descriptor.scale != spec.scale
        or descriptor.input_channels != 3
        or descriptor.output_channels != 3
    ):
        raise ValueError("Checkpoint architecture, scale or RGB contract mismatch")
    dtype = torch.float16 if descriptor.supports_half else torch.float32
    descriptor = descriptor.eval().to(device="cuda", dtype=dtype)

    def infer(patch):
        tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1))).unsqueeze(0)
        with torch.inference_mode():
            prediction = descriptor(tensor.to(device="cuda", dtype=dtype))
            return prediction.squeeze(0).permute(1, 2, 0).float().cpu().numpy()

    return infer, {**evidence, "dtype": str(dtype), "architecture": expected}


def reconstruct(source, output_path, spec: SRCheckpoint, *, tile=512, overlap=64) -> dict:
    """Produce encoded-sRGB x4 NPY memmap for optional pipeline integration.

    Reuse the tested halo/blend engine and its visible one-time OOM tile retry.
    Load failures propagate before allocation. No substitute backend is selected.
    """
    infer, evidence = load_checkpoint(spec)
    result = SuperResolutionEngine().reconstruct_tiles(
        source,
        output_path,
        infer,
        scale=spec.scale,
        tiles=TileParameters(tile, overlap, False, True),
    )
    return {
        "requested": spec.backend,
        "effective": spec.backend,
        "checkpoint": evidence,
        "output": str(result.path),
        "tile_count": result.tile_count,
        "effective_tile": result.effective_tile_size,
        "retry_reasons": list(result.retry_reasons),
        "fallback": None,
        "validation": "Execution only; scientific fidelity and full-scale Colab validation pending",
    }
