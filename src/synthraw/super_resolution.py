"""Disk-backed tiled reconstruction for PRD_v3's reconstruction milestone."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synthraw.environment import TileParameters


@dataclass(frozen=True)
class TiledSRResult:
    """Reconstruction location and first-tile timing, without loading the output."""

    path: Path
    tile_count: int
    benchmark_seconds: float
    estimated_seconds: float


class SuperResolutionEngine:
    """Execute optional RGB model callables on bounded, overlapping tiles."""

    def reconstruct_tiles(
        self,
        source: np.ndarray,
        output_path: str | Path,
        model: Callable[[np.ndarray], np.ndarray],
        *,
        scale: int,
        tiles: TileParameters,
    ) -> TiledSRResult:
        """Blend reflected halo tiles into float32 NPY storage (PRD compute budget).

        The callable receives HWC float32 RGB and returns exactly scale times
        those dimensions. Only one tile and one normalization strip live in RAM.
        Output and scratch paths must be new; failures remove partial products.
        """
        if source.ndim != 3 or source.shape[2] != 3 or min(source.shape[:2]) < 1:
            raise ValueError("source must be nonempty HWC RGB")
        if type(scale) is not int or scale < 1:
            raise ValueError("scale must be a positive integer")
        tile, overlap = tiles.tile_size, tiles.overlap
        if type(tile) is not int or type(overlap) is not int or tile < 1 or not 0 <= overlap < tile:
            raise ValueError("tile size must exceed nonnegative overlap")
        path = Path(output_path)
        weights_path = path.with_name(path.name + ".weights.npy")
        if path.exists() or weights_path.exists():
            raise FileExistsError("output or scratch file already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = source.shape[:2]
        stride = tile - overlap
        positions = [(y, x) for y in range(0, height, stride) for x in range(0, width, stride)]
        output = weights = None
        benchmark = 0.0
        try:
            output = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float32, shape=(height * scale, width * scale, 3)
            )
            weights = np.lib.format.open_memmap(
                weights_path, mode="w+", dtype=np.float32, shape=(height * scale, width * scale)
            )
            output[:] = 0
            weights[:] = 0
            halo = overlap
            for index, (y, x) in enumerate(positions):
                h, w = min(tile, height - y), min(tile, width - x)
                y0, x0 = max(0, y - halo), max(0, x - halo)
                y1, x1 = min(height, y + h + halo), min(width, x + w + halo)
                patch = np.asarray(source[y0:y1, x0:x1], dtype=np.float32)
                if not np.isfinite(patch).all() or np.any((patch < 0) | (patch > 1)):
                    raise ValueError("source must contain finite RGB values in [0, 1]")
                padding = (
                    (max(0, halo - y), max(0, y + h + halo - height)),
                    (max(0, halo - x), max(0, x + w + halo - width)),
                    (0, 0),
                )
                patch = np.pad(patch, padding, mode="reflect")
                started = time.perf_counter()
                prediction = np.asarray(model(patch), dtype=np.float32)
                if index == 0:
                    benchmark = time.perf_counter() - started
                if prediction.shape != (patch.shape[0] * scale, patch.shape[1] * scale, 3):
                    raise ValueError("model returned an incompatible shape")
                if not np.isfinite(prediction).all():
                    raise ValueError("model returned nonfinite values")
                core = prediction[halo * scale : (halo + h) * scale, halo * scale : (halo + w) * scale]
                # Positive triangular windows retain coverage at outer image edges.
                wy = 1 - np.abs((np.arange(h * scale) + 0.5) * 2 / (h * scale) - 1)
                wx = 1 - np.abs((np.arange(w * scale) + 0.5) * 2 / (w * scale) - 1)
                weight = (wy[:, None] * wx[None, :]).astype(np.float32)
                region = np.s_[y * scale : (y + h) * scale, x * scale : (x + w) * scale]
                output[region] += core * weight[..., None]
                weights[region] += weight
            for y in range(0, height * scale, tile):
                output[y : y + tile] /= weights[y : y + tile, :, None]
            output.flush()
            weights.flush()
        except BaseException:
            output = weights = None
            path.unlink(missing_ok=True)
            raise
        finally:
            output = weights = None
            weights_path.unlink(missing_ok=True)
        return TiledSRResult(path, len(positions), benchmark, benchmark * len(positions))
