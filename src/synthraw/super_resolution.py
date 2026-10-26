"""Disk-backed tiled reconstruction for PRD_v3's reconstruction milestone."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
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
    effective_tile_size: int
    retry_reasons: tuple[str, ...] = ()


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
        """Retry OOM once with half-size tiles (PRD failure hierarchy).

        A second OOM propagates to the backend selector. Other model failures
        propagate immediately. No torch import is needed for CUDA OOM detection.
        """
        try:
            return self._reconstruct_tiles_once(source, output_path, model, scale=scale, tiles=tiles)
        except (MemoryError, RuntimeError) as error:
            if not isinstance(error, MemoryError) and "out of memory" not in str(error).lower():
                raise
            reduced_size = max(1, tiles.tile_size // 2)
            reduced = replace(tiles, tile_size=reduced_size, overlap=min(tiles.overlap, reduced_size - 1))
            reason = f"{type(error).__name__}: {error}; tile {tiles.tile_size} -> {reduced_size}"
        result = self._reconstruct_tiles_once(source, output_path, model, scale=scale, tiles=reduced)
        return replace(result, retry_reasons=(reason,))

    def _reconstruct_tiles_once(
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
        return TiledSRResult(path, len(positions), benchmark, benchmark * len(positions), tile)


@dataclass(frozen=True)
class SeamDiagnostic:
    """Boundary/interior residuals against an aligned reference (PRD validation)."""

    residual_map_path: Path
    boundary_mean: float
    interior_mean: float
    ratio: float
    meets_target: bool


def diagnose_tile_seams(
    reconstruction: np.ndarray,
    reference: np.ndarray,
    output_path: str | Path,
    *,
    tile_size: int,
    overlap: int,
    scale: int,
    boundary_width: int = 1,
    strip_rows: int = 64,
) -> SeamDiagnostic:
    """Store mean absolute RGB residuals and compare tile starts to interiors.

    Both inputs must share the reconstructed grid and radiometric convention.
    The reference should be an independently reconstructed baseline, so image
    edges are not automatically mistaken for seams. Empty regions are rejected.
    A zero interior residual yields ratio zero for perfect output, infinity for
    nonzero boundary error. The PRD target is strictly below 1.25.
    """
    if reconstruction.ndim != 3 or reconstruction.shape[2] != 3 or reconstruction.shape != reference.shape:
        raise ValueError("reconstruction and reference must share a nonempty HWC RGB shape")
    if min(reconstruction.shape[:2]) < 1:
        raise ValueError("inputs must be nonempty")
    values = (tile_size, overlap, scale, boundary_width, strip_rows)
    if any(type(value) is not int for value in values):
        raise ValueError("tile and strip parameters must be integers")
    if tile_size < 1 or not 0 <= overlap < tile_size or min(scale, boundary_width, strip_rows) < 1:
        raise ValueError("invalid tile or strip parameters")
    height, width = reconstruction.shape[:2]
    stride = (tile_size - overlap) * scale

    def near_boundary(indices: np.ndarray, extent: int) -> np.ndarray:
        # Tile starts strictly inside the image, excluding the outer border.
        previous = (indices // stride) * stride
        following = previous + stride
        return ((previous > 0) & (indices - previous < boundary_width)) | (
            (following < extent) & (following - indices < boundary_width)
        )

    columns = np.arange(width)
    column_boundary = near_boundary(columns, width)
    rows = np.arange(height)
    row_boundary = near_boundary(rows, height)
    boundary_count = int(row_boundary.sum()) * width + int(column_boundary.sum()) * int((~row_boundary).sum())
    if boundary_count == 0 or boundary_count == height * width:
        raise ValueError("diagnostic requires both tile boundaries and interior pixels")
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = None
    boundary_sum = interior_sum = 0.0
    try:
        output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(height, width))
        for y in range(0, height, strip_rows):
            a = np.asarray(reconstruction[y : y + strip_rows], dtype=np.float32)
            b = np.asarray(reference[y : y + strip_rows], dtype=np.float32)
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("inputs must be finite")
            residual = np.mean(np.abs(a - b), axis=2)
            if not np.isfinite(residual).all():
                raise ValueError("residual overflow")
            boundary = row_boundary[y : y + strip_rows, None] | column_boundary[None, :]
            output[y : y + strip_rows] = residual
            boundary_sum += float(residual[boundary].sum(dtype=np.float64))
            interior_sum += float(residual[~boundary].sum(dtype=np.float64))
        output.flush()
    except BaseException:
        output = None
        path.unlink(missing_ok=True)
        raise
    finally:
        output = None
    boundary_mean = boundary_sum / boundary_count
    interior_mean = interior_sum / (height * width - boundary_count)
    ratio = boundary_mean / interior_mean if interior_mean else (float("inf") if boundary_mean else 0.0)
    return SeamDiagnostic(path, boundary_mean, interior_mean, ratio, ratio < 1.25)


@dataclass(frozen=True)
class SourceFidelityDiagnostic:
    """Source-grid RGB errors in the shared input convention (PRD SR validation)."""

    mean_absolute_error: float
    root_mean_square_error: float
    maximum_absolute_error: float
    psnr_db: float


def diagnose_source_fidelity(reconstruction: np.ndarray, source: np.ndarray) -> SourceFidelityDiagnostic:
    """Compare exact area downsampling to source RGB, including fractional scales.

    Inputs must use the same transfer function and primaries. Read one SR row
    at a time so disk-backed full-resolution RGB never becomes a RAM copy.
    Metrics use a unit peak and no clipping; perfect fidelity has infinite PSNR.
    This measures source consistency, not whether invented detail is correct.
    """
    for array in (reconstruction, source):
        if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
            raise ValueError("inputs must be nonempty HWC RGB")
    height, width = source.shape[:2]
    sr_height, sr_width = reconstruction.shape[:2]
    if sr_height < height or sr_width < width:
        raise ValueError("reconstruction must be at least source size on both axes")
    x_edges = np.linspace(0, sr_width, width + 1)
    x_index = np.minimum(x_edges.astype(np.int64), sr_width - 1)
    x_fraction = x_edges - x_index
    x_scale, y_scale = sr_width / width, sr_height / height
    absolute_sum = squared_sum = maximum = 0.0
    for y in range(height):
        top, bottom = y * y_scale, min(sr_height, (y + 1) * y_scale)
        downsampled = np.zeros((width, 3), dtype=np.float64)
        for row in range(int(np.floor(top)), int(np.ceil(bottom))):
            pixels = np.asarray(reconstruction[row], dtype=np.float64)
            if not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 1)):
                raise ValueError("inputs must contain finite RGB values in [0, 1]")
            integral = np.vstack((np.zeros((1, 3)), np.cumsum(pixels, axis=0)))
            edges = integral[x_index] + x_fraction[:, None] * pixels[x_index]
            weight = min(bottom, row + 1) - max(top, row)
            downsampled += np.diff(edges, axis=0) * (weight / (x_scale * y_scale))
        original = np.asarray(source[y], dtype=np.float64)
        if not np.isfinite(original).all() or np.any((original < 0) | (original > 1)):
            raise ValueError("inputs must contain finite RGB values in [0, 1]")
        error = downsampled - original
        absolute_sum += float(np.abs(error).sum())
        squared_sum += float(np.square(error).sum())
        maximum = max(maximum, float(np.abs(error).max()))
    count = height * width * 3
    mse = squared_sum / count
    return SourceFidelityDiagnostic(
        absolute_sum / count,
        float(np.sqrt(mse)),
        maximum,
        float(-10 * np.log10(mse)) if mse else float("inf"),
    )
