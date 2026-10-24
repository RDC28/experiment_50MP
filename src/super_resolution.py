"""Tiled pretrained super-resolution to the exact target size (PRD sections 10.2 to 10.5, 21.4).

Scale handling (PRD 10.3). The network has one native scale, x4. The source is
super-resolved at x4 and then resampled with Lanczos-3 in linear light to the
exact target dimensions of ``resolution.compute_target_resolution``. For a 4K
source (target scale about 2.45) that last step is an antialiased downscale, so
no interpolation is stacked on top of the inferred detail; for sources that
need more than x4 the remaining factor is classical interpolation and the
manifest says so. The network never runs at a scale it was not trained for.

Tiling (PRD 10.4). The source is cut into overlapping tiles of ``tile_size``
pixels. One row of tiles (a strip) is super-resolved at a time and the tiles
are blended along x at model resolution; the strip is then resampled to its
target rows and blended into the output along y. Only one strip at model
resolution is ever resident, never the full x4 image. Blending weights are a
partition of unity: inside each overlap the outer quarter next to each tile
edge, where zero padding and missing context disturb the network, gets no
weight, and the middle half is a raised-cosine crossfade. Weights depend only
on the tile geometry, so the result is deterministic and a flat image stays
flat. Tile starts follow the network's attention-window grid. Where two tiles
overlap, their disagreement and the resulting blend error are measured (soft
seams), and the finished image gets a hard-seam height at every tile boundary.

Memory fallback (PRD 10.4, 21.4, 22.3). When the device runs out of memory on
a tile, that tile is split into smaller overlapping sub-tiles and later tiles
use the smaller size; below the minimum tile the model moves to the CPU. Every
event is recorded.

Anti-ringing (PRD 25.2: no halos, no clipping caused by the pipeline). The
network output is clamped to [0, 1], yet Lanczos-3 rings on the crisp edges
and fine texture a perceptual network draws: the first Colab balanced run went
down to -0.17 linear, which the integer TIFF clips to black (dark halos around
bright detail). Each resampled sample is therefore held within the range of the
network-output pixels under the main lobe of the kernel (``resample_antiring``),
the anti-ringing clamp video scalers use. Ringing goes, edges stay as sharp as
Lanczos makes them, and the range before and after is recorded.

Hallucination controls (PRD 10.5). ``detail_strength`` blends the network
output with the Lanczos interpolation of the source, and ``project_to_source``
corrects low-frequency drift so that the area-reduced output agrees with the
source. ``interpolation_difference`` maps how far the result departs from
interpolation, which is where inferred detail lives.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from . import color_management as cm
from .quality_metrics import luma
from .sr_models import MIN_TILE, DeviceOutOfMemory

MODES = ("conservative", "balanced", "aggressive")
MIN_OVERLAP = 16
# Offset of the gain-form low-frequency correction, in linear light (about sRGB code 8 of 255 is 0.0024):
# below it the correction acts additively, above it in proportion to the signal.
PROJECTION_OFFSET = 1e-3


@dataclass(frozen=True)
class SRPreset:
    """One PRD 10.2 operating mode. Every value is recorded in the manifest."""

    mode: str
    label: str
    network: str  # "fidelity" or "perceptual" network of the backend (sr_models.BACKENDS)
    consistency_strength: float  # share of the low-frequency source residual put back
    intent: str
    warning: str = ""

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "label": self.label,
            "network": self.network,
            "consistency_strength": self.consistency_strength,
            "intent": self.intent,
            "warning": self.warning,
        }


PRESETS = {
    "conservative": SRPreset(
        "conservative",
        "Mode A, conservative",
        "fidelity",
        1.0,
        "Source consistency first: the fidelity network (pixel loss, no adversarial texture) and full "
        "low-frequency correction. Minimal invented texture; preferred for faces.",
    ),
    "balanced": SRPreset(
        "balanced",
        "Mode B, balanced",
        "perceptual",
        1.0,
        "Moderate detail reconstruction: the perceptual (adversarial) network for sharper edges and inferred "
        "texture, with full low-frequency correction so tone and colour stay on the source. Default for "
        "general scenes.",
    ),
    "aggressive": SRPreset(
        "aggressive",
        "Mode C, aggressive (experimental)",
        "perceptual",
        0.0,
        "Strongest perceptual reconstruction: the perceptual network unconstrained. Experimental comparison "
        "for PRD E02 only.",
        warning="Highest risk of invented content and drift; never the default and not for identity-critical regions.",
    ),
}


class SRModel:
    """Protocol of the network callables the tiler drives (``sr_models.TorchModel`` in production).

    ``__call__`` maps an sRGB-encoded (h, w, 3) float32 tile in [0, 1] to (h * scale, w * scale, 3),
    raising ``DeviceOutOfMemory`` when the tile does not fit; ``fallback`` returns a model on a
    device with more memory, or None.
    """

    scale: int
    device: str
    alignment: int = 1  # tile starts must be multiples of this (the attention window size), 1 for CNNs

    def __call__(self, encoded: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        raise NotImplementedError

    def fallback(self) -> SRModel | None:  # pragma: no cover - protocol
        return None


# --------------------------------------------------------------------------- tile geometry


def tile_spans(length: int, tile: int, overlap: int, align: int = 1) -> list[tuple[int, int]]:
    """(start, end) of tiles of about ``tile`` pixels covering ``length``, overlapping by at least ``overlap``.

    Tiles are spread evenly and every start is a multiple of ``align``. Window-attention networks such as
    SwinIR partition their input into fixed windows counted from the tile origin; aligned starts give
    neighbouring tiles the same window grid where they overlap, so they compute the same thing there (the
    measured difference with unaligned tiles was up to 6 code values after low-pass filtering). The last
    tile ends exactly at the border and may be up to ``align - 1`` pixels longer than ``tile``. A length that
    fits one such tile gives one tile, so tiling a span this function produced never splits it again.
    """
    if not 0 <= overlap < tile or align < 1:
        raise ValueError(f"need 0 <= overlap < tile and align >= 1, got {overlap}, {tile}, {align}")
    if length < tile + align:
        return [(0, length)]
    count = -(-(length - overlap) // (tile - overlap))
    while True:
        starts = [round(i * (length - tile) / (count - 1)) // align * align for i in range(count)]
        ends = [start + tile for start in starts[:-1]] + [length]
        if all(ends[i] - starts[i + 1] >= overlap for i in range(count - 1)):
            return list(zip(starts, ends, strict=True))
        count += 1


def axis_weights(centers: np.ndarray, spans: list[tuple[int, int]], length: int, overlap: int) -> np.ndarray:
    """Blending weight of each tile at each sample centre (in source pixels), shape (tiles, samples).

    Each tile's raw weight is 0 within ``overlap / 4`` of an interior tile edge and rises with a raised
    cosine over the next ``overlap / 2``; image borders keep full weight. The raw weights are then
    normalised so they sum to exactly 1 at every sample.
    """
    centers = np.asarray(centers, dtype=np.float64)
    margin, ramp = overlap / 4, max(overlap / 2, 1e-9)
    raw = np.empty((len(spans), centers.size), dtype=np.float64)
    for index, (start, end) in enumerate(spans):
        rising = np.ones_like(centers) if start == 0 else _raised_cosine((centers - start - margin) / ramp)
        falling = np.ones_like(centers) if end >= length else _raised_cosine((end - margin - centers) / ramp)
        raw[index] = rising * falling * ((centers >= start) & (centers < end))
    total = raw.sum(axis=0)
    if np.any(total <= 0):
        raise ValueError("tile weights leave samples uncovered; increase the overlap or the tile size")
    return (raw / total).astype(np.float32)


def _raised_cosine(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _alignment(model) -> int:
    return int(getattr(model, "alignment", 1) or 1)


# --------------------------------------------------------------------------- tiled inference


@dataclass
class TilingReport:
    tile_size: int
    tile_overlap: int
    grid: tuple[int, int] = (0, 0)
    tiles: int = 0
    model_calls: int = 0
    inference_seconds: float = 0.0
    fallbacks: list[dict] = field(default_factory=list)
    boundaries: list[dict] = field(default_factory=list)
    final_device: str = ""
    resample_range: dict = field(default_factory=lambda: new_range_record())

    def to_dict(self) -> dict:
        disagreements = [b["mean_abs_8bit"] for b in self.boundaries]
        low = [b["low_frequency_p99_abs_8bit"] for b in self.boundaries]
        blend = [b["blend_error_p99_abs_8bit"] for b in self.boundaries]
        return {
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "grid_rows_cols": list(self.grid),
            "tiles": self.tiles,
            "model_calls": self.model_calls,
            "inference_seconds": round(self.inference_seconds, 3),
            "fallbacks": self.fallbacks,
            "final_device": self.final_device,
            "resample_range": _range_summary(self.resample_range),
            "overlap_disagreement": {
                "unit": "sRGB 8-bit code values, inside the crossfade zones",
                "low_frequency": "difference smoothed with a Gaussian of 2 source pixels",
                "blend_error": "low-frequency difference times the smaller blending weight: how far the crossfade "
                "can move the image from the better-placed tile (the soft-seam measure)",
                "boundaries": len(self.boundaries),
                "blend_error_p99_abs_max": max(blend, default=None),
                "low_frequency_p99_abs_max": max(low, default=None),
                "mean_abs_max": max(disagreements, default=None),
                "p99_abs_max": max((b["p99_abs_8bit"] for b in self.boundaries), default=None),
                "max_abs": max((b["max_abs_8bit"] for b in self.boundaries), default=None),
                "per_boundary": self.boundaries,
            },
        }


class _Inference:
    """Runs tiles through the model, splitting tiles and falling back to the CPU on out-of-memory."""

    def __init__(self, model: SRModel, overlap: int, report: TilingReport, log: Callable[[str], None]):
        self.model = model
        self.overlap = overlap
        self.max_side = 1 << 30
        self.report = report
        self.log = log

    def __call__(self, encoded: np.ndarray) -> np.ndarray:
        side = max(encoded.shape[:2])
        if side >= self.max_side + _alignment(self.model):  # aligned spans may exceed the tile by align - 1
            return _tiled_native(encoded, self.max_side, min(self.overlap, self.max_side // 2), self)
        start = time.perf_counter()
        try:
            out = self.model(encoded)
        except DeviceOutOfMemory as error:
            message = str(error)
        else:
            self.report.inference_seconds += time.perf_counter() - start
            self.report.model_calls += 1
            self.report.final_device = getattr(self.model, "device", "")
            return out
        # Retry outside the except block: the exception's traceback holds the failed attempt's device tensors,
        # which must be released before the cache can be emptied.
        gc.collect()
        release = getattr(self.model, "release_cache", None)
        if release is not None:
            release()
        smaller = side // 2 // 8 * 8
        if smaller >= MIN_TILE:
            self.max_side = smaller
            self._event("split", f"{message}; tiles reduced to {smaller} px", side)
        else:
            fallback = self.model.fallback()
            if fallback is None:
                raise DeviceOutOfMemory(f"{message}; no fallback device left")
            self._event("device", f"{message}; continuing on {fallback.device}", side)
            self.model = fallback
            self.max_side = 1 << 30
        return self(encoded)

    def _event(self, kind: str, message: str, side: int) -> None:
        self.report.fallbacks.append({"kind": kind, "tile_side": side, "message": message})
        self.log(f"super-resolution memory fallback: {message}")


def _tiled_native(
    encoded: np.ndarray,
    tile: int,
    overlap: int,
    infer: Callable,
    report: TilingReport | None = None,
    origin_y: int = 0,
) -> np.ndarray:
    """Super-resolve an encoded image tile by tile, blended at model scale (x and y)."""
    height, width = encoded.shape[:2]
    scale, align = infer.model.scale, _alignment(infer.model)
    ys, xs = tile_spans(height, tile, overlap, align), tile_spans(width, tile, overlap, align)
    wy = axis_weights((np.arange(height * scale) + 0.5) / scale, ys, height, overlap)
    wx = axis_weights((np.arange(width * scale) + 0.5) / scale, xs, width, overlap)
    out = np.zeros((height * scale, width * scale, 3), dtype=np.float32)
    for iy, (y0, y1) in enumerate(ys):
        previous = None
        for ix, (x0, x1) in enumerate(xs):
            prediction = infer(encoded[y0:y1, x0:x1])
            rows, cols = slice(y0 * scale, y1 * scale), slice(x0 * scale, x1 * scale)
            weight = wy[iy, rows, None] * wx[ix, None, cols]
            out[rows, cols] += weight[..., None] * prediction
            if report is not None:
                report.tiles += 1
                if previous is not None and len(ys) == 1:
                    boundary = _disagreement("x", previous, (x0, x1), prediction, wx, ix, scale)
                    boundary["source_rows"] = [origin_y + y0, origin_y + y1]
                    report.boundaries.append(boundary)
                previous = (prediction, (x0, x1))
    return out


def _disagreement(axis, previous, span, prediction, weights, index, scale) -> dict:
    """Difference between two neighbouring tiles where both carry weight (the crossfade zone)."""
    prev_prediction, (px0, px1) = previous
    x0, x1 = span
    lo, hi = x0 * scale, px1 * scale
    both = np.nonzero((weights[index - 1, lo:hi] > 0) & (weights[index, lo:hi] > 0))[0]
    a = prev_prediction[:, lo - px0 * scale + both]
    b = prediction[:, lo - x0 * scale + both]
    mix = np.minimum(weights[index - 1, lo + both], weights[index, lo + both])[None, :, None]
    return _difference_stats(axis, (x0 + px1) / 2, (a - b) * 255, 2.0 * scale, mix)


def _difference_stats(axis: str, position: float, diff8: np.ndarray, sigma: float, mix: np.ndarray) -> dict:
    """Statistics of a signed difference (8-bit units) between two overlapping tiles.

    Raw and low-pass (Gaussian of 2 source pixels) differences describe how much the tiles disagree.
    The blend error weights the low-pass difference by the smaller of the two blending weights: the
    crossfaded result lies within that distance of the tile with more weight, so it bounds how far
    blending can move the image from either tile's own result at each sample. It is what a soft seam
    is made of; raw differences also count texture that the crossfade hides.
    """
    if diff8.size == 0:
        diff8, mix = np.zeros((1, 1, 3), np.float32), np.zeros((1, 1, 1), np.float32)
    low = np.abs(ndimage.gaussian_filter(diff8, (sigma, sigma, 0), mode="nearest"))
    diff8 = np.abs(diff8)
    return {
        "axis": axis,
        "position_source_px": round(float(position), 2),
        "mean_abs_8bit": round(float(diff8.mean()), 4),
        "p99_abs_8bit": round(float(np.percentile(diff8, 99)), 4),
        "max_abs_8bit": round(float(diff8.max()), 4),
        "low_frequency_p99_abs_8bit": round(float(np.percentile(low, 99)), 4),
        "blend_error_p99_abs_8bit": round(float(np.percentile(low * mix, 99)), 4),
    }


# --------------------------------------------------------------------------- full upscale


def super_resolve(
    source_linear: np.ndarray,
    width: int,
    height: int,
    model: SRModel,
    *,
    tile_size: int,
    tile_overlap: int,
    detail_strength: float = 1.0,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, TilingReport]:
    """Tiled SR of a working-space image, resampled to exactly (height, width, 3) float32 linear light."""
    if source_linear.ndim != 3 or source_linear.shape[2] != 3 or source_linear.dtype != np.float32:
        raise ValueError(f"expected an (H, W, 3) float32 image, got {source_linear.shape} {source_linear.dtype}")
    if tile_overlap < MIN_OVERLAP or tile_overlap * 2 > tile_size:
        raise ValueError(f"tile_overlap must be between {MIN_OVERLAP} and tile_size / 2, got {tile_overlap}")
    if not 0 <= detail_strength <= 1:
        raise ValueError(f"detail_strength must be in [0, 1], got {detail_strength}")
    src_h, src_w = source_linear.shape[:2]
    scale = model.scale
    report = TilingReport(tile_size, tile_overlap)
    infer = _Inference(model, tile_overlap, report, log)
    encoded = cm.linear_to_srgb(np.clip(source_linear, 0, 1))

    align = _alignment(model)
    strips = tile_spans(src_h, tile_size, tile_overlap, align)
    report.grid = (len(strips), len(tile_spans(src_w, tile_size, tile_overlap, align)))
    step_y = src_h / height  # source rows per target row
    wy = axis_weights((np.arange(height) + 0.5) * step_y, strips, src_h, tile_overlap)
    out = np.zeros((height, width, 3), dtype=np.float32)
    previous = None
    for index, (y0, y1) in enumerate(strips):
        rows = np.nonzero(wy[index])[0]
        j0, j1 = int(rows[0]), int(rows[-1]) + 1
        native = _tiled_native(encoded[y0:y1], tile_size, tile_overlap, infer, report, origin_y=y0)
        for start in range(0, native.shape[0], 128):  # decode in place, in row blocks, to bound temporaries
            native[start : start + 128] = cm.srgb_to_linear(native[start : start + 128])
        box = (0.0, (j0 * step_y - y0) * scale, float(src_w * scale), (j1 * step_y - y0) * scale)
        strip = resample_antiring(native, width, j1 - j0, box, report.resample_range, wy[index, j0:j1] > 0)
        del native
        if detail_strength < 1:
            lanczos_box = (0.0, j0 * step_y - y0, float(src_w), j1 * step_y - y0)
            lanczos = resample_box(source_linear[y0:y1], width, j1 - j0, lanczos_box, Image.Resampling.LANCZOS)
            strip = lanczos + np.float32(detail_strength) * (strip - lanczos)
            del lanczos
        if previous is not None:
            report.boundaries.append(_strip_disagreement(previous, strip, j0, wy, index, step_y))
        out[j0:j1] += wy[index, j0:j1, None, None] * strip
        previous = (strip, j0)
        log(f"super-resolution strip {index + 1}/{len(strips)}: target rows {j0}-{j1}")
    report.final_device = report.final_device or getattr(infer.model, "device", "")
    return out, report


def _range_summary(record: dict) -> dict:
    samples = record["samples"]
    return {
        "stage": "network output (clamped to [0, 1]) resampled with Lanczos-3 to the target",
        "min_linear_before": round(record["min"], 6) if samples else None,
        "max_linear_before": round(record["max"], 6) if samples else None,
        "below_zero_fraction_before": round(record["below_zero"] / samples, 8) if samples else None,
        "min_linear_after": round(record["min_after"], 6) if samples else None,
        "max_linear_after": round(record["max_after"], 6) if samples else None,
        "changed_fraction": round(record["changed"] / samples, 8) if samples else None,
        "bound": "anti-ringing: each sample clamped to the range of the network-output pixels under the main lobe "
        "of the kernel",
    }


def _strip_disagreement(previous, strip: np.ndarray, j0: int, wy: np.ndarray, index: int, step_y: float) -> dict:
    prev_strip, prev_j0 = previous
    hi = prev_j0 + prev_strip.shape[0]
    rows = np.arange(j0, hi)
    rows = rows[(wy[index - 1, rows] > 0) & (wy[index, rows] > 0)]
    a = cm.linear_to_srgb(np.clip(prev_strip[rows - prev_j0], 0, 1))
    b = cm.linear_to_srgb(np.clip(strip[rows - j0], 0, 1))
    mix = np.minimum(wy[index - 1, rows], wy[index, rows])[:, None, None]
    position = (rows.mean() + 0.5) * step_y if rows.size else j0 * step_y
    return _difference_stats("y", position, (a - b) * 255, 2.0 / step_y, mix)


def new_range_record() -> dict:
    """Accumulator for ``resample_antiring``: range before and after the clamp, over all strips."""
    return {
        "min": float("inf"),
        "max": float("-inf"),
        "min_after": float("inf"),
        "max_after": float("-inf"),
        "below_zero": 0,
        "changed": 0,
        "samples": 0,
    }


def resample_antiring(
    image: np.ndarray,
    width: int,
    height: int,
    box: tuple,
    record: dict | None = None,
    rows_used: np.ndarray | None = None,
    block_rows: int = 256,
) -> np.ndarray:
    """Lanczos-3 resize of ``box`` of a float32 (H, W, C) image without ringing.

    Each output sample is clamped to the minimum and maximum of the input pixels within one output pixel of
    its centre, which is where the main (positive) lobe of the stretched kernel lies; values beyond that
    range come only from the negative lobes. ``record`` accumulates the range before and after, over the
    output rows flagged in ``rows_used`` (all rows by default).
    """
    out = resample_box(image, width, height, box, Image.Resampling.LANCZOS)
    in_h, in_w = image.shape[:2]
    x0, y0, x1, y1 = box
    step_x, step_y = (x1 - x0) / width, (y1 - y0) / height
    cols = np.clip(np.floor(x0 + (np.arange(width) + 0.5) * step_x).astype(np.int64), 0, in_w - 1)
    rows = np.clip(np.floor(y0 + (np.arange(height) + 0.5) * step_y).astype(np.int64), 0, in_h - 1)
    reach_x, reach_y = int(np.ceil(max(step_x, 1.0))), int(np.ceil(max(step_y, 1.0)))
    vertical = np.ones((2 * reach_y + 1, 1), np.uint8)
    horizontal = np.ones((1, 2 * reach_x + 1), np.uint8)
    used = np.ones(height, dtype=bool) if rows_used is None else np.asarray(rows_used, dtype=bool)
    for start in range(0, height, block_rows):
        sampled = rows[start : start + block_rows]
        lo, hi = max(0, int(sampled[0]) - reach_y), min(in_h, int(sampled[-1]) + reach_y + 1)
        block = out[start : start + block_rows]
        mask = used[start : start + block_rows]
        if record is not None and mask.any():
            before = block[mask]  # boolean indexing copies
            record["min"] = min(record["min"], float(before.min()))
            record["max"] = max(record["max"], float(before.max()))
            record["below_zero"] += int(np.count_nonzero(before < 0))
            record["samples"] += before.size
        for channel in range(image.shape[2]):
            band = np.ascontiguousarray(image[lo:hi, :, channel])
            # cv2's default morphology border never wins a minimum or maximum, so image edges are neutral.
            low = cv2.erode(cv2.erode(band, vertical)[sampled - lo], horizontal)[:, cols]
            high = cv2.dilate(cv2.dilate(band, vertical)[sampled - lo], horizontal)[:, cols]
            np.clip(block[..., channel], low, high, out=block[..., channel])
            del band, low, high
        if record is not None and mask.any():
            values = block[mask]
            record["min_after"] = min(record["min_after"], float(values.min()))
            record["max_after"] = max(record["max_after"], float(values.max()))
            record["changed"] += int(np.count_nonzero(values != before))
            del before
    return out


def resample_box(image: np.ndarray, width: int, height: int, box: tuple, resampling) -> np.ndarray:
    """Resize the region ``box`` (x0, y0, x1, y1 in image pixels) of a float32 (H, W, C) image.

    Pillow reads filter support from outside the box when the image has it, so resampling a band of
    an image with the matching box equals resampling the whole image and cutting out the band.
    """
    out = np.empty((height, width, image.shape[2]), dtype=np.float32)
    for channel in range(image.shape[2]):
        plane = Image.fromarray(np.ascontiguousarray(image[..., channel]))
        out[..., channel] = np.asarray(plane.resize((width, height), resampling, box=box), dtype=np.float32)
        del plane
    return out


# --------------------------------------------------------------------------- hallucination controls


def project_to_source(output_linear: np.ndarray, source_linear: np.ndarray, strength: float) -> dict:
    """Low-frequency consistency correction, in place, applied as a gain so it never darkens to below zero.

    Per source pixel the gain is ``(source + e) / (area_down(out) + e)`` with ``e = PROJECTION_OFFSET``. Its
    logarithm, times ``strength``, is upsampled with Keys bicubic and the output becomes
    ``(out + e) * gain - e``. The area-reduced result then matches the source as closely as an additive
    residual would, which removes colour and tone drift at the source scale without touching detail above it.
    Unlike an additive residual, the excess energy is taken from where the energy is: near a bright light whose
    glow the network spread, an additive correction subtracts the same amount from the dark pixels around it
    and drives them below zero (black halos once clipped), while the gain keeps every non-negative sample above
    ``-e`` (results are bounded at 0). The area reduction is the one the source-consistency metrics use, so this
    step makes those metrics partly true by construction; the report says so.
    """
    from .baselines import downsample_area

    if not 0 <= strength <= 1:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    height, width = output_linear.shape[:2]
    src_h, src_w = source_linear.shape[:2]
    offset = np.float32(PROJECTION_OFFSET)
    before, after, gains = [], [], []
    minimum = float(output_linear.min())
    for channel in range(3):
        plane = output_linear[..., channel]
        if strength > 0:
            np.maximum(plane, 0, out=plane)
        reduced = downsample_area(np.ascontiguousarray(plane)[..., None], src_w, src_h)[..., 0]
        before.append(float(np.abs(source_linear[..., channel] - reduced).mean()))
        if strength > 0:
            target = np.maximum(source_linear[..., channel], 0) + offset
            log_gain = np.float32(strength) * np.log(target / (reduced + offset)).astype(np.float32)
            gains += [float(log_gain.min()), float(log_gain.max())]
            upsampled = Image.fromarray(np.ascontiguousarray(log_gain)).resize(
                (width, height), Image.Resampling.BICUBIC
            )
            gain = np.exp(np.asarray(upsampled, dtype=np.float32))
            del upsampled
            plane += offset
            plane *= gain
            plane -= offset
            np.maximum(plane, 0, out=plane)
            del gain
            reduced = downsample_area(np.ascontiguousarray(plane)[..., None], src_w, src_h)[..., 0]
            after.append(float(np.abs(source_linear[..., channel] - reduced).mean()))
        else:
            after.append(before[-1])
        del plane, reduced
    return {
        "strength": strength,
        "operator": "gain (source + e) / (area-reduced output + e) per source pixel, e = "
        f"{PROJECTION_OFFSET}, upsampled in log space with Keys bicubic, applied as (out + e) * gain - e in "
        "linear light; results bounded at 0",
        "mean_abs_residual_linear_before": round(float(np.mean(before)), 6),
        "mean_abs_residual_linear_after": round(float(np.mean(after)), 6),
        "gain_range": [round(float(np.exp(min(gains))), 4), round(float(np.exp(max(gains))), 4)] if gains else None,
        "min_linear_before": round(minimum, 6),
        "min_linear_after": round(float(output_linear.min()), 6),
    }


def interpolation_difference(
    output_linear: np.ndarray, source_linear: np.ndarray, chunk_rows: int = 512
) -> tuple[np.ndarray, dict]:
    """Per-pixel |SR - Lanczos| in sRGB luma (8-bit code values) and its statistics.

    The Lanczos interpolation of the source is what classical reconstruction would give, so the
    difference marks inferred (synthesised) detail and any drift; it is the PRD 10.5 difference
    heatmap. The map is float16 at output resolution.
    """
    height, width = output_linear.shape[:2]
    src_h, src_w = source_linear.shape[:2]
    step_y = src_h / height
    diff = np.empty((height, width), dtype=np.float16)
    histogram = np.zeros(256, dtype=np.int64)
    total = 0.0
    for j0 in range(0, height, chunk_rows):
        j1 = min(j0 + chunk_rows, height)
        y0 = max(0, int(j0 * step_y) - 8)
        y1 = min(src_h, int(np.ceil(j1 * step_y)) + 8)
        box = (0.0, j0 * step_y - y0, float(src_w), j1 * step_y - y0)
        lanczos = resample_box(source_linear[y0:y1], width, j1 - j0, box, Image.Resampling.LANCZOS)
        a = luma(cm.linear_to_srgb(np.clip(output_linear[j0:j1], 0, 1)))
        b = luma(cm.linear_to_srgb(np.clip(lanczos, 0, 1)))
        block = np.abs(a - b) * 255
        diff[j0:j1] = block
        total += float(block.sum(dtype=np.float64))
        histogram += np.bincount(np.minimum(block, 255).astype(np.int64).ravel(), minlength=256)[:256]
        del lanczos, a, b, block
    cumulative = np.cumsum(histogram) / histogram.sum()
    return diff, {
        "unit": "sRGB luma, 8-bit code values, |super-resolution - Lanczos| at output resolution",
        "mean": round(total / (height * width), 4),
        "p95": int(np.searchsorted(cumulative, 0.95)),
        "p99": int(np.searchsorted(cumulative, 0.99)),
        "fraction_above_8": round(float(1 - cumulative[7]), 6),
    }


def seam_scores(image_linear: np.ndarray, boundaries: list[dict], source_linear: np.ndarray) -> dict:
    """Hard-seam height at every tile-boundary line, in sRGB 8-bit code values, where the content is flat.

    Luma is smoothed along each line (Gaussian, 8 pixels; texture crossing the line averages out, a seam's
    offset does not) and the step across a line is measured three pixels wide, because resampling spreads
    a hard tile edge over two to three output pixels. At every sample along a boundary line p, the excess
    is the step across p minus the median step across the lines 4 to 12 pixels away; the score is the mean
    excess over the samples where the Lanczos interpolation of the source is flat across p (content edges
    lying on the line are left out). A score near 0 means no seam; a seam of k code values scores about k on
    smooth content and somewhat less where texture crosses the line.
    Soft (feathered) seams are measured by the blend error instead.
    """
    height, width = image_linear.shape[:2]
    src_h, src_w = source_linear.shape[:2]
    results = []
    for axis, size, src in (("x", width, src_w), ("y", height, src_h)):
        positions = sorted({round(b["position_source_px"] * size / src) for b in boundaries if b["axis"] == axis})
        for p in positions:
            lines = [p] + [p + k for k in (*range(-12, -3), *range(4, 13)) if 2 <= p + k < size - 1]
            if not 2 <= p < size - 1 or len(lines) < 5:
                continue
            lo, hi = min(lines) - 2, max(lines) + 2
            if axis == "x":
                band = image_linear[:, lo:hi]
                box = (lo * src_w / width, 0.0, hi * src_w / width, float(src_h))
                reference = resample_box(source_linear, hi - lo, height, box, Image.Resampling.LANCZOS)
            else:
                band = image_linear[lo:hi]
                box = (0.0, lo * src_h / height, float(src_w), hi * src_h / height)
                reference = resample_box(source_linear, width, hi - lo, box, Image.Resampling.LANCZOS)
            score = _excess_step(band, reference, axis, [q - lo for q in lines])
            results.append({"axis": axis, "position_px": p, "score": None if score is None else round(score, 4)})
    scores = [r["score"] for r in results if r["score"] is not None]
    return {
        "method": "mean excess luma step (8-bit codes) across each boundary line over lines 4 to 12 px away, "
        "on samples where the content is flat across the line",
        "max_score": max(scores, default=None),
        "mean_score": round(float(np.mean(scores)), 4) if scores else None,
        "per_boundary": results,
    }


def _line_steps(band: np.ndarray, axis: str, lines: list[int]) -> np.ndarray:
    """|y[p + 1] - y[p - 2]| in 8-bit codes along each line p, luma smoothed along the line; (lines, samples)."""
    y = luma(cm.linear_to_srgb(np.clip(band, 0, 1))) * 255
    if axis == "y":
        y = y.T
    y = ndimage.gaussian_filter1d(y, 8.0, axis=0, mode="nearest")
    return np.stack([np.abs(y[:, line + 1] - y[:, line - 2]) for line in lines])


def _excess_step(band: np.ndarray, reference: np.ndarray, axis: str, lines: list[int]) -> float | None:
    """Mean of (step at lines[0] - median step at the other lines) over samples flat in the reference."""
    steps = _line_steps(band, axis, lines)
    flat = _line_steps(reference, axis, lines[:1])[0] <= 1.0
    if flat.mean() < 0.1:
        return None
    excess = steps[0] - np.median(steps[1:], axis=0)
    return max(0.0, float(excess[flat].mean()))
