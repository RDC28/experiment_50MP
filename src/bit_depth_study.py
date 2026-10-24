"""Bit-depth experiments (PRD section 14.5 and experiment E01).

The study stores one processed linear image in each representation PRD 14.5
names and measures what each one keeps:

* 8-bit sRGB-encoded integer RGB (the review PNG),
* 16-bit sRGB-encoded integer RGB (the TIFF master's encoding),
* 16-bit floating-point linear RGB (half EXR),
* 32-bit floating-point linear RGB (float EXR; the working precision and the reference),
* optionally the quantised synthetic mosaic at the simulated ADC bit depth.

For each it reports quantisation error, file size, numerical headroom, gradient
smoothness after an identical strong tonal edit, and the error accumulated by
repeated forward and inverse transformations. Everything except file writing
runs in row blocks, so a 50 MP image needs no extra full-size float copies.

:func:`expansion_information` answers E01 on the source itself: it expands the
source codes to 16-bit integers and to floating point and measures distinct
levels, Shannon entropy and exact recoverability. A wider container holds the
same information; precision is not original scene information.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import color_management as cm
from . import exporters
from .quality_metrics import LUMA_WEIGHTS

_CHUNK_ROWS = 256
_CODES = 65536  # distinct levels are counted on a 16-bit sRGB encoding of each edited result
_TO_P3 = np.linalg.inv(cm.matrix_to_working(cm.DISPLAY_P3_PRIMARIES, cm.D65_XY)).astype(np.float32)
_FROM_P3 = cm.matrix_to_working(cm.DISPLAY_P3_PRIMARIES, cm.D65_XY).astype(np.float32)

EDIT_DESCRIPTION = "exposure +{stops:g} stops in linear light, then sRGB display encoding (shadow push)"
CYCLE_DESCRIPTION = (
    "linear Rec.709 -> Display P3 -> -1 stop -> store -> +1 stop -> Rec.709 -> store, repeated; "
    "each store re-quantises to the representation"
)


@dataclass(frozen=True)
class Representation:
    name: str
    description: str
    domain: str
    store: Callable[[np.ndarray], np.ndarray]  # working linear float32 -> stored samples
    load: Callable[[np.ndarray], np.ndarray]  # stored samples -> working linear float32
    levels_in_unit_range: int  # distinct representable values in [0, 1]
    max_value: float  # largest representable working value
    min_value: float  # smallest representable working value


def _store_uint8(linear: np.ndarray) -> np.ndarray:
    clean = np.clip(np.nan_to_num(linear), 0, 1)
    return np.rint(cm.linear_to_srgb(clean) * np.float32(255)).astype(np.uint8)


def _load_uint8(codes: np.ndarray) -> np.ndarray:
    return cm.srgb_to_linear(codes.astype(np.float32) / np.float32(255))


def _store_uint16(linear: np.ndarray) -> np.ndarray:
    return cm.encode_srgb_uint16(np.nan_to_num(linear))[0]


HALF_MAX = float(np.finfo(np.float16).max)
REPRESENTATIONS = (
    Representation(
        "rgb8_srgb",
        "8-bit integer RGB, sRGB encoded",
        "sRGB code values, clipped to [0, 1]",
        _store_uint8,
        _load_uint8,
        256,
        1.0,
        0.0,
    ),
    Representation(
        "rgb16_srgb",
        "16-bit integer RGB, sRGB encoded",
        "sRGB code values, clipped to [0, 1]",
        _store_uint16,
        cm.srgb_uint16_to_linear,
        65536,
        1.0,
        0.0,
    ),
    Representation(
        "rgb16_float_linear",
        "16-bit floating-point linear RGB (half)",
        "linear-light half floats",
        lambda linear: linear.astype(np.float16),
        lambda stored: stored.astype(np.float32),
        0x3C00 + 1,  # non-negative half values up to 1.0, including zero and subnormals
        HALF_MAX,
        -HALF_MAX,
    ),
    Representation(
        "rgb32_float_linear",
        "32-bit floating-point linear RGB (float, the reference)",
        "linear-light float32",
        lambda linear: np.array(linear, dtype=np.float32, copy=True),
        lambda stored: stored,
        0x3F800000 + 1,  # non-negative float32 values up to 1.0
        float(np.finfo(np.float32).max),
        -float(np.finfo(np.float32).max),
    ),
)


def run_study(
    linear: np.ndarray,
    out_dir: Path | None = None,
    *,
    edit_stops: float = 3.0,
    cycles: int = 5,
    file_metadata: dict | None = None,
) -> dict:
    """Measure each PRD 14.5 representation of a working-space (H, W, 3) float32 image."""
    if linear.ndim != 3 or linear.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) linear RGB array, got {linear.shape}")
    if not 0 < edit_stops <= 8 or not 1 <= cycles <= 100:
        raise ValueError("edit_stops must be in (0, 8] and cycles in [1, 100]")
    gain = np.float32(2.0**edit_stops)
    shadow_limit = float(2.0**-edit_stops)  # luminance the edit lifts to at most 1.0, so nothing clips
    accumulators = {r.name: _Accumulator() for r in REPRESENTATIONS}
    reference_levels = np.zeros((3, _CODES), dtype=np.int64)
    shadow_pixels = 0
    for start in range(0, linear.shape[0], _CHUNK_ROWS):
        reference = np.ascontiguousarray(linear[start : start + _CHUNK_ROWS], dtype=np.float32)
        shadows = (reference @ LUMA_WEIGHTS) < shadow_limit
        shadow_pixels += int(np.count_nonzero(shadows))
        reference_edit = _edit(reference, gain)
        _count_levels(reference_levels, reference_edit, shadows)
        reference_encoded = cm.linear_to_srgb(np.clip(reference, 0, 1))
        for representation in REPRESENTATIONS:
            acc = accumulators[representation.name]
            loaded = representation.load(representation.store(reference))
            representable = np.clip(reference, representation.min_value, representation.max_value)
            acc.add_error(loaded - representable)
            acc.clip_sq += float(np.square((representable - reference).astype(np.float64)).sum())
            acc.outside += int(np.count_nonzero(representable != reference))
            encoded = cm.linear_to_srgb(np.clip(loaded, 0, 1))
            acc.encoded_sq += float(np.square((encoded - reference_encoded).astype(np.float64)).sum())
            edited = _edit(loaded, gain)
            acc.edit_sq += float(np.square((edited - reference_edit).astype(np.float64)).sum())
            _count_levels(acc.levels, edited, shadows)
            cycled = reference
            for _ in range(cycles):
                cycled = _cycle(cycled, representation)
            drift = cycled - loaded
            acc.cycle_sq += float(np.square(drift.astype(np.float64)).sum())
            acc.cycle_max = max(acc.cycle_max, float(np.abs(drift).max(initial=0.0)))
    samples = linear.size
    reference_shadow_levels = _mean_levels(reference_levels)
    rows = []
    for representation in REPRESENTATIONS:
        acc = accumulators[representation.name]
        levels = _mean_levels(acc.levels)
        row = {
            "name": representation.name,
            "description": representation.description,
            "domain": representation.domain,
            "levels_in_unit_range": representation.levels_in_unit_range,
            "max_value": representation.max_value,
            "outside_range_fraction": acc.outside / samples,
            "clipping_rmse_linear": float(np.sqrt(acc.clip_sq / samples)),
            "quantization_rmse_linear": float(np.sqrt(acc.sq / samples)),
            "quantization_max_abs_linear": acc.max_abs,
            "psnr_encoded_db": _psnr(acc.encoded_sq / samples),
            "edit_rmse_encoded": float(np.sqrt(acc.edit_sq / samples)),
            "edit_shadow_levels": levels,
            "edit_shadow_level_ratio": levels / reference_shadow_levels if reference_shadow_levels else None,
            "cycle_drift_rmse_linear": float(np.sqrt(acc.cycle_sq / samples)),
            "cycle_drift_max_abs_linear": acc.cycle_max,
            "file": None,
        }
        rows.append(row)
    if out_dir is not None:
        for row in rows:
            row["file"] = _write_file(Path(out_dir), row["name"], linear, file_metadata or {})
    return {
        "domain": "working-space linear Rec.709 input; the float32 representation is the reference",
        "parameters": {
            "edit": EDIT_DESCRIPTION.format(stops=edit_stops),
            "shadow_definition": f"reference luminance below {shadow_limit:g} (2^-{edit_stops:g})",
            "cycles": cycles,
            "cycle": CYCLE_DESCRIPTION,
            "quantization_error": "stored and reloaded value vs the input clipped to the representable range",
            "clipping_error": "input values the representation cannot hold (headroom), measured separately",
            "cycle_drift": "error after the repeated cycles relative to a single store",
        },
        "shadow_pixel_fraction": shadow_pixels / (linear.shape[0] * linear.shape[1]),
        "reference_edit_shadow_levels": reference_shadow_levels,
        "representations": rows,
    }


def mosaic_quantization(signal: np.ndarray, black: int, white: int, bit_depth: int) -> dict:
    """Noise-free quantisation error of a synthetic mosaic at its ADC bit depth (signal: fraction of range)."""
    span = white - black
    sq = 0.0
    max_abs = 0.0
    clipped = 0
    count = 0
    for start in range(0, signal.shape[0], _CHUNK_ROWS * 4):
        block = signal[start : start + _CHUNK_ROWS * 4].astype(np.float64)
        inside = (block >= 0) & (block <= 1)
        clipped += int(block.size - np.count_nonzero(inside))
        values = block[inside]
        error = (np.rint(black + values * span) - black) / span - values
        sq += float(np.square(error).sum())
        max_abs = max(max_abs, float(np.abs(error).max(initial=0.0)))
        count += values.size
    return {
        "name": f"mosaic_{bit_depth}bit",
        "description": f"synthetic CFA mosaic, {bit_depth}-bit ADC codes (one channel per pixel, simulated)",
        "domain": "camera-native signal as a fraction of the black-to-white range",
        "levels_in_unit_range": span + 1,
        "outside_range_fraction": clipped / max(signal.size, 1),
        "rmse": float(np.sqrt(sq / max(count, 1))),
        "max_abs_error": max_abs,
        "expected_rmse": 1 / (span * np.sqrt(12)),
    }


def expansion_information(codes: np.ndarray) -> dict:
    """E01: does expanding the source's integer codes to more bits add information? Measured, not assumed."""
    if codes.dtype.kind != "u":
        raise ValueError(f"expected unsigned integer source codes, got {codes.dtype}")
    planes = codes.reshape(-1, codes.shape[-1]) if codes.ndim == 3 else codes.reshape(-1, 1)
    code_max = float(np.iinfo(codes.dtype).max)
    expansions = {
        f"source codes ({codes.dtype.itemsize * 8}-bit integer)": (planes, lambda v: v),
        "16-bit integer expansion": (
            np.rint(planes.astype(np.float64) * (65535 / code_max)).astype(np.uint16),
            lambda v: np.rint(v.astype(np.float64) * (code_max / 65535)).astype(codes.dtype),
        ),
        "16-bit float expansion (normalised)": (
            (planes / code_max).astype(np.float16),
            lambda v: np.rint(v.astype(np.float64) * code_max).astype(codes.dtype),
        ),
        "32-bit float expansion (normalised)": (
            (planes / code_max).astype(np.float32),
            lambda v: np.rint(v.astype(np.float64) * code_max).astype(codes.dtype),
        ),
    }
    rows = []
    for name, (values, back) in expansions.items():
        distinct = [int(np.unique(values[:, c]).size) for c in range(values.shape[1])]
        entropy = [_entropy(values[:, c]) for c in range(values.shape[1])]
        rows.append(
            {
                "representation": name,
                "bits_per_sample": values.dtype.itemsize * 8,
                "distinct_values_per_channel": distinct,
                "entropy_bits_per_sample": [round(e, 6) for e in entropy],
                "recovers_source_exactly": bool(np.array_equal(back(values), planes)),
            }
        )
    source = rows[0]
    added = any(
        sum(r["distinct_values_per_channel"]) > sum(source["distinct_values_per_channel"])
        or sum(r["entropy_bits_per_sample"]) > sum(source["entropy_bits_per_sample"]) + 1e-9
        for r in rows
    )
    lossy = [r["representation"] for r in rows if not r["recovers_source_exactly"]]
    finding = (
        "No expansion has more distinct values or entropy than the source codes: more bits per sample add "
        "container precision, not information."
    )
    if lossy:
        finding += f" Not exactly invertible (they lose source codes): {', '.join(lossy)}."
    else:
        finding += " Every expansion maps back to the source codes without loss."
    return {"rows": rows, "information_added": added, "lossy_representations": lossy, "finding": finding}


# --------------------------------------------------------------------------- internals


class _Accumulator:
    def __init__(self) -> None:
        self.sq = 0.0
        self.clip_sq = 0.0
        self.max_abs = 0.0
        self.outside = 0
        self.encoded_sq = 0.0
        self.edit_sq = 0.0
        self.cycle_sq = 0.0
        self.cycle_max = 0.0
        self.levels = np.zeros((3, _CODES), dtype=np.int64)

    def add_error(self, error: np.ndarray) -> None:
        self.sq += float(np.square(error.astype(np.float64)).sum())
        self.max_abs = max(self.max_abs, float(np.abs(error).max(initial=0.0)))


def _edit(linear: np.ndarray, gain: np.float32) -> np.ndarray:
    return cm.linear_to_srgb(np.clip(linear * gain, 0, 1))


def _cycle(linear: np.ndarray, representation: Representation) -> np.ndarray:
    darker = (linear @ _TO_P3.T) * np.float32(0.5)
    restored = (representation.load(representation.store(darker)) * np.float32(2.0)) @ _FROM_P3.T
    return representation.load(representation.store(restored))


def _count_levels(histograms: np.ndarray, edited: np.ndarray, mask: np.ndarray) -> None:
    if not mask.any():
        return
    codes = np.rint(edited[mask] * np.float32(_CODES - 1)).astype(np.int64)
    for channel in range(3):
        histograms[channel] += np.bincount(codes[:, channel], minlength=_CODES)


def _mean_levels(histograms: np.ndarray) -> float:
    return float(np.count_nonzero(histograms, axis=1).mean())


def _psnr(mse: float) -> float:
    return float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))


def _entropy(values: np.ndarray) -> float:
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _write_file(out_dir: Path, name: str, linear: np.ndarray, metadata: dict) -> dict:
    """Write the representation in its natural format and return path and size."""
    attributes = {"comments": str(metadata.get("notice", "bit-depth study sample")), "bitDepthStudy": name}
    if name == "rgb8_srgb":
        stored = _store_uint8(linear)
        path = exporters.write_review_image(out_dir / f"{name}.png", stored)
        del stored
        return {"path": str(path), "format": "PNG, 8-bit, sRGB profile", "size_bytes": path.stat().st_size}
    if name == "rgb16_srgb":
        stored = _store_uint16(linear)
        result = exporters.write_tiff16(out_dir / f"{name}.tif", stored, metadata)
        del stored
        return {"path": str(result.path), "format": "TIFF, 16-bit, Deflate", "size_bytes": result.size_bytes}
    precision = "half" if name == "rgb16_float_linear" else "float"
    result = exporters.write_exr(out_dir / f"{name}.exr", linear, attributes, precision=precision)
    return {"path": str(result.path), "format": f"OpenEXR, {precision}, ZIP", "size_bytes": result.size_bytes}
