"""The 95% gate and its reported diagnostics, in linear RAW space on the 16-bit scale (section 6).

Arrays are black-subtracted DN16. A "layout" is either a CFA plane index (H, W) with values 0, 1, 2
(R, G, B; both greens pooled into G), or None for (H, W, 3) camera-RGB images.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from prd_v6 import constants, resample

PLANES = ("R", "G", "B")
VarianceFn = Callable[[np.ndarray, int], np.ndarray]


def noise_variance(x: np.ndarray, *, binned: tuple[float, float, float] = (1.0, 1.0, 1.0), plane: int = 1):
    """Variance in DN16^2 of a black-subtracted DN16 value under the measured A1 II noise model.

    Shot noise from the measured gain, read noise and 14-bit quantization, all on the 14-bit scale and
    multiplied by 4^2. `binned[plane]` is the number of same-colour photosites averaged into a value.
    """
    g = constants.GAIN_E_PER_DN14
    r = constants.READ_NOISE_E
    dn14 = np.maximum(np.asarray(x, dtype=np.float64) / constants.SCALE_16, 0.0)
    var14 = dn14 / g + (r / g) ** 2 + 1.0 / 12.0
    return var14 * constants.SCALE_16**2 / binned[plane]


def binned_counts(ratio: float) -> tuple[float, float, float]:
    """Same-colour photosites per output pixel at a downsampling ratio (RGGB: 1/4 R, 1/2 G, 1/4 B)."""
    area = ratio * ratio
    return (max(area / 4, 1.0), max(area / 2, 1.0), max(area / 4, 1.0))


def plane_values(values: np.ndarray, layout: np.ndarray | None, mask: np.ndarray, plane: int) -> np.ndarray:
    if layout is None:
        return np.asarray(values[..., plane], dtype=np.float64)[mask]
    return np.asarray(values, dtype=np.float64)[mask & (layout == plane)]


def split_planes(values, layout, mask) -> list[np.ndarray]:
    return [plane_values(values, layout, mask, p) for p in range(3)]


def fit_k(xs: list[np.ndarray], ys: list[np.ndarray]) -> float:
    """One global exposure scalar over the mask, shared by all planes (section 6.2)."""
    num = sum(float(np.dot(x, y)) for x, y in zip(xs, ys, strict=True))
    den = sum(float(np.dot(y, y)) for y in ys)
    if den <= 0 or not math.isfinite(num / den):
        raise ValueError("candidate has no signal in the mask")
    return num / den


def p1(x: np.ndarray, y: np.ndarray, k: float) -> float:
    energy = float(np.dot(x, x))
    return 1.0 - math.sqrt(float(np.sum((k * y - x) ** 2)) / energy) if energy > 0 else float("nan")


def p1_l1(x: np.ndarray, y: np.ndarray, k: float) -> float:
    total = float(np.sum(np.abs(x)))
    return 1.0 - float(np.sum(np.abs(k * y - x))) / total if total > 0 else float("nan")


def p2(x: np.ndarray, y: np.ndarray, k: float, var_ref: np.ndarray, var_syn: np.ndarray | float) -> float:
    if x.size == 0:
        return float("nan")
    sigma = np.sqrt(var_ref + k * k * var_syn)
    tolerance = np.maximum(constants.P2_RELATIVE * x, constants.P2_SIGMAS * sigma)
    return float(np.mean(np.abs(k * y - x) <= tolerance))


def score(
    reference: np.ndarray,
    candidate: np.ndarray,
    layout: np.ndarray | None,
    mask: np.ndarray,
    *,
    variance: VarianceFn,
    candidate_noisy: bool,
    k: float | None = None,
    zones: bool = False,
    extra_masks: dict[str, np.ndarray] | None = None,
) -> dict:
    """Score one candidate against the reference: P1, P2 per plane and the gate result.

    `variance(x, plane)` is sigma_ref^2; sigma_syn^2 uses the same model on the candidate when it
    carries the pipeline's noise, and is zero otherwise.
    """
    xs = split_planes(reference, layout, mask)
    ys = split_planes(candidate, layout, mask)
    fitted = fit_k(xs, ys) if k is None else float(k)
    result: dict = {"k": fitted, "k_ev": math.log2(fitted) if fitted > 0 else float("nan"), "planes": {}}
    for p, name in enumerate(PLANES):
        x, y = xs[p], ys[p]
        var_ref = variance(x, p)
        var_syn = variance(y, p) if candidate_noisy else 0.0
        plane = {
            "samples": int(x.size),
            "A": p1(x, y, fitted),
            "P2": p2(x, y, fitted, var_ref, var_syn),
            "A_l1": p1_l1(x, y, fitted),
            "A_unfitted": p1(x, y, 1.0),
            "P2_unfitted": p2(x, y, 1.0, var_ref, var_syn),
        }
        plane_k = float(np.dot(x, y) / np.dot(y, y)) if np.dot(y, y) > 0 else float("nan")
        plane["diagnostic_plane_fitted"] = {"k": plane_k, "A": p1(x, y, plane_k)}
        if zones:
            white = constants.SIGNAL_16
            bands = {"shadows": x < 0.02 * white, "midtones": (x >= 0.02 * white) & (x <= 0.5 * white)}
            bands["highlights"] = x > 0.5 * white
            plane["zones"] = {
                zone: {
                    "fraction": float(np.mean(sel)) if x.size else float("nan"),
                    "A": p1(x[sel], y[sel], fitted) if sel.any() else None,
                    "P2": p2(x[sel], y[sel], fitted, var_ref[sel], _take(var_syn, sel)) if sel.any() else None,
                }
                for zone, sel in bands.items()
            }
        result["planes"][name] = plane
    for label, extra in (extra_masks or {}).items():
        sub = mask & extra
        xs_e, ys_e = split_planes(reference, layout, sub), split_planes(candidate, layout, sub)
        result[label] = {
            name: {
                "samples": int(xs_e[p].size),
                "A": p1(xs_e[p], ys_e[p], fitted) if xs_e[p].size else None,
                "P2": p2(
                    xs_e[p],
                    ys_e[p],
                    fitted,
                    variance(xs_e[p], p),
                    variance(ys_e[p], p) if candidate_noisy else 0.0,
                )
                if xs_e[p].size
                else None,
            }
            for p, name in enumerate(PLANES)
        }
    return finish(result)


def _take(values, selection):
    return values[selection] if isinstance(values, np.ndarray) else values


def finish(result: dict) -> dict:
    planes = result["planes"]
    result["min_A"] = min(planes[p]["A"] for p in PLANES)
    result["min_P2"] = min(planes[p]["P2"] for p in PLANES)
    result["passes"] = all(planes[p]["A"] >= constants.P1_MIN and planes[p]["P2"] >= constants.P2_MIN for p in PLANES)
    result["worst_plane"] = min(PLANES, key=lambda p: planes[p]["A"])
    return result


def band_split(
    reference: np.ndarray, candidate: np.ndarray, layout: np.ndarray, mask: np.ndarray, k: float, source_hw
) -> dict:
    """50 MP only: error energy below and above the source Nyquist frequency, per CFA sub-image.

    The low band of the error is its area-downsample to the source size, cubic-upsampled back.
    """
    out = {}
    energies = {p: [0.0, 0.0, 0.0] for p in range(3)}  # low, high, reference
    for dy in (0, 1):
        for dx in (0, 1):
            x = np.asarray(reference[dy::2, dx::2], dtype=np.float32)
            error = k * np.asarray(candidate[dy::2, dx::2], dtype=np.float32) - x
            m = mask[dy::2, dx::2]
            plane = int(layout[dy, dx])
            shrink = (min(source_hw[0], error.shape[0]), min(source_hw[1], error.shape[1]))
            low = resample.cubic(resample.area(error, shrink), error.shape)
            high = error - low
            energies[plane][0] += float(np.sum(low[m].astype(np.float64) ** 2))
            energies[plane][1] += float(np.sum(high[m].astype(np.float64) ** 2))
            energies[plane][2] += float(np.sum(x[m].astype(np.float64) ** 2))
    for p, name in enumerate(PLANES):
        low, high, ref = energies[p]
        out[name] = {
            "nrmse_below_source_nyquist": math.sqrt(low / ref) if ref else None,
            "nrmse_above_source_nyquist": math.sqrt(high / ref) if ref else None,
        }
    return out


def error_map(reference: np.ndarray, candidate: np.ndarray, k: float, mask: np.ndarray, limit: float = 0.2):
    """Signed relative error (k*y - x) / x, clipped to +-limit, NaN outside the mask."""
    x = np.asarray(reference, dtype=np.float32)
    rel = (k * np.asarray(candidate, dtype=np.float32) - x) / np.maximum(x, 1.0)
    rel = np.clip(rel, -limit, limit)
    if rel.ndim == 3:
        rel[~mask] = np.nan
    else:
        rel = np.where(mask, rel, np.nan)
    return rel


def histogram(reference, candidate, k, mask, layout, bins: int = 81, limit: float = 0.2) -> dict:
    edges = np.linspace(-limit, limit, bins + 1)
    out = {"edges": edges.round(6).tolist()}
    for p, name in enumerate(PLANES):
        x = plane_values(reference, layout, mask, p)
        y = plane_values(candidate, layout, mask, p)
        rel = np.clip((k * y - x) / np.maximum(x, 1.0), -limit, limit)
        out[name] = np.histogram(rel, bins=edges)[0].tolist()
    return out
