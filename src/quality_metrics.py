"""Source-consistency metrics and image statistics (PRD sections 23.2 and 27.4).

A 50 MP result is first reduced back to source resolution with an area
average, then compared with the source. Both images are compared as sRGB
encoded values in [0, 1] (the domain the TIFF master and the display use).

High source consistency shows the reconstruction did not drift from the
source. It does not show that added high-frequency detail is correct.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import color_management as cm

LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
MS_SSIM_WEIGHTS = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
_SSIM_SIGMA = 1.5
_SSIM_RADIUS = 5  # 11-tap Gaussian window, as in Wang et al.
LPIPS_NOTE = "not computed: LPIPS needs pretrained network weights, which Phase 0/1 does not download"


def source_consistency(reference_linear: np.ndarray, candidate_linear: np.ndarray) -> tuple[dict, np.ndarray]:
    """Compare two same-size working-space images; returns (metrics, per-pixel CIEDE2000 map)."""
    if reference_linear.shape != candidate_linear.shape:
        raise ValueError(f"shape mismatch {reference_linear.shape} vs {candidate_linear.shape}")
    reference = cm.linear_to_srgb(np.clip(reference_linear, 0, 1))
    candidate = cm.linear_to_srgb(np.clip(candidate_linear, 0, 1))
    delta_e, lightness_ref, lightness_cand = _ciede2000(reference, candidate)
    metrics = {
        "domain": "sRGB-encoded values in [0, 1] at source resolution (area-average reduction)",
        "psnr_db": psnr(reference, candidate),
        "ssim": ssim(reference, candidate),
        "ms_ssim": ms_ssim(luma(reference), luma(candidate)),
        "lpips": None,
        "lpips_note": LPIPS_NOTE,
        "edge_displacement": edge_displacement(luma(reference), luma(candidate)),
        "color_difference_ciede2000": {
            "mean": float(delta_e.mean()),
            "p95": float(np.percentile(delta_e, 95)),
            "max": float(delta_e.max()),
        },
        "luminance_distribution": _lightness_shift(lightness_ref, lightness_cand),
    }
    return metrics, delta_e


def luma(encoded: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(encoded @ LUMA_WEIGHTS)


def psnr(reference: np.ndarray, candidate: np.ndarray, data_range: float = 1.0) -> float:
    mse = _mean_squared_error(reference, candidate)
    return float("inf") if mse == 0 else float(10 * np.log10(data_range**2 / mse))


def ssim(reference: np.ndarray, candidate: np.ndarray, data_range: float = 1.0) -> float:
    """Mean SSIM over RGB channels with the Wang et al. Gaussian window."""
    values = [
        _ssim_maps(reference[..., c].astype(np.float64), candidate[..., c].astype(np.float64), data_range)[0].mean()
        for c in range(reference.shape[2])
    ]
    return float(np.mean(values))


def ms_ssim(reference: np.ndarray, candidate: np.ndarray, data_range: float = 1.0) -> dict:
    """Multiscale SSIM on luma (Wang, Simoncelli and Bovik 2003).

    Uses as many of the five standard scales as the image size allows and
    renormalises the weights when fewer fit.
    """
    x = reference.astype(np.float64)
    y = candidate.astype(np.float64)
    levels = 0
    while levels < len(MS_SSIM_WEIGHTS) and min(x.shape) // (2**levels) >= 2 * _SSIM_RADIUS + 1:
        levels += 1
    if levels == 0:
        return {"value": None, "levels": 0}
    weights = np.array(MS_SSIM_WEIGHTS[:levels])
    weights = weights / weights.sum()
    factors = []
    for level in range(levels):
        ssim_map, cs_map = _ssim_maps(x, y, data_range)
        if level == levels - 1:
            factors.append(max(float(ssim_map.mean()), 0.0))
        else:
            factors.append(max(float(cs_map.mean()), 0.0))
            x, y = _halve(x), _halve(y)
    return {"value": float(np.prod(np.power(factors, weights))), "levels": levels}


def edge_displacement(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Symmetric distance between Canny edges of two luma images, in source pixels."""
    from skimage.feature import canny

    reference_edges = canny(reference, sigma=1.0, low_threshold=0.05, high_threshold=0.15)
    candidate_edges = canny(candidate, sigma=1.0, low_threshold=0.05, high_threshold=0.15)
    if not reference_edges.any() or not candidate_edges.any():
        return {"mean_px": None, "p95_px": None, "edge_pixel_ratio": None, "reference_edge_pixels": 0}
    to_reference = ndimage.distance_transform_edt(~reference_edges)[candidate_edges]
    to_candidate = ndimage.distance_transform_edt(~candidate_edges)[reference_edges]
    distances = np.concatenate([to_reference, to_candidate])
    return {
        "mean_px": float((to_reference.mean() + to_candidate.mean()) / 2),
        "p95_px": float(np.percentile(distances, 95)),
        "edge_pixel_ratio": float(candidate_edges.sum() / reference_edges.sum()),
        "reference_edge_pixels": int(reference_edges.sum()),
    }


def image_statistics(linear: np.ndarray, chunk_rows: int = 512) -> dict:
    """Per-channel statistics of a working-space image, accumulated in row blocks."""
    channels = linear.shape[2]
    total = np.zeros(channels)
    total_sq = np.zeros(channels)
    minimum = np.full(channels, np.inf)
    maximum = np.full(channels, -np.inf)
    below = above = 0
    for start in range(0, linear.shape[0], chunk_rows):
        block = linear[start : start + chunk_rows].astype(np.float64)
        total += block.sum(axis=(0, 1))
        total_sq += np.square(block).sum(axis=(0, 1))
        minimum = np.minimum(minimum, block.min(axis=(0, 1)))
        maximum = np.maximum(maximum, block.max(axis=(0, 1)))
        below += int(np.count_nonzero(block < 0))
        above += int(np.count_nonzero(block > 1))
    count = linear.shape[0] * linear.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 0))
    names = ["R", "G", "B"][:channels]
    return {
        "domain": "linear working space",
        "mean": dict(zip(names, mean.round(6).tolist(), strict=True)),
        "std": dict(zip(names, std.round(6).tolist(), strict=True)),
        "min": dict(zip(names, minimum.round(6).tolist(), strict=True)),
        "max": dict(zip(names, maximum.round(6).tolist(), strict=True)),
        "below_zero_fraction": below / linear.size,
        "above_one_fraction": above / linear.size,
    }


def high_frequency_energy(encoded: np.ndarray) -> float:
    """Variance of the Laplacian of luma: a simple sharpness and ringing indicator."""
    return float(ndimage.laplace(luma(encoded).astype(np.float64)).var())


# --------------------------------------------------------------------------- internals


def _ssim_maps(x: np.ndarray, y: np.ndarray, data_range: float) -> tuple[np.ndarray, np.ndarray]:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    def blur(a: np.ndarray) -> np.ndarray:
        return ndimage.gaussian_filter(a, _SSIM_SIGMA, truncate=_SSIM_RADIUS / _SSIM_SIGMA)

    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x**2
    var_y = blur(y * y) - mu_y**2
    cov = blur(x * y) - mu_x * mu_y
    cs_map = (2 * cov + c2) / (var_x + var_y + c2)
    ssim_map = (2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1) * cs_map
    r = _SSIM_RADIUS
    if min(x.shape) > 2 * r:
        ssim_map, cs_map = ssim_map[r:-r, r:-r], cs_map[r:-r, r:-r]
    return ssim_map, cs_map


def _halve(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    a = a[:h, :w]
    return (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]) / 4


def _mean_squared_error(a: np.ndarray, b: np.ndarray, chunk_rows: int = 512) -> float:
    total = 0.0
    for start in range(0, a.shape[0], chunk_rows):
        diff = a[start : start + chunk_rows].astype(np.float64) - b[start : start + chunk_rows]
        total += float(np.square(diff).sum())
    return total / a.size


def _ciede2000(reference: np.ndarray, candidate: np.ndarray, chunk_rows: int = 256):
    """CIEDE2000 between two sRGB-encoded images, in row blocks to bound float64 temporaries."""
    from skimage.color import deltaE_ciede2000, rgb2lab

    height, width = reference.shape[:2]
    delta_e = np.empty((height, width), dtype=np.float32)
    lightness_ref = np.empty((height, width), dtype=np.float32)
    lightness_cand = np.empty((height, width), dtype=np.float32)
    for start in range(0, height, chunk_rows):
        rows = slice(start, start + chunk_rows)
        lab_ref = rgb2lab(reference[rows])
        lab_cand = rgb2lab(candidate[rows])
        delta_e[rows] = deltaE_ciede2000(lab_ref, lab_cand)
        lightness_ref[rows] = lab_ref[..., 0]
        lightness_cand[rows] = lab_cand[..., 0]
    return delta_e, lightness_ref, lightness_cand


def _lightness_shift(reference: np.ndarray, candidate: np.ndarray) -> dict:
    ref = reference.ravel()
    cand = candidate.ravel()
    ref_std = float(ref.std())
    return {
        "domain": "CIE L* (0 to 100)",
        "mean_shift": float(cand.mean() - ref.mean()),
        "std_ratio": float(cand.std() / ref_std) if ref_std > 0 else None,
        "wasserstein_1": float(np.abs(np.sort(ref) - np.sort(cand)).mean()),
    }
