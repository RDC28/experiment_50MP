"""Bounded, explicitly heuristic colour-constancy evidence for analytic ISP."""

import numpy as np

from prd_v5.analytic import AnalyticISP
from prd_v5.fields import resolve_isp
from prd_v5.hypothesis import CameraField


def neutral_white_balance(camera_to_srgb, neutral) -> np.ndarray:
    """Analytic gains whose DNG AsShotNeutral equals the green-normalized neutral."""
    white = np.linalg.solve(np.asarray(camera_to_srgb, dtype=np.float64), np.ones(3))
    return white / np.asarray(neutral, dtype=np.float64) / white[1]


def resolve_gray_world(
    base: AnalyticISP, source: np.ndarray, *, strip_rows: int = 64
) -> tuple[AnalyticISP, tuple[CameraField, ...], dict]:
    """Estimate G-normalized gains from an explicitly sRGB-rendered proxy.

    This assumes the average non-endpoint scene colour is neutral and treats
    inverse-sRGB channels as a camera-channel proxy. Existing rendered white
    balance, tone mapping and camera colour response cannot be disentangled.
    It is opt-in evidence, not a learned model or recovered capture illuminant.
    Every pixel is examined in bounded strips; endpoint pixels are excluded in
    all channels. Empty/degenerate evidence fails without a unity fallback.
    The estimate is the AsShotNeutral; gains map it through base.camera_to_srgb
    exactly as the DNG writer inverts them. Returned gains drive the declared
    analytic forward model, whose inverse divides by those gains. They do not
    white-balance the input rendered image.
    """
    if not isinstance(base, AnalyticISP):
        raise ValueError("base must be an AnalyticISP")
    if type(strip_rows) is not int or strip_rows < 1:
        raise ValueError("strip_rows must be a positive integer")
    if not isinstance(source, np.ndarray) or source.ndim != 3 or source.shape[-1] != 3 or min(source.shape) < 1:
        raise ValueError("source must be nonempty HxWx3")
    if source.dtype.kind != "f":
        raise ValueError("source must be normalized floating point encoded sRGB")
    sums = np.zeros(3, dtype=np.float64)
    count = 0
    for start in range(0, source.shape[0], strip_rows):
        block = np.asarray(source[start : start + strip_rows], dtype=np.float64)
        if not np.isfinite(block).all() or np.any(block < 0) or np.any(block > 1):
            raise ValueError("source must contain finite normalized [0, 1] samples")
        valid = np.all((block > 0) & (block < 1), axis=-1)
        samples = block[valid]
        linear = np.where(samples <= 0.04045, samples / 12.92, ((samples + 0.055) / 1.055) ** 2.4)
        sums += np.sum(linear, axis=0, dtype=np.float64)
        count += len(samples)
    if count == 0 or np.any(sums <= 0) or not np.isfinite(sums).all():
        raise ValueError("No nondegenerate gray-world evidence")
    means = sums / count
    neutral = means / means[1]
    gains = neutral_white_balance(base.camera_to_srgb, neutral)
    if not np.isfinite(gains).all() or np.any(gains <= 0):
        raise ValueError("Gray-world evidence produced invalid gains")
    limitations = (
        "Gray-world scene prior and rendered linear-sRGB channel proxy only; existing WB, "
        "tone and camera response are unknown. Not calibrated illuminant/capture WB recovery. "
        "Analytic inverse divides by selected gains; this does not neutralize rendered input."
    )
    report = {
        "status": "effective",
        "method": "gray_world_linear_srgb_proxy",
        "eligible_pixels": count,
        "excluded_endpoint_pixels": int(source.shape[0] * source.shape[1] - count),
        "linear_channel_means": means.tolist(),
        "as_shot_neutral": neutral.tolist(),
        "white_balance": gains.tolist(),
        "confidence": None,
        "limitations": limitations,
    }
    candidate = CameraField(
        "white_balance",
        gains.tolist(),
        "ratio",
        "ESTIMATE",
        "ESTIMATED",
        None,
        report,
        "Gray-world on non-endpoint inverse-sRGB pixels; neutral-average scene prior",
        limitations,
    )
    isp, records = resolve_isp(base, {"white_balance": "ESTIMATE"}, {"white_balance": (candidate,)})
    return isp, records, report
