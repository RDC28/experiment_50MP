"""Human/crowd reconstruction building blocks (PRD 10, 18).

The default backend is interpolation, not pretrained SR. Both comparison modes
use the same stand-in until Phase 2 is available. Arrays remain unclipped linear
Rec.709 float32, including Lanczos overshoot; no sensor data is recovered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import cv2
import numpy as np

from . import baselines
from .pipeline import INTERPOLATION, StageRecord

UpscaleMode = Literal["conservative", "balanced"]
STAND_IN_DESCRIPTION = "stand-in: Lanczos interpolation, Phase 2 SR not yet available"


@dataclass(frozen=True)
class UpscaleResult:
    linear: np.ndarray
    backend: str
    mode: UpscaleMode
    description: str
    classification: str

    def stage_record(self) -> StageRecord:
        """Use the common manifest shape; the caller records timing and memory."""
        return StageRecord(
            name=f"human_upscale_{self.mode}",
            classification=self.classification,
            parameters={
                "sr_backend": self.backend,
                "mode": self.mode,
                "width": self.linear.shape[1],
                "height": self.linear.shape[0],
            },
            input_precision="float32 linear",
            output_precision="float32 linear",
            notes=[self.description],
        )


class Upscaler(Protocol):
    """Swappable backend: do not mutate input; return finite linear RGB float32.

    The result must describe the actual operation and its stage classification.
    A future pretrained backend must record its model provenance in description.
    """

    def __call__(self, linear: np.ndarray, width: int, height: int, mode: UpscaleMode) -> UpscaleResult: ...


def lanczos_stand_in(linear: np.ndarray, width: int, height: int, mode: UpscaleMode) -> UpscaleResult:
    """Conservative and balanced deliberately share the Phase 1 baseline."""
    _validate_image(linear)
    if mode not in ("conservative", "balanced"):
        raise ValueError("mode must be conservative or balanced")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (width, height)):
        raise ValueError("target width and height must be positive integers")
    return UpscaleResult(
        baselines.resample(linear, width, height, "lanczos"),
        "lanczos_stand_in",
        mode,
        STAND_IN_DESCRIPTION,
        INTERPOLATION,
    )


# One replacement point for the future Phase 2 adapter; no model code or downloads.
UPSCALERS: dict[str, Upscaler] = {"lanczos_stand_in": lanczos_stand_in}


def get_upscaler(sr_backend: str = "lanczos_stand_in") -> Upscaler:
    """Resolve a backend explicitly; never silently substitute another method."""
    try:
        return UPSCALERS[sr_backend]
    except KeyError:
        raise ValueError(f"unknown sr_backend {sr_backend!r}; expected one of {sorted(UPSCALERS)}") from None


def _validate_image(linear: np.ndarray) -> None:
    if (
        not isinstance(linear, np.ndarray)
        or linear.dtype != np.float32
        or linear.ndim != 3
        or linear.shape[2] != 3
        or min(linear.shape[:2]) < 1
        or not np.isfinite(linear).all()
    ):
        raise ValueError("expected a non-empty finite (H, W, 3) linear float32 image")


def blend_upscale(baseline: np.ndarray, selected: np.ndarray, strength: float) -> np.ndarray:
    """Interpolation blend; endpoint defaults preserve float32 samples exactly."""
    _validate_image(baseline)
    _validate_image(selected)
    _validate_strength(strength)
    if baseline.shape != selected.shape:
        raise ValueError("upscaler and baseline shapes must match")
    if strength == 1:
        return selected.copy()
    if strength == 0:
        return baseline.copy()
    return (baseline + strength * (selected - baseline)).astype(np.float32)


def sharpen_edges(linear: np.ndarray, strength: float, face_exclusion: np.ndarray) -> np.ndarray:
    """Simulation: bounded unsharp detail outside complete observed face boxes.

    Exclusion is output-sized binary support, including low-confidence faces.
    A one-pixel reflected Gaussian radius never changes excluded samples.
    """
    _validate_image(linear)
    _validate_strength(strength)
    if face_exclusion.shape != linear.shape[:2] or face_exclusion.dtype != np.bool_:
        raise ValueError("face_exclusion must be output-sized boolean support")
    if strength == 0:
        return linear.copy()
    blurred = cv2.GaussianBlur(linear, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT)
    detail = np.clip(linear - blurred, -0.02, 0.02)
    detail[face_exclusion] = 0
    return (linear + strength * detail).astype(np.float32)


def _validate_strength(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("strength must be finite numeric in [0, 1]")


def adjust_tonal_contrast(linear: np.ndarray, strength: float) -> np.ndarray:
    """Simulation: log-luminance slope 1+strength about linear mid-grey 0.18.

    Strength is [-0.5, 0.5]. RGB ratios are retained; gain is bounded to [0.5, 2]
    to limit shadow/highlight changes. Nonpositive luminance stays unchanged.
    Masters retain overshoot; export clipping is reported by the run layer.
    """
    _validate_image(linear)
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not np.isfinite(strength)
        or not -0.5 <= strength <= 0.5
    ):
        raise ValueError("tonal_contrast must be finite numeric in [-0.5, 0.5]")
    if strength == 0:
        return linear.copy()
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    gain = np.exp(np.clip(strength * np.log(np.maximum(luminance, 1e-6) / 0.18), -np.log(2), np.log(2)))
    gain[luminance <= 0] = 1
    return (linear * gain[..., None]).astype(np.float32)


def balance_colour(linear: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    """Simulation: independent linear RGB gains, each [0.5, 1.5], no clipping."""
    _validate_image(linear)
    validate_colour_gains(gains)
    if tuple(gains) == (1, 1, 1):
        return linear.copy()
    return (linear * np.asarray(gains, dtype=np.float32)).astype(np.float32)


def validate_colour_gains(gains) -> None:
    if not isinstance(gains, (tuple, list)) or len(gains) != 3:
        raise ValueError("colour_balance requires three linear RGB gains")
    for value in gains:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0.5 <= value <= 1.5
        ):
            raise ValueError("colour_balance gains must be finite numeric in [0.5, 1.5]")


def add_grain(linear: np.ndarray, strength: float, face_exclusion: np.ndarray, seed: int) -> np.ndarray:
    """Simulation: seeded achromatic uniform grain, bounded to 0.02 linear.

    Amplitude scales with sqrt(clipped luminance): darker pixels receive less
    noise. Spatial scale is one output pixel with independent samples; no
    source detail is recovered. Complete observed face boxes are suppressed.
    """
    _validate_image(linear)
    _validate_strength(strength)
    if face_exclusion.shape != linear.shape[:2] or face_exclusion.dtype != np.bool_:
        raise ValueError("face_exclusion must be output-sized boolean support")
    if strength == 0:
        return linear.copy()
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    noise = np.random.default_rng(seed).uniform(-1, 1, linear.shape[:2]).astype(np.float32)
    noise *= 0.02 * strength * np.sqrt(np.clip(luminance, 0, 1))
    noise[face_exclusion] = 0
    return (linear + noise[..., None]).astype(np.float32)


def smooth_skin(linear: np.ndarray, strength: float, skin_mask: np.ndarray, face_limit: np.ndarray) -> np.ndarray:
    """Simulation: skin-weighted bilateral smoothing with hard face blend caps.

    Masks are output-sized [0, 1] float weights. face_limit is one outside
    observed boxes and the configured face strength inside. Bounded deltas
    retain linear overshoot; zero strength is a bit-identical no-op.
    """
    _validate_image(linear)
    _validate_strength(strength)
    for mask in (skin_mask, face_limit):
        if (
            mask.shape != linear.shape[:2]
            or mask.dtype != np.float32
            or not np.isfinite(mask).all()
            or np.any((mask < 0) | (mask > 1))
        ):
            raise ValueError("smoothing masks must be output-sized float32 weights in [0, 1]")
    if strength == 0:
        return linear.copy()
    filtered = cv2.bilateralFilter(linear, 5, 0.05, 2, borderType=cv2.BORDER_REFLECT)
    weight = np.minimum(strength * skin_mask, face_limit)
    delta = np.clip(filtered - linear, -0.02, 0.02) * weight[..., None]
    output = (linear + delta).astype(np.float32)
    output[weight == 0] = linear[weight == 0]
    return output
