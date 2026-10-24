"""Human/crowd reconstruction building blocks (PRD 10, 18).

The run-scoped Phase 2 adapter exposes learned SR with explicit provenance.
The legacy registry supplies the explicit interpolation comparison path.
Arrays remain linear Rec.709 float32; no sensor data is recovered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np

from . import baselines, sr_models
from . import super_resolution as sr
from .pipeline import (
    INFERENCE,
    INTERPOLATION,
    SEAM_BLEND_ERROR_LIMIT,
    SEAM_SCORE_LIMIT,
    SR_NOTICE,
    StageError,
    StageRecord,
    _current_model,
)

UpscaleMode = Literal["conservative", "balanced"]
STAND_IN_DESCRIPTION = "stand-in: Lanczos interpolation; no learned inference or recovered detail"


@dataclass(frozen=True)
class UpscaleResult:
    linear: np.ndarray
    backend: str
    mode: UpscaleMode
    description: str
    classification: str
    metadata: dict = field(default_factory=dict)

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
                **self.metadata,
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


class Phase2Upscaler:
    """Run-scoped learned SR adapter; cache networks, never full-size candidates.

    The caller owns this instance for one run and calls close even on failure.
    General blending belongs to the human reconstruction stage, so SR detail
    strength stays at one rather than introducing a second blend control.
    """

    def __init__(self, config, project_root: str | Path, *, log=print):
        if config.sr_backend not in sr_models.BACKENDS:
            raise ValueError(f"unknown learned sr_backend {config.sr_backend!r}")
        self.config = config
        self.weights_dir = (
            Path(config.sr_weights_dir) if config.sr_weights_dir else sr_models.default_weights_dir(Path(project_root))
        )
        self.log = log
        self.loaded = {}

    def __call__(self, linear, width, height, mode):
        _validate_image(linear)
        if mode not in ("conservative", "balanced"):
            raise ValueError("mode must be conservative or balanced")
        config = self.config
        preset = sr.PRESETS[mode]
        weights = sr_models.BACKENDS[config.sr_backend][preset.network]
        try:
            if not sr_models.available():
                raise RuntimeError("PyTorch and spandrel are required; install the [sr] extra")
            if weights not in self.loaded:
                self.loaded[weights] = sr_models.load_model(
                    weights,
                    self.weights_dir,
                    device=config.sr_device,
                    precision=config.sr_precision,
                    seed=config.random_seed,
                    log=self.log,
                )
        except Exception as error:
            raise StageError(
                f"human_upscale_{mode}",
                RuntimeError(f"{error}; use sr_backend: lanczos_stand_in for interpolation"),
                None,
            ) from error
        model = self.loaded[weights]
        tile, basis = config.sr_tile_size, "configured"
        if not tile:
            tile, basis = sr_models.auto_tile_size(
                model.device,
                model.precision,
                model.architecture,
                source_width=linear.shape[1],
                scale=model.scale,
            )
        tile = max(tile, 2 * config.sr_tile_overlap)
        output, tiling = sr.super_resolve(
            linear,
            width,
            height,
            model,
            tile_size=tile,
            tile_overlap=config.sr_tile_overlap,
            detail_strength=1.0,
            log=self.log,
        )
        model = self.loaded[weights] = _current_model(model, tiling)
        metadata = {
            "preset": preset.to_dict(),
            "weights": weights,
            "model": {**model.describe(), "weights": sr_models.WEIGHTS[weights].to_dict()},
            "native_scale": model.scale,
            "final_resample": "Lanczos-3 in linear light to exact target",
            "device": model.device,
            "precision": model.precision,
            "tiling": {**tiling.to_dict(), "tile_basis": basis},
        }
        if config.preserve_source_structure:
            metadata["source_projection"] = sr.project_to_source(output, linear, preset.consistency_strength)
        seams = sr.seam_scores(output, tiling.boundaries, linear)
        disagreement = metadata["tiling"]["overlap_disagreement"]
        seams["passed"] = (seams["max_score"] or 0) <= SEAM_SCORE_LIMIT and (
            disagreement["blend_error_p99_abs_max"] or 0
        ) <= SEAM_BLEND_ERROR_LIMIT
        seams["limits"] = {"max_score": SEAM_SCORE_LIMIT, "blend_error_p99_abs_8bit": SEAM_BLEND_ERROR_LIMIT}
        metadata["seams"] = seams
        return UpscaleResult(
            output,
            config.sr_backend,
            mode,
            f"{preset.label}; {weights}; native x4 then Lanczos-3 to target. {SR_NOTICE}",
            INFERENCE,
            metadata,
        )

    def close(self):
        """Release device caches and network references at the end of a run."""
        while self.loaded:
            _, model = self.loaded.popitem()
            model.release_cache()


# Stateless interpolation backends; learned backends require a run-scoped Phase2Upscaler.
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
