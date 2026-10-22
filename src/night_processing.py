"""Nighttime scene reconstruction, Layer 5B (PRD sections 8.3, 12, 19 and 23.5).

The night layer runs after upscaling. It works on the working-space (linear
Rec.709, float32) upscale, with region masks computed once on the source:

* Luminance-region segmentation (PRD 8.3) labels every source pixel as one of
  :data:`REGION_NAMES`. Labels can be visualised and overridden by hand.
* Shadow, highlight, colour, chroma-noise, banding and atmospheric processing
  (PRD 12.1 to 12.5) are applied region by region with soft (feathered) masks.
* The failure rule (PRD 12.6) compares every block of the processed image
  with its input and falls back to a more conservative local result where the
  processing developed halos, banding, black crush or excess texture. Every
  fallback block is recorded.
* Night metrics (PRD 23.5) are measured before and after.

Every stage carries a provenance label (:data:`PROVENANCE_LABELS`) that maps to
the ``information`` record of the run manifest, and a PRD 5.2 classification.
Clipped highlights get a roll-off, never recovered radiance (PRD 12.2).

A 50 MP image is processed in horizontal strips with overlapping margins, so
peak memory is the input and output buffers plus a few strip-sized
temporaries. Results do not depend on the strip height, except for the
seeded dither and grain noise.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from scipy import ndimage

from . import baselines
from . import color_management as cm
from . import config as lab_config
from .quality_metrics import LUMA_WEIGHTS, luma

# --------------------------------------------------------------------------- provenance

PRESERVED = "preserved"
ESTIMATED = "estimated"
SYNTHESIZED = "synthesized"
SIMULATED = "simulated"
PROVENANCE_LABELS = (PRESERVED, ESTIMATED, SYNTHESIZED, SIMULATED)
# Keys of the manifest ``information`` record (pipeline._provenance) each label reports under.
MANIFEST_INFORMATION_KEYS = {
    PRESERVED: "preserved",
    ESTIMATED: "interpolated",
    SYNTHESIZED: "inferred",
    SIMULATED: "simulated",
}
# PRD 5.2 classifications used by night stages.
TONAL = "Preservation (pointwise tone/colour remapping of existing values, adds no information)"
SPATIAL = "Interpolation (values estimated from neighbouring samples)"
INFERENCE = "Inference"
SIMULATION = "Simulation"

# --------------------------------------------------------------------------- parameters

_CLIP_LEVEL = 1.0 - 0.5 / 65535  # linear value that encodes to the top 16-bit code
_CODE = 1.0 / 255  # one 8-bit code value in the sRGB-encoded domain


def _param(
    default: Any,
    description: str,
    accepted: str,
    effect: str,
    *,
    warning: str = "",
    choices: tuple | None = None,
    bounds: tuple[float, float] | None = None,
    levels: dict[str, float] | None = None,
):
    meta = {
        "description": description,
        "accepted": accepted,
        "effect": effect,
        "warning": warning,
        "choices": choices,
        "bounds": bounds,
        "levels": levels,
    }
    return field(default=default, metadata=meta)


LIFT_LEVELS = {"none": 0.0, "low": 0.15, "medium": 0.35, "high": 0.6}
ROLLOFF_LEVELS = {"none": 0.0, "low": 0.35, "moderate": 0.6, "strong": 0.85}
SATURATION_LEVELS = {"none": 0.0, "minimal": 0.03, "moderate": 0.12, "strong": 0.25}
DETAIL_LEVELS = {"none": 0.0, "minimal": 0.25, "low": 0.5, "moderate": 1.0, "high": 1.5}
ATMOSPHERIC_LEVELS = {"none": 0.0, "low": 0.1, "moderate": 0.25, "high": 0.5}
DENOISE_LEVELS = {"off": 0.0, "conservative": 0.35, "moderate": 0.6, "strong": 0.9}
TEXTURE_TOLERANCES = ("low", "medium", "high")
# Detail gain allowed in near-black regions, and the largest high-frequency energy ratio
# (processed / input) a shadow, sky or smooth block may reach before the failure rule falls back.
_SHADOW_DETAIL = {"low": 0.15, "medium": 0.4, "high": 0.8}
# Sharpening ignores detail below this many 8-bit codes (noise coring), so it does not amplify noise.
_CORING_CODES = {"low": 2.0, "medium": 1.0, "high": 0.5}
_TEXTURE_RATIO_LIMIT = {"low": 1.6, "medium": 2.5, "high": 4.0}
# Debanding acts fully where the local RMS residual from the mean is below this many codes, not at all
# above twice it: quantised gradients stay below, noisy or textured surfaces above.
_DEBAND_RESIDUAL_CODES = 0.5
# A halo pixel lies more than this many codes outside the luma range of the source samples it sits between.
_HALO_CODES = 2.0
_MOOD_CAP = 0.15


class _Section:
    """Validation shared by the parameter sections."""

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            _validate(f"{self.section}.{f.name}", f, getattr(self, f.name))

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


def _validate(name: str, f: dataclasses.Field, value: Any) -> None:
    meta = f.metadata
    levels, choices, bounds = meta["levels"], meta["choices"], meta["bounds"]
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if levels is not None:
        if isinstance(value, str):
            if value not in levels:
                raise ValueError(f"{name} must be one of {tuple(levels)} or a number, got {value!r}")
            return
        if not is_number:
            raise ValueError(f"{name} must be one of {tuple(levels)} or a number, got {value!r}")
    elif choices is not None:
        if value not in choices or type(value) is not type(choices[0]):
            raise ValueError(f"{name} must be one of {choices}, got {value!r}")
        return
    elif isinstance(f.default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false, got {value!r}")
        return
    elif isinstance(f.default, (int, float)) and not is_number:
        raise ValueError(f"{name} must be a number, got {value!r}")
    elif isinstance(f.default, int) and not isinstance(f.default, bool) and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if bounds is not None and not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name} must be between {bounds[0]} and {bounds[1]}, got {value!r}")


def _level(value: str | float, levels: dict[str, float]) -> float:
    return float(levels[value]) if isinstance(value, str) else float(value)


@dataclass(frozen=True)
class SuperResolutionParams(_Section):
    section = "super_resolution"
    enabled: bool = _param(
        True,
        "Run the SR slots before night processing (PRD 10).",
        "true | false",
        "false starts night processing from the Lanczos baseline.",
    )
    mode: str = _param(
        "balanced",
        "SR operating mode used as the night-processing input (PRD 10.2).",
        "conservative | balanced",
        "Until the Phase 2 backend lands both modes are Lanczos stand-ins and give identical images.",
        choices=("conservative", "balanced"),
    )
    tile_size: str | int = _param(
        "auto",
        "SR tile side in pixels (PRD 6.1, 10.4).",
        "auto, or 64 to 8192",
        "Recorded for the Phase 2 backend; the Lanczos stand-in does not tile.",
        levels={"auto": 0.0},
        bounds=(64, 8192),
    )
    tile_overlap: int = _param(
        64,
        "SR tile overlap in pixels (PRD 6.1).",
        "0 to 512",
        "Recorded for the Phase 2 backend; the Lanczos stand-in does not tile.",
        bounds=(0, 512),
    )


@dataclass(frozen=True)
class ShadowParams(_Section):
    section = "shadows"
    lift_strength: str | float = _param(
        "low",
        "Shadow lift (PRD 12.1, 19.2).",
        "none | low | medium | high, or 0 to 1",
        "Raises tones below lift_range with a toe curve; redistributes existing tones, recovers nothing.",
        warning="medium and high reveal compression noise and can look artificially bright.",
        levels=LIFT_LEVELS,
        bounds=(0.0, 1.0),
    )
    lift_range: float = _param(
        0.05,
        "Linear luminance below which the lift acts.",
        "0.005 to 0.2",
        "Larger values lift more of the image.",
        bounds=(0.005, 0.2),
    )
    preserve_black_level: bool = _param(
        True,
        "Keep black at black (PRD 4.2 stable black levels).",
        "true | false",
        "true: the lift curve passes through zero; false: the lift also raises the black floor.",
        warning="false gives milky blacks.",
    )
    black_point: float = _param(
        0.0,
        "Linear luminance mapped to black (PRD 19.2).",
        "0 to 0.02",
        "Deepens blacks by clipping values below it.",
        warning="Clips shadow detail; the failure rule falls back where it crushes near-black regions.",
        bounds=(0.0, 0.02),
    )
    color_preservation: float = _param(
        0.85,
        "Shadow colour preservation (PRD 12.3 shadow colour neutrality, 19.2).",
        "0 to 1",
        "1 keeps shadow chroma; lower values neutralise it towards grey.",
        bounds=(0.0, 1.0),
    )
    banding_reduction: bool = _param(
        True,
        "Smooth quantisation steps in smooth gradients and dark regions (PRD 12.1).",
        "true | false",
        "Replaces values by a local average only where they differ by less than two 8-bit codes.",
    )
    dither_amplitude: float = _param(
        0.5,
        "Triangular dither added to smooth gradients after debanding, in 8-bit code values.",
        "0 to 2",
        "Breaks residual contours; labelled simulated.",
        bounds=(0.0, 2.0),
    )
    invented_texture_tolerance: str = _param(
        "low",
        "How much texture processing may add in shadows, sky and smooth regions (PRD 12.1).",
        "low | medium | high",
        "Limits shadow sharpening and the texture threshold of the failure rule.",
        warning="high accepts plausible but unverified shadow texture.",
        choices=TEXTURE_TOLERANCES,
    )


@dataclass(frozen=True)
class HighlightParams(_Section):
    section = "highlights"
    protect: bool = _param(
        True,
        "Protect bright lights and specular reflections (PRD 12.2).",
        "true | false",
        "Excludes them from sharpening, local contrast and colour adjustments.",
    )
    rolloff_strength: str | float = _param(
        "moderate",
        "Highlight shoulder (PRD 19.2).",
        "none | low | moderate | strong, or 0 to 1",
        "Compresses tones above a knee smoothly; full-scale whites end slightly below full scale.",
        warning="A roll-off never recovers clipped radiance; it only reshapes the transition.",
        levels=ROLLOFF_LEVELS,
        bounds=(0.0, 1.0),
    )
    halo_suppression: bool = _param(
        True,
        "Suppress halos around bright edges (PRD 12.2).",
        "true | false",
        "Clamps every channel near lights to the range of the surrounding source pixels.",
    )
    halo_suppression_strength: float = _param(
        1.0,
        "Strength of the halo clamp.",
        "0 to 1",
        "1 clamps each channel near lights to the 3x3 source range plus one code; overshoot "
        "within that range (up to one source pixel's step) remains and the halo metric reports it.",
        bounds=(0.0, 1.0),
    )
    preserve_light_color: bool = _param(
        True,
        "Keep the hue and saturation of coloured lights (PRD 12.3).",
        "true | false",
        "true: the shoulder scales luminance; false: it acts per channel and bleaches coloured lights.",
    )


@dataclass(frozen=True)
class ColorParams(_Section):
    section = "color"
    preserve_source_mood: bool = _param(
        True,
        "Keep the source rendering as the colour reference (PRD 12.3).",
        "true | false",
        f"Caps warm/cool balance and saturation changes at +/-{_MOOD_CAP}.",
    )
    chromatic_adaptation: str = _param(
        "optional",
        "White-balance-like chromatic adaptation (PRD 12.3).",
        "optional | off",
        "optional: Bradford adaptation from color_temperature_k to 6504 K when they differ.",
        choices=("optional", "off"),
    )
    color_temperature_k: float = _param(
        6504.0,
        "Colour temperature treated as neutral (PRD 19.2).",
        "1700 to 25000",
        "Lower values cool the image, higher values warm it; 6504 leaves it unchanged.",
        bounds=(1700.0, 25000.0),
    )
    warm_cool_balance: float = _param(
        0.0,
        "Warm/cool lighting balance (PRD 12.3).",
        "-1 to 1",
        "Positive values strengthen warm lights and weaken cool ones; negative values the reverse.",
        bounds=(-1.0, 1.0),
    )
    saturation_adjustment: str | float = _param(
        "minimal",
        "Global saturation change (PRD 12.3).",
        "none | minimal | moderate | strong, or -1 to 1",
        "Scales chroma by 1 + value outside protected lights.",
        levels=SATURATION_LEVELS,
        bounds=(-1.0, 1.0),
    )
    saturation_luminance_coupling: float = _param(
        0.5,
        "Saturation versus luminance (PRD 12.3).",
        "0 to 1",
        "Reduces chroma progressively in the deepest shadows, where chroma noise lives.",
        bounds=(0.0, 1.0),
    )
    local_color_contrast: float = _param(
        0.0,
        "Local colour contrast in midtones (PRD 12.3).",
        "0 to 1",
        "Amplifies chroma differences against a local average.",
        bounds=(0.0, 1.0),
    )
    gamut_mapping: str = _param(
        "preserve_hue",
        "Gamut clipping management (PRD 12.3).",
        "preserve_hue | clip",
        "preserve_hue desaturates out-of-range colours towards their luminance; clip leaves it to the encoder.",
        choices=("preserve_hue", "clip"),
    )


@dataclass(frozen=True)
class TextureParams(_Section):
    section = "texture"
    architecture_detail_strength: str | float = _param(
        "moderate",
        "Architectural detail (PRD 19.2).",
        "none | minimal | low | moderate | high, or 0 to 2",
        "Scales sharpening in midtone and dark textured regions.",
        levels=DETAIL_LEVELS,
        bounds=(0.0, 2.0),
    )
    sky_detail_strength: str | float = _param(
        "minimal",
        "Sharpening allowed in sky and atmosphere.",
        "none | minimal | low | moderate | high, or 0 to 2",
        "Keeps skies smooth; higher values risk invented stars and noise.",
        levels=DETAIL_LEVELS,
        bounds=(0.0, 2.0),
    )
    atmospheric_detail_strength: str | float = _param(
        "low",
        "Mild atmospheric local contrast (PRD 12.5).",
        "none | low | moderate | high, or 0 to 1",
        "Large-radius local contrast on midtone structures, never on lights.",
        levels=ATMOSPHERIC_LEVELS,
        bounds=(0.0, 1.0),
    )
    atmospheric_softness: float = _param(
        0.0,
        "Atmospheric softness (PRD 19.2).",
        "0 to 1",
        "Softens local contrast in sky and atmosphere regions.",
        bounds=(0.0, 1.0),
    )
    sharpening_radius: float = _param(
        1.0,
        "Sharpening radius in source pixels (PRD 19.2).",
        "0.3 to 5",
        "Scaled by the upscale factor; larger radii emphasise coarser structure.",
        bounds=(0.3, 5.0),
    )
    sharpening_strength: float = _param(
        0.35,
        "Sharpening strength (PRD 19.2).",
        "0 to 2",
        "Unsharp-mask amount before the per-region detail factors.",
        warning="Values above 1 create halos; the failure rule then falls back.",
        bounds=(0.0, 2.0),
    )


@dataclass(frozen=True)
class NoiseParams(_Section):
    section = "noise"
    denoise_source: str | float = _param(
        "conservative",
        "Source denoising strength (PRD 12.4, 19.2).",
        "off | conservative | moderate | strong, or 0 to 1",
        "Mostly chroma noise in shadows, sky and smooth regions; luminance only lightly in near-black.",
        warning="strong flattens genuine low-light texture.",
        levels=DENOISE_LEVELS,
        bounds=(0.0, 1.0),
    )
    synthetic_grain: str = _param(
        "optional",
        "Synthetic grain (PRD 16).",
        "optional | off",
        "optional: added when grain_amount is above zero; labelled simulated.",
        choices=("optional", "off"),
    )
    grain_amount: float = _param(
        0.0,
        "Grain standard deviation in sRGB-encoded units at mid grey (PRD 19.2).",
        "0 to 0.05",
        "Adds luminance grain that is strongest in midtones; it recovers no sensor data.",
        bounds=(0.0, 0.05),
    )
    grain_size: float = _param(
        1.0,
        "Grain size in source pixels.",
        "0.5 to 3",
        "Larger grain is coarser.",
        bounds=(0.5, 3.0),
    )
    synthetic_sensor_noise: bool = _param(
        False,
        "Simulated sensor noise (PRD 14).",
        "false",
        "Belongs to Phase 5 sensor simulation and is not available here.",
        choices=(False,),
    )


@dataclass(frozen=True)
class OpticsParams(_Section):
    section = "optics"
    bloom: str = _param(
        "optional_subtle",
        "Light-source bloom (PRD 13.2, 19.2).",
        "optional_subtle | off",
        "optional_subtle: added when bloom_amount is above zero; labelled simulated.",
        choices=("optional_subtle", "off"),
    )
    bloom_amount: float = _param(
        0.0,
        "Bloom strength.",
        "0 to 0.2",
        "Adds a soft glow from lights.",
        warning="Above 0.05 bloom looks stylised (PRD 13.3).",
        bounds=(0.0, 0.2),
    )
    bloom_radius: float = _param(
        6.0,
        "Bloom radius in source pixels.",
        "1 to 32",
        "Larger glows spread further.",
        bounds=(1.0, 32.0),
    )
    chromatic_aberration: bool = _param(
        False, "Chromatic aberration (PRD 13).", "false", "Disabled by policy (PRD 19.1).", choices=(False,)
    )
    vignette: bool = _param(False, "Vignetting (PRD 13).", "false", "Disabled by policy (PRD 19.1).", choices=(False,))


@dataclass(frozen=True)
class FailureRuleParams(_Section):
    section = "failure_rule"
    enabled: bool = _param(
        True,
        "Fall back locally where processing develops artefacts (PRD 12.6).",
        "true | false",
        "Every fallback block is recorded in the report.",
        warning="false keeps halos, banding, crush or texture that the processing introduced.",
    )
    block_size: int = _param(
        32,
        "Side of the evaluation blocks in output pixels.",
        "16 to 256",
        "Smaller blocks localise fallbacks more finely.",
        bounds=(16, 256),
    )
    halo_increase: float = _param(
        0.02,
        "Largest allowed increase of the halo fraction near lights.",
        "0 to 1",
        "Lower values fall back more readily.",
        bounds=(0.0, 1.0),
    )
    banding_increase: float = _param(
        0.15,
        "Largest allowed increase of the banding score in smooth regions.",
        "0 to 1",
        "Lower values fall back more readily.",
        bounds=(0.0, 1.0),
    )
    black_crush_increase: float = _param(
        0.02,
        "Largest allowed increase of the crushed fraction in near-black regions.",
        "0 to 1",
        "Lower values fall back more readily.",
        bounds=(0.0, 1.0),
    )


_SECTIONS = {
    "super_resolution": SuperResolutionParams,
    "shadows": ShadowParams,
    "highlights": HighlightParams,
    "color": ColorParams,
    "texture": TextureParams,
    "noise": NoiseParams,
    "optics": OpticsParams,
    "failure_rule": FailureRuleParams,
}


@dataclass(frozen=True)
class NightParams:
    """All Layer 5B controls, grouped like the PRD 19.1 profile."""

    super_resolution: SuperResolutionParams = field(default_factory=SuperResolutionParams)
    shadows: ShadowParams = field(default_factory=ShadowParams)
    highlights: HighlightParams = field(default_factory=HighlightParams)
    color: ColorParams = field(default_factory=ColorParams)
    texture: TextureParams = field(default_factory=TextureParams)
    noise: NoiseParams = field(default_factory=NoiseParams)
    optics: OpticsParams = field(default_factory=OpticsParams)
    failure_rule: FailureRuleParams = field(default_factory=FailureRuleParams)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> NightParams:
        values = values or {}
        unknown = sorted(set(values) - set(_SECTIONS))
        if unknown:
            raise ValueError(f"unknown night sections: {', '.join(unknown)}")
        sections = {}
        for name, section_cls in _SECTIONS.items():
            given = values.get(name) or {}
            if not isinstance(given, dict):
                raise ValueError(f"night section {name} must be a mapping")
            known = {f.name for f in dataclasses.fields(section_cls)}
            bad = sorted(set(given) - known)
            if bad:
                raise ValueError(f"unknown keys in night section {name}: {', '.join(bad)}")
            sections[name] = section_cls(**given)
        return cls(**sections)

    def to_dict(self) -> dict:
        return {name: getattr(self, name).to_dict() for name in _SECTIONS}

    def replace(self, **sections: dict[str, Any]) -> NightParams:
        """Copy with some keys changed, e.g. ``replace(shadows={"lift_strength": "none"})``."""
        data = self.to_dict()
        for name, changes in sections.items():
            if name not in data:
                raise ValueError(f"unknown night section {name}")
            data[name].update(changes)
        return NightParams.from_dict(data)

    def conservative(self) -> NightParams:
        """The more conservative local result of the PRD 12.6 failure rule.

        Keeps the global colour and highlight treatment (so a fallback block
        matches its surroundings in colour) and debanding, and drops the lift,
        black point and every detail-adding operation.
        """
        return self.replace(
            shadows={"lift_strength": 0.0, "black_point": 0.0},
            highlights={"halo_suppression": True, "halo_suppression_strength": 1.0},
            color={"local_color_contrast": 0.0},
            texture={
                "sharpening_strength": 0.0,
                "atmospheric_detail_strength": 0.0,
                "atmospheric_softness": 0.0,
            },
        )


def parameter_table(params: NightParams | None = None) -> list[dict]:
    """Rows of (parameter, default, current, accepted, description, effect, warning) for display."""
    params = params or NightParams()
    defaults = NightParams()
    rows = []
    for name, section_cls in _SECTIONS.items():
        for f in dataclasses.fields(section_cls):
            rows.append(
                {
                    "parameter": f"{name}.{f.name}",
                    "default": getattr(getattr(defaults, name), f.name),
                    "current": getattr(getattr(params, name), f.name),
                    "accepted": f.metadata["accepted"],
                    "description": f.metadata["description"],
                    "effect": f.metadata["effect"],
                    "warning": f.metadata["warning"],
                }
            )
    return rows


# Experiment-level defaults of the night profile; the file values override them.
NIGHT_PROFILE_DEFAULTS = {"profile": "nighttime_scenery", "input_category": "night"}


@dataclass(frozen=True)
class NightConfig:
    """A validated PRD 19.1 profile: the shared experiment settings plus the Layer 5B controls."""

    experiment: lab_config.ExperimentConfig = field(
        default_factory=lambda: lab_config.from_dict(NIGHT_PROFILE_DEFAULTS)
    )
    night: NightParams = field(default_factory=NightParams)

    def to_dict(self) -> dict:
        return {**self.experiment.to_dict(), **self.night.to_dict()}


def load_night_config(path: str | Path | None = None, **overrides: Any) -> NightConfig:
    """Load a night profile such as ``configs/night_default.yaml``.

    Flat keys and the ``export`` block are validated by
    :func:`config.load_profile`, the nested sections by :class:`NightParams`.
    Overrides are flat experiment keys or section mappings, e.g.
    ``load_night_config(path, runtime_preset="quick", shadows={"lift_strength": "none"})``.
    """
    experiment, sections = lab_config.load_profile(path, tuple(_SECTIONS), NIGHT_PROFILE_DEFAULTS, **overrides)
    return NightConfig(experiment, NightParams.from_dict(sections))


# --------------------------------------------------------------------------- upscaler interface


@dataclass(frozen=True)
class UpscaleResult:
    image: np.ndarray
    method: str
    label: str
    provenance: str
    classification: str
    stand_in_for: str | None = None

    def describe(self) -> dict:
        return {
            "method": self.method,
            "label": self.label,
            "provenance": self.provenance,
            "classification": self.classification,
            "stand_in_for": self.stand_in_for,
        }


class Upscaler(Protocol):
    """Anything that resizes a linear float32 (H, W, 3) image to (height, width, 3).

    A Phase 2 SR backend plugs in by implementing this call and returning
    ``provenance=SYNTHESIZED`` and ``classification=INFERENCE``.
    """

    name: str

    def __call__(self, linear: np.ndarray, width: int, height: int) -> UpscaleResult: ...


@dataclass(frozen=True)
class BaselineUpscaler:
    """A classical baseline from :mod:`baselines`, optionally standing in for an SR slot."""

    name: str
    method: str = "lanczos"
    stand_in_for: str | None = None

    def __call__(self, linear: np.ndarray, width: int, height: int) -> UpscaleResult:
        image = baselines.resample(linear, width, height, self.method)
        if self.stand_in_for:
            label = f"{self.method.capitalize()} stand-in for {self.stand_in_for} (Phase 2 SR backend not merged)"
        else:
            label = f"{self.method.capitalize()} baseline"
        return UpscaleResult(
            image, baselines.METHODS[self.method].description, label, ESTIMATED, SPATIAL, self.stand_in_for
        )


def sr_slot(mode: str) -> Upscaler:
    """The upscaler filling the PRD 19.3 SR comparison slot for ``mode``.

    Phase 2's SR backend is not merged, so both slots are Lanczos stand-ins.
    """
    if mode not in ("conservative", "balanced"):
        raise ValueError(f"unknown SR mode {mode!r}")
    return BaselineUpscaler(f"{mode}_sr", "lanczos", f"{mode.capitalize()} SR")


# --------------------------------------------------------------------------- region segmentation

REGION_NAMES = (
    "near_black",
    "low_luminance_textured",
    "midtone",
    "bright_light",
    "clipped_highlight",
    "smooth_gradient",
    "specular_reflection",
    "sky_atmosphere",
)
REGION_COLORS = {  # visualisation palette, sRGB 8-bit
    "near_black": (40, 40, 160),
    "low_luminance_textured": (60, 150, 200),
    "midtone": (120, 120, 120),
    "bright_light": (255, 190, 40),
    "clipped_highlight": (255, 40, 40),
    "smooth_gradient": (80, 200, 100),
    "specular_reflection": (230, 90, 230),
    "sky_atmosphere": (150, 120, 255),
}
_LIGHTS = ("bright_light", "clipped_highlight")


@dataclass(frozen=True)
class RegionThresholds:
    """Segmentation thresholds; luminance values are sRGB-encoded luma in [0, 1]."""

    near_black: float = 0.06
    low_luminance: float = 0.25
    bright_light: float = 0.75
    light_tophat: float = 0.2
    specular_luminance: float = 0.25
    specular_tophat: float = 0.08
    specular_streak_width: float = 0.1  # fraction of source height; reflections are narrower than this
    specular_streak_length: float = 0.2  # fraction of source height; reflections run at least this far down
    clipped_linear: float = 0.996
    smooth_texture: float = 0.004
    smooth_edge: float = 0.03
    sky_texture: float = 0.004
    sky_max_luminance: float = 0.6
    sky_max_row_fraction: float = 0.75
    halo_ring_px: int = 3

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True, eq=False)
class RegionMap:
    """Exclusive per-pixel region labels at source resolution (PRD 8.3).

    ``labels[y, x]`` indexes :data:`REGION_NAMES`. Processing uses feathered
    versions of the labels (:meth:`soft_strip`); :meth:`override` returns a
    new map with pixels re-labelled by hand, and records the change.
    """

    labels: np.ndarray
    thresholds: RegionThresholds = field(default_factory=RegionThresholds)
    overrides: tuple = ()

    @property
    def shape(self) -> tuple[int, int]:
        return self.labels.shape

    def mask(self, *names: str) -> np.ndarray:
        indices = [REGION_NAMES.index(n) for n in names]
        return np.isin(self.labels, indices)

    def fractions(self) -> dict[str, float]:
        counts = np.bincount(self.labels.ravel(), minlength=len(REGION_NAMES))
        return {name: float(c / self.labels.size) for name, c in zip(REGION_NAMES, counts, strict=True)}

    def override(self, name: str, where, note: str = "") -> RegionMap:
        """Assign ``name`` to the pixels in ``where``: a boolean mask or a list of [x, y, width, height] boxes."""
        if name not in REGION_NAMES:
            raise ValueError(f"unknown region {name!r}; expected one of {REGION_NAMES}")
        if isinstance(where, np.ndarray) and where.dtype == bool:
            if where.shape != self.shape:
                raise ValueError(f"override mask shape {where.shape} does not match {self.shape}")
            selected, description = where, f"mask of {int(where.sum())} pixels"
        else:
            selected = np.zeros(self.shape, dtype=bool)
            for box in where:
                x, y, w, h = (int(v) for v in box)
                if w < 1 or h < 1 or x < 0 or y < 0 or x + w > self.shape[1] or y + h > self.shape[0]:
                    raise ValueError(f"override box {list(box)} outside the {self.shape[1]}x{self.shape[0]} source")
                selected[y : y + h, x : x + w] = True
            description = f"boxes {[list(map(int, b)) for b in where]}"
        labels = self.labels.copy()
        labels[selected] = REGION_NAMES.index(name)
        record = {"region": name, "where": description, "pixels": int(selected.sum()), "note": note}
        return RegionMap(labels, self.thresholds, (*self.overrides, record))

    @cached_property
    def _soft(self) -> dict[str, np.ndarray]:
        """Feathered label indicators and the halo ring, uint8 at source resolution."""
        soft = {}
        for index, name in enumerate(REGION_NAMES):
            indicator = (self.labels == index).astype(np.float32)
            soft[name] = _to_u8(_blur(indicator, 1.0))
        lights = self.mask(*_LIGHTS, "specular_reflection").astype(np.uint8)
        size = 2 * self.thresholds.halo_ring_px + 1
        ring = cv2.dilate(lights, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))) & (1 - lights)
        soft["ring"] = _to_u8(_blur(ring.astype(np.float32), 1.0))
        return soft

    def soft_strip(self, out_width: int, out_height: int, y0: int, y1: int) -> dict[str, np.ndarray]:
        """Feathered region weights for output rows [y0, y1) of an (out_height, out_width) image."""
        return {
            name: _warp_rows(plane, out_width, out_height, y0, y1).astype(np.float32) * np.float32(1 / 255)
            for name, plane in self._soft.items()
        }

    def visualize(self, source_linear: np.ndarray | None = None, opacity: float = 0.6) -> np.ndarray:
        """uint8 RGB label map, blended over the source luminance when it is given."""
        palette = np.array([REGION_COLORS[n] for n in REGION_NAMES], dtype=np.float32)
        colours = palette[self.labels]
        if source_linear is not None:
            grey = luma(cm.linear_to_srgb(np.clip(source_linear, 0, 1)))[..., None] * 255
            colours = opacity * colours + (1 - opacity) * grey
        return np.clip(colours, 0, 255).round().astype(np.uint8)

    def to_dict(self) -> dict:
        return {
            "regions": list(REGION_NAMES),
            "fractions": self.fractions(),
            "thresholds": self.thresholds.to_dict(),
            "overrides": list(self.overrides),
        }


def segment_regions(source_linear: np.ndarray, thresholds: RegionThresholds | None = None) -> RegionMap:
    """Label the source into the PRD 8.3 night regions (deterministic heuristics, no models).

    Specular reflections are compact bright glints, or bright streaks that are
    horizontally narrow, vertically elongated and lie below a light source,
    as reflections on wet ground or water do. Sky is the low-texture top of
    each column: a pixel is sky only if every pixel above it is sky-like or a
    light such as a star, so smooth facades and roads below a skyline never
    join the sky.
    """
    t = thresholds or RegionThresholds()
    encoded = cm.linear_to_srgb(np.clip(source_linear, 0, 1))
    y = luma(encoded)
    height, width = y.shape

    clipped = source_linear.max(axis=2) >= t.clipped_linear
    clipped = cv2.dilate(clipped.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    tophat15 = cv2.morphologyEx(y, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bright = (y >= t.bright_light) | ((y >= 0.45) & (tophat15 >= t.light_tophat))
    bright = cv2.dilate(bright.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~clipped
    sources = clipped | bright
    compact = np.minimum(  # bright only on a small spot: narrower than 9 px both across and along
        cv2.morphologyEx(y, cv2.MORPH_TOPHAT, np.ones((1, 9), np.uint8)),
        cv2.morphologyEx(y, cv2.MORPH_TOPHAT, np.ones((9, 1), np.uint8)),
    )
    glints = (y >= t.specular_luminance) & (compact >= t.specular_tophat) & ~sources
    across = 2 * int(t.specular_streak_width * height / 2) + 1
    along = 2 * int(t.specular_streak_length * height / 2) + 1
    narrow = cv2.morphologyEx(y, cv2.MORPH_TOPHAT, np.ones((1, across), np.uint8))
    streaks = ((y >= t.specular_luminance) & (narrow >= t.specular_tophat)).astype(np.uint8)
    streaks = cv2.morphologyEx(streaks, cv2.MORPH_OPEN, np.ones((along, 1), np.uint8)).astype(bool)
    # Only a light outside the streak can cast it; streak pixels below light-source level are reflection.
    anchors = cv2.dilate((sources & ~streaks).astype(np.uint8), np.ones((1, across), np.uint8)).astype(bool)
    reflected = streaks & np.logical_or.accumulate(anchors, axis=0) & ~clipped & (y < t.bright_light)
    specular = glints | reflected
    bright &= ~specular
    lights = sources | specular

    detail = y - _blur(y, 1.5)
    texture = np.sqrt(_blur(detail * detail, 2.5))
    smoothed = _blur(y, 1.0)
    gradient = np.hypot(cv2.Sobel(smoothed, cv2.CV_32F, 1, 0), cv2.Sobel(smoothed, cv2.CV_32F, 0, 1)) / 8
    edges = ndimage.maximum_filter(gradient, size=5)
    smooth = (texture < t.smooth_texture) & (edges < t.smooth_edge) & ~lights

    # Small lights such as stars are replaced by their surroundings before judging the sky's texture,
    # and lights do not end the sky below them.
    unlit = cv2.dilate(lights.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    unlit = np.where(
        unlit, cv2.morphologyEx(y, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))), y
    )
    unlit_detail = unlit - _blur(unlit, 1.5)
    sky_like = (np.sqrt(_blur(unlit_detail * unlit_detail, 2.5)) < t.sky_texture) & (y < t.sky_max_luminance) & ~lights
    sky_like[int(height * t.sky_max_row_fraction) :] = False
    sky = np.logical_and.accumulate(sky_like | lights, axis=0) & sky_like

    labels = np.full((height, width), REGION_NAMES.index("midtone"), dtype=np.uint8)
    # Assigned from lowest to highest priority, so later assignments win.
    labels[y < t.low_luminance] = REGION_NAMES.index("low_luminance_textured")
    labels[smooth & (y >= t.near_black)] = REGION_NAMES.index("smooth_gradient")
    labels[y < t.near_black] = REGION_NAMES.index("near_black")
    labels[sky] = REGION_NAMES.index("sky_atmosphere")
    labels[specular] = REGION_NAMES.index("specular_reflection")
    labels[bright] = REGION_NAMES.index("bright_light")
    labels[clipped] = REGION_NAMES.index("clipped_highlight")
    return RegionMap(labels, t)


# --------------------------------------------------------------------------- tone and colour (pointwise)


def planckian_xy(kelvin: float) -> tuple[float, float]:
    """CIE 1931 xy of a Planckian radiator (Kim et al. 2002 cubic fit, 1667 to 25000 K)."""
    t = float(kelvin)
    if t <= 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


def adaptation_matrix(kelvin: float) -> np.ndarray:
    """Linear Rec.709 matrix adapting a ``kelvin`` white to the 6504 K Planckian white (Bradford)."""
    to_xyz = cm.rgb_to_xyz_matrix(cm.SRGB_PRIMARIES, cm.D65_XY)
    adapt = cm.bradford_adaptation(cm.xy_to_xyz(planckian_xy(kelvin)), cm.xy_to_xyz(planckian_xy(6504.0)))
    return (cm.XYZ_TO_LINEAR_SRGB @ adapt @ to_xyz).astype(np.float32)


@dataclass(frozen=True)
class ToneCurve:
    """Luminance curve: black point, shadow lift (toe) and highlight shoulder."""

    black_point: float = 0.0
    lift: float = 0.0
    lift_range: float = 0.05
    preserve_black_level: bool = True
    rolloff: float = 0.0

    @property
    def knee(self) -> float:
        return 1.0 - 0.4 * self.rolloff

    @property
    def pedestal(self) -> float:
        return 0.0 if self.preserve_black_level else 0.1 * self.lift * self.lift_range

    def toe(self, y: np.ndarray) -> np.ndarray:
        """Black point and lift; passes through zero, identity above lift_range."""
        bp, s = np.float32(self.black_point), np.float32(self.lift_range)
        y = np.maximum(y - bp, 0) / (1 - bp) if self.black_point > 0 else np.maximum(y, 0)
        if self.lift > 0:
            u = np.clip(1 - y / s, 0, 1)
            y = y + np.float32(2 * self.lift) * y * u * u
        return y.astype(np.float32, copy=False)

    def shoulder(self, y: np.ndarray) -> np.ndarray:
        """Smooth roll-off above the knee (continuous slope 1 at the knee)."""
        if self.rolloff <= 0:
            return y
        k = np.float32(self.knee)
        above = y > k
        out = y.copy()
        out[above] = k + (1 - k) * np.tanh((y[above] - k) / (1 - k))
        return out

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return self.shoulder(self.toe(y)) + np.float32(self.pedestal)

    def is_identity(self) -> bool:
        return self.black_point == 0 and self.lift == 0 and self.rolloff == 0


def tone_curve(params: NightParams) -> ToneCurve:
    s, h = params.shadows, params.highlights
    return ToneCurve(
        black_point=s.black_point,
        lift=_level(s.lift_strength, LIFT_LEVELS),
        lift_range=s.lift_range,
        preserve_black_level=s.preserve_black_level,
        rolloff=_level(h.rolloff_strength, ROLLOFF_LEVELS),
    )


def _global_ops(linear: np.ndarray, params: NightParams) -> np.ndarray:
    """Chromatic adaptation and the tone curve; pointwise, so identical in every strip."""
    out = linear
    c = params.color
    if c.chromatic_adaptation == "optional" and c.color_temperature_k != 6504.0:
        out = out @ adaptation_matrix(c.color_temperature_k).T
    curve = tone_curve(params)
    if curve.is_identity():
        return np.array(out, dtype=np.float32, copy=True)
    y = out @ LUMA_WEIGHTS
    slope = np.float32((1 + 2 * curve.lift) / (1 - curve.black_point))
    if params.highlights.preserve_light_color:
        mapped = curve.shoulder(curve.toe(y))
        out = out * np.where(y > 1e-8, mapped / np.maximum(y, 1e-8), slope)[..., None]
    else:
        out = curve.shoulder(out * np.where(y > 1e-8, curve.toe(y) / np.maximum(y, 1e-8), slope)[..., None])
    if curve.pedestal:
        out = out + np.float32(curve.pedestal)
    return out.astype(np.float32, copy=False)


def gamut_map(linear: np.ndarray) -> np.ndarray:
    """Bring out-of-range colours into [0, 1] by desaturating towards their luminance (hue preserved)."""
    out = linear.copy()
    y = out @ LUMA_WEIGHTS
    high = out.max(axis=2)
    over = high > 1
    if over.any():
        yo = np.clip(y[over], 0, 1)[:, None]
        t = np.clip((1 - yo) / np.maximum(high[over][:, None] - yo, 1e-8), 0, 1)
        out[over] = yo + t * (out[over] - yo)
    under = out.min(axis=2) < 0
    if under.any():
        yu = np.clip(y[under], 0, 1)[:, None]
        lo = out[under].min(axis=1, keepdims=True)
        t = np.clip(yu / np.maximum(yu - lo, 1e-8), 0, 1)
        out[under] = yu + t * (out[under] - yu)
    return np.clip(out, 0, 1, out=out)


# --------------------------------------------------------------------------- strip machinery


def _blur(plane: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur truncated at 3 sigma (the strip margin assumes that support)."""
    if sigma <= 0:
        return plane
    radius = max(1, math.ceil(3 * sigma))
    return cv2.GaussianBlur(plane, (2 * radius + 1, 2 * radius + 1), sigma, borderType=cv2.BORDER_REFLECT_101)


def _to_u8(plane: np.ndarray) -> np.ndarray:
    return np.clip(plane * 255 + 0.5, 0, 255).astype(np.uint8)


def _warp_rows(plane: np.ndarray, out_width: int, out_height: int, y0: int, y1: int) -> np.ndarray:
    """Rows [y0, y1) of a bilinear resize of ``plane`` to (out_height, out_width), without the full resize.

    Sample positions depend only on absolute output coordinates, so every strip
    reproduces exactly the rows of the full resize.
    """
    sx = plane.shape[1] / out_width
    sy = plane.shape[0] / out_height
    ys = (np.arange(y0, y1, dtype=np.float64) + 0.5) * sy - 0.5
    xs = (np.arange(out_width, dtype=np.float64) + 0.5) * sx - 0.5
    return _bilinear(plane, ys, xs)


def _neighbours(coords: np.ndarray, size: int, origin: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the two samples around each coordinate (edge replicated); sample ``origin`` is index 0."""
    index = np.floor(coords).astype(np.int64) - origin
    return np.clip(index, 0, size - 1), np.clip(index + 1, 0, size - 1)


def _bilinear(plane: np.ndarray, ys: np.ndarray, xs: np.ndarray, y_origin: int = 0) -> np.ndarray:
    """float32 samples of ``plane`` at rows ``ys`` and columns ``xs`` (edge replicated).

    ``ys`` are absolute coordinates; row ``y_origin`` of the absolute grid is ``plane[0]``.
    """

    def axis(coords: np.ndarray, size: int, origin: int):
        base = np.floor(coords)
        index = base.astype(np.int64) - origin
        return np.clip(index, 0, size - 1), np.clip(index + 1, 0, size - 1), (coords - base).astype(np.float32)

    r0, r1, fy = axis(ys, plane.shape[0], y_origin)
    c0, c1, fx = axis(xs, plane.shape[1], 0)
    extra = (1,) * (plane.ndim - 2)
    fy, fx = fy.reshape(-1, 1, *extra), fx.reshape(-1, *extra)
    rows = plane[r0].astype(np.float32) * (1 - fy) + plane[r1].astype(np.float32) * fy
    return rows[:, c0] * (1 - fx) + rows[:, c1] * fx


@dataclass
class _Strip:
    """Per-strip context: padded rows [p0, p1) around the core rows [y0, y1)."""

    y0: int
    y1: int
    p0: int
    p1: int
    scale: float
    weights: dict[str, np.ndarray]
    source: _Envelope
    size: tuple[int, int]  # output (width, height)
    _envelopes: dict = field(default_factory=dict)

    @property
    def core(self) -> slice:
        return slice(self.y0 - self.p0, self.y1 - self.p0)

    def w(self, *names: str) -> np.ndarray:
        total = self.weights[names[0]].copy()
        for name in names[1:]:
            total += self.weights[name]
        return total

    def hard(self, *names: str) -> np.ndarray:
        return self.w(*names) > 0.5

    def _source(self, params: NightParams | None) -> tuple[np.ndarray, ...]:
        key = None if params is None else id(params)
        if key not in self._envelopes:
            self._envelopes[key] = self.source.strip(*self.size, self.p0, self.p1, params)
        return self._envelopes[key]

    def envelope(self, params: NightParams | None) -> tuple[np.ndarray, np.ndarray]:
        """Encoded per-channel source range around each padded pixel, after ``params``' pointwise operations."""
        low, high = self._source(params)[:2]
        return low, high

    def footprint(self, params: NightParams | None) -> tuple[np.ndarray, np.ndarray]:
        """Encoded luma range of the 2x2 source samples around each padded pixel, after ``params``' operations."""
        low, high = self._source(params)[2:]
        return low, high


class _Envelope:
    """Source ranges: the reference of the halo clamp and, separately, of the halo check.

    The clamp uses per-channel ranges of 3x3 source neighbourhoods. The halo
    check uses the luma range of the 2x2 source samples an output pixel lies
    between, which every interpolation without overshoot stays inside. That
    range is tighter than the clamp's (which reaches one more source pixel out,
    plus one code), so a dark ring the clamp lets through still counts.
    Both are computed per strip from the source rows the strip needs, after the
    same pointwise operations (colour adaptation, tone curve) as the image, so
    they compare like with like.
    """

    def __init__(self, source_linear: np.ndarray):
        self.linear = source_linear

    def strip(
        self, out_width: int, out_height: int, p0: int, p1: int, params: NightParams | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        height, width = self.linear.shape[:2]
        ys = (np.arange(p0, p1, dtype=np.float64) + 0.5) * (height / out_height) - 0.5
        xs = (np.arange(out_width, dtype=np.float64) + 0.5) * (width / out_width) - 0.5
        a = max(0, math.floor(ys[0]) - 1)
        b = min(height, math.floor(ys[-1]) + 3)
        part = self.linear[a:b] if params is None else _global_ops(self.linear[a:b], params)
        encoded = cm.linear_to_srgb(np.clip(part, 0, 1))
        low = ndimage.minimum_filter(encoded, size=(3, 3, 1), mode="nearest")
        high = ndimage.maximum_filter(encoded, size=(3, 3, 1), mode="nearest")
        y = encoded @ LUMA_WEIGHTS
        r0, r1 = _neighbours(ys, y.shape[0], a)
        c0, c1 = _neighbours(xs, y.shape[1])
        top, bottom = y[r0], y[r1]
        a00, a01, a10, a11 = top[:, c0], top[:, c1], bottom[:, c0], bottom[:, c1]
        return (
            _bilinear(low, ys, xs, y_origin=a),
            _bilinear(high, ys, xs, y_origin=a),
            np.minimum(np.minimum(a00, a01), np.minimum(a10, a11)),
            np.maximum(np.maximum(a00, a01), np.maximum(a10, a11)),
        )


def _filter_margin(params: NightParams, scale: float) -> int:
    t, o = params.texture, params.optics
    sigmas = [1.5 * scale * 1.5, 6.0 * scale, 4.0 * scale, t.sharpening_radius * scale, 2.0]
    if o.bloom != "off" and o.bloom_amount > 0:
        sigmas.append(o.bloom_radius * scale)
    return math.ceil(3 * max(sigmas)) + 2


def _strips(height: int, strip_rows: int, margin: int):
    for y0 in range(0, height, strip_rows):
        y1 = min(height, y0 + strip_rows)
        yield y0, y1, max(0, y0 - margin), min(height, y1 + margin)


# --------------------------------------------------------------------------- night processing


def _process(linear: np.ndarray, strip: _Strip, params: NightParams, regions_scale: float) -> np.ndarray:
    """All deterministic Layer 5B operations on one padded strip (no dither, grain or bloom)."""
    s = regions_scale
    sh, hi, co, tx, no = params.shadows, params.highlights, params.color, params.texture, params.noise
    out = _global_ops(linear, params)
    encoded = cm.linear_to_srgb(out)
    y = np.ascontiguousarray(encoded @ LUMA_WEIGHTS)
    chroma = encoded - y[..., None]

    light = strip.w(*_LIGHTS)
    specular = strip.weights["specular_reflection"]
    ring = strip.weights["ring"]
    protected = np.clip(light + specular, 0, 1) if hi.protect else np.zeros_like(light)
    shadow = strip.w("near_black", "low_luminance_textured")
    near_black = strip.weights["near_black"]
    smooth = strip.w("smooth_gradient", "sky_atmosphere")
    sky = strip.weights["sky_atmosphere"]
    mid = strip.weights["midtone"]
    textured = strip.weights["low_luminance_textured"]

    # Chroma noise (PRD 12.4): conservative chroma blur, strongest in shadows and smooth areas.
    strength = _level(no.denoise_source, DENOISE_LEVELS)
    if strength > 0:
        weight = (strength * np.clip(shadow + smooth + 0.5 * mid, 0, 1) * (1 - protected))[..., None]
        blurred = np.stack([_blur(chroma[..., c], 1.5 * s * (0.5 + strength)) for c in range(3)], axis=-1)
        chroma += weight * (blurred - chroma)
        y += 0.3 * strength * near_black * (_blur(y, 0.8 * s) - y)

    # Banding reduction (PRD 12.1): local average accepted only within two 8-bit codes.
    if sh.banding_reduction:
        average = _blur(y, 6.0 * s)
        difference = average - y
        accept = np.clip(2 - np.abs(difference) / _CODE, 0, 1)
        # Quantisation steps leave residuals of at most half a code (RMS about 0.3); noise and texture
        # leave larger ones. Snapping those pixels to the mean would flatten texture into plateaus.
        residual = np.sqrt(_blur(difference * difference, 3.0 * s))
        accept *= np.clip(2 - residual / (_DEBAND_RESIDUAL_CODES * _CODE), 0, 1)
        y += np.clip(smooth + near_black, 0, 1) * (1 - protected) * (1 - ring) * accept * difference

    # Detail: sharpening by region (PRD 19.2), limited by the invented-texture tolerance.
    amount = tx.sharpening_strength
    if amount > 0:
        arch = _level(tx.architecture_detail_strength, DETAIL_LEVELS)
        tolerance = _SHADOW_DETAIL[sh.invented_texture_tolerance]
        gain = (
            mid * arch
            + textured * arch * min(1.0, 2 * tolerance)
            + near_black * arch * tolerance
            + sky * _level(tx.sky_detail_strength, DETAIL_LEVELS) * 0.5
        ) * (1 - protected)
        if hi.halo_suppression:
            gain *= 1 - ring
        detail = y - _blur(y, tx.sharpening_radius * s)
        coring = _CORING_CODES[sh.invented_texture_tolerance] * _CODE
        y += amount * gain * detail * (1 - np.exp(-np.square(detail / coring)))

    # Atmospheric local contrast and softness (PRD 12.5).
    atmospheric = _level(tx.atmospheric_detail_strength, ATMOSPHERIC_LEVELS)
    if atmospheric > 0 or tx.atmospheric_softness > 0:
        large = _blur(y, 6.0 * s)
        tolerance = _SHADOW_DETAIL[sh.invented_texture_tolerance]
        y += atmospheric * (mid + tolerance * textured) * (1 - protected) * (1 - ring) * (y - large)
        y += 0.5 * tx.atmospheric_softness * sky * (large - y)

    # Colour (PRD 12.3).
    factor = np.ones_like(y)
    saturation = _level(co.saturation_adjustment, SATURATION_LEVELS)
    balance = co.warm_cool_balance
    if co.preserve_source_mood:
        saturation = float(np.clip(saturation, -_MOOD_CAP, _MOOD_CAP))
        balance = float(np.clip(balance, -_MOOD_CAP, _MOOD_CAP))
    factor *= 1 + saturation
    if balance:
        warmth = np.tanh((encoded[..., 0] - encoded[..., 2]) * 10)
        factor *= 1 + 0.3 * balance * warmth
    if co.saturation_luminance_coupling > 0:
        depth = np.clip(1 - y / 0.25, 0, 1)
        factor *= 1 - 0.5 * co.saturation_luminance_coupling * depth * depth
    factor *= 1 - (1 - sh.color_preservation) * shadow
    if hi.preserve_light_color:
        factor = factor + protected * (1 - factor)
    chroma *= factor[..., None]
    if co.local_color_contrast > 0:
        local = np.stack([_blur(chroma[..., c], 6.0 * s) for c in range(3)], axis=-1)
        chroma += (co.local_color_contrast * mid * (1 - protected))[..., None] * (chroma - local)

    encoded = y[..., None] + chroma
    # Halo suppression (PRD 12.2): clamp each channel near lights to the source range around the pixel,
    # which removes most ringing and sharpening overshoot without leaving coloured fringes.
    if hi.halo_suppression and hi.halo_suppression_strength > 0:
        low, high = strip.envelope(params)
        clamped = np.clip(encoded, low - _CODE, high + _CODE)
        # Full strength inside the ring, feathered only at its outer edge.
        encoded += (hi.halo_suppression_strength * np.clip(2 * ring, 0, 1))[..., None] * (clamped - encoded)

    out = cm.srgb_to_linear(encoded)
    return gamut_map(out) if co.gamut_mapping == "preserve_hue" else out


def _simulated(linear: np.ndarray, strip: _Strip, params: NightParams, seed: int) -> tuple[np.ndarray, list[str]]:
    """Dither, bloom and grain on the core rows; all labelled simulated."""
    core = strip.core
    applied = []
    sh, no, op = params.shadows, params.noise, params.optics
    out = linear
    if op.bloom != "off" and op.bloom_amount > 0:
        source = linear * strip.w(*_LIGHTS)[..., None]
        glow = np.stack([_blur(source[..., c], op.bloom_radius * strip.scale) for c in range(3)], axis=-1)
        out = out + np.float32(op.bloom_amount) * glow
        applied.append("bloom")
    out = out[core]
    rng = np.random.default_rng([seed, 5, strip.y0])
    shape = out.shape[:2]
    encoded = None
    if sh.banding_reduction and sh.dither_amplitude > 0:
        weight = np.clip(strip.w("smooth_gradient", "sky_atmosphere", "near_black")[core], 0, 1)
        weight *= 1 - np.clip(strip.w(*_LIGHTS, "ring")[core], 0, 1)
        noise = (rng.random(shape, dtype=np.float32) - rng.random(shape, dtype=np.float32)) * np.float32(
            sh.dither_amplitude * _CODE
        )
        encoded = cm.linear_to_srgb(np.clip(out, 0, None))
        encoded += (weight * noise)[..., None]
        applied.append("dither")
    if no.synthetic_grain != "off" and no.grain_amount > 0:
        if encoded is None:
            encoded = cm.linear_to_srgb(np.clip(out, 0, None))
        grain = rng.standard_normal(shape, dtype=np.float32)
        sigma = no.grain_size * strip.scale * 0.5
        if sigma > 0.5:
            grain = _blur(grain, sigma) * np.float32(sigma * 2 * math.sqrt(math.pi))
        y = encoded @ LUMA_WEIGHTS
        envelope = np.clip(4 * y * (1 - y), 0, 1)  # strongest at mid grey, none at black or white
        encoded += (np.float32(no.grain_amount) * envelope * grain)[..., None]
        applied.append("grain")
    if encoded is not None:
        out = cm.srgb_to_linear(np.clip(encoded, 0, None))
    if params.color.gamut_mapping == "preserve_hue" and applied:
        out = gamut_map(out)
    return out.astype(np.float32, copy=False), applied


@dataclass
class NightResult:
    image: np.ndarray
    stages: list[dict]
    fallbacks: list[dict]
    fallback_fraction: float
    metrics_before: dict
    metrics_after: dict
    comparison: dict
    information: dict


def stage_records(params: NightParams, upscale: UpscaleResult | None = None) -> list[dict]:
    """The Layer 5B stages with provenance labels and PRD 5.2 classifications."""
    sh, hi, co, tx, no, op = (
        params.shadows,
        params.highlights,
        params.color,
        params.texture,
        params.noise,
        params.optics,
    )
    curve = tone_curve(params)

    def record(name, provenance, classification, enabled, parameters, note=""):
        return {
            "name": name,
            "provenance": provenance,
            "classification": classification,
            "enabled": bool(enabled),
            "parameters": parameters,
            "note": note,
        }

    records = [record("region_segmentation", PRESERVED, "Measurement (labels only, no image data)", True, {})]
    if upscale is not None:
        records.append(record("upscale", upscale.provenance, upscale.classification, True, upscale.describe()))
    records += [
        record(
            "chromatic_adaptation",
            ESTIMATED,
            TONAL,
            co.chromatic_adaptation == "optional" and co.color_temperature_k != 6504.0,
            {"color_temperature_k": co.color_temperature_k},
        ),
        record(
            "shadow_tone",
            ESTIMATED,
            TONAL,
            curve.lift > 0 or curve.black_point > 0,
            {"lift": curve.lift, "black_point": curve.black_point, "preserve_black_level": sh.preserve_black_level},
            "Redistributes existing tones; it does not recover shadow measurements.",
        ),
        record(
            "highlight_rolloff",
            ESTIMATED,
            TONAL,
            curve.rolloff > 0,
            {"knee": round(curve.knee, 4), "preserve_light_color": hi.preserve_light_color},
            "A roll-off of clipped highlights; clipped radiance is not recovered.",
        ),
        record(
            "chroma_noise_control",
            ESTIMATED,
            SPATIAL,
            _level(no.denoise_source, DENOISE_LEVELS) > 0,
            {"strength": _level(no.denoise_source, DENOISE_LEVELS)},
        ),
        record("banding_reduction", ESTIMATED, SPATIAL, sh.banding_reduction, {"max_change_codes": 2}),
        record(
            "region_sharpening",
            ESTIMATED,
            SPATIAL,
            tx.sharpening_strength > 0,
            {
                "strength": tx.sharpening_strength,
                "radius_source_px": tx.sharpening_radius,
                "invented_texture_tolerance": sh.invented_texture_tolerance,
            },
        ),
        record(
            "atmospheric_local_contrast",
            ESTIMATED,
            SPATIAL,
            _level(tx.atmospheric_detail_strength, ATMOSPHERIC_LEVELS) > 0 or tx.atmospheric_softness > 0,
            {
                "detail": _level(tx.atmospheric_detail_strength, ATMOSPHERIC_LEVELS),
                "softness": tx.atmospheric_softness,
            },
        ),
        record(
            "night_color",
            ESTIMATED,
            TONAL,
            True,
            {
                "saturation": _level(co.saturation_adjustment, SATURATION_LEVELS),
                "warm_cool_balance": co.warm_cool_balance,
                "shadow_color_preservation": sh.color_preservation,
                "gamut_mapping": co.gamut_mapping,
            },
        ),
        record(
            "halo_suppression",
            ESTIMATED,
            SPATIAL,
            hi.halo_suppression,
            {"strength": hi.halo_suppression_strength},
        ),
        record("failure_rule", ESTIMATED, SPATIAL, params.failure_rule.enabled, params.failure_rule.to_dict()),
        record(
            "dither",
            SIMULATED,
            SIMULATION,
            sh.banding_reduction and sh.dither_amplitude > 0,
            {"amplitude_codes": sh.dither_amplitude},
        ),
        record(
            "bloom",
            SIMULATED,
            SIMULATION,
            op.bloom != "off" and op.bloom_amount > 0,
            {"amount": op.bloom_amount, "radius_source_px": op.bloom_radius},
        ),
        record(
            "grain",
            SIMULATED,
            SIMULATION,
            no.synthetic_grain != "off" and no.grain_amount > 0,
            {"amount": no.grain_amount, "size_source_px": no.grain_size},
            "Synthetic grain recovers no low-light sensor measurements.",
        ),
    ]
    return records


def information_record(stages: list[dict]) -> dict:
    """The manifest ``information`` record (preserved / interpolated / inferred / simulated) for these stages."""
    names: dict[str, list[str]] = {key: [] for key in MANIFEST_INFORMATION_KEYS.values()}
    for stage in stages:
        if stage["enabled"]:
            names[MANIFEST_INFORMATION_KEYS[stage["provenance"]]].append(stage["name"])
    return {
        "preserved": "source structure and region labels up to the source sampling limit",
        "interpolated": ", ".join(names["interpolated"]) or "none",
        "inferred": ", ".join(names["inferred"]) or "none (no learned models)",
        "simulated": ", ".join(names["simulated"]) or "none",
    }


def process_night(
    image: np.ndarray,
    regions: RegionMap,
    source_linear: np.ndarray,
    params: NightParams | None = None,
    *,
    seed: int = 42,
    out: np.ndarray | None = None,
    strip_rows: int = 256,
    upscale: UpscaleResult | None = None,
) -> NightResult:
    """Apply Layer 5B to an upscaled working-space image.

    ``image`` is the (H, W, 3) float32 upscale of ``source_linear``; ``regions``
    were segmented on the source. ``out`` may be ``image`` itself to process in
    place, which keeps a 50 MP run at one full-size buffer.
    """
    params = params or NightParams()
    if image.ndim != 3 or image.dtype != np.float32:
        raise ValueError(f"expected an (H, W, 3) float32 image, got {image.shape} {image.dtype}")
    if regions.shape != source_linear.shape[:2]:
        raise ValueError("regions must be segmented on source_linear")
    height, width = image.shape[:2]
    scale = width / source_linear.shape[1]
    out = np.empty_like(image) if out is None else out
    in_place = out is image
    conservative = params.conservative()
    block = params.failure_rule.block_size
    margin = _filter_margin(params, scale) + 3 * block
    envelope = _Envelope(source_linear)
    before, after, paired = _Metrics(scale), _Metrics(scale), _Paired()
    fallbacks: list[dict] = []
    fallback_weight = 0.0
    simulated: set[str] = set()
    history = np.empty((0, width, 3), dtype=np.float32)  # unmodified rows above the current strip (in place)

    for y0, y1, p0, p1 in _strips(height, strip_rows, margin):
        if in_place:
            above = history[history.shape[0] - (y0 - p0) :] if y0 > p0 else history[:0]
            padded = np.concatenate([above, image[y0:p1]]) if y0 > p0 else image[p0:p1].copy()
        else:
            padded = np.ascontiguousarray(image[p0:p1])
        strip = _Strip(
            y0,
            y1,
            p0,
            p1,
            scale,
            regions.soft_strip(width, height, p0, p1),
            envelope,
            (width, height),
        )
        before.add(padded, strip, strip.footprint(None))
        candidate = _process(padded, strip, params, scale)
        reference = strip.footprint(params)
        if params.failure_rule.enabled:
            safe = _process(padded, strip, conservative, scale)
            weight, records = _failure_rule(padded, candidate, strip, params, block, width, height)
            # A block that failed in a neighbouring strip still feathers into this one.
            candidate += weight[..., None] * (safe - candidate)
            fallbacks.extend(records)
            fallback_weight += float(weight[strip.core].sum())
            # Halo reference for the blended result: each pixel against the tone curve it went through.
            reference = tuple(
                r + weight * (c - r) for r, c in zip(reference, strip.footprint(conservative), strict=True)
            )
            del safe
        result, applied = _simulated(candidate, strip, params, seed)
        simulated.update(applied)
        after.add_core(candidate, result, strip, reference)
        paired.add(padded[strip.core], result, strip)
        if in_place:
            history = np.concatenate([history, image[y0:y1]])[-margin:].copy()
        out[y0:y1] = result
        del padded, candidate, result

    stages = stage_records(params, upscale)
    return NightResult(
        out,
        stages,
        fallbacks,
        fallback_weight / (height * width),
        before.result(),
        after.result(),
        {**paired.result(), "local_contrast_change": _ratios(before, after)},
        information_record(stages),
    )


# --------------------------------------------------------------------------- failure rule (PRD 12.6)


def _failure_rule(
    before: np.ndarray,
    candidate: np.ndarray,
    strip: _Strip,
    params: NightParams,
    block: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, list[dict]]:
    """Per-block fallback weights for the padded rows of a strip, and the core blocks that failed."""
    rule = params.failure_rule
    texture_limit = _TEXTURE_RATIO_LIMIT[params.shadows.invented_texture_tolerance]
    # Block rows evaluated: two beyond the core on each side, for the dilation and the feathering.
    b0 = max(0, strip.y0 // block - 2)
    b1 = min(math.ceil(height / block), math.ceil(strip.y1 / block) + 2)
    r0, r1 = b0 * block - strip.p0, min(b1 * block, height) - strip.p0
    if r0 < 0 or r1 > before.shape[0]:
        raise AssertionError("strip margin too small for the failure-rule blocks")
    rows = slice(r0, r1)

    y_in = _encoded_luma(before)
    y_out = _encoded_luma(candidate)
    ring = strip.hard("ring")[rows]
    smooth = strip.hard("smooth_gradient", "sky_atmosphere")[rows]
    near_black = strip.hard("near_black")[rows]
    quiet = strip.hard("near_black", "low_luminance_textured", "sky_atmosphere", "smooth_gradient")[rows]

    halo_in = (_halo_excursion(y_in, *strip.footprint(None)) > _HALO_CODES * _CODE)[rows] & ring
    halo_out = (_halo_excursion(y_out, *strip.footprint(params)) > _HALO_CODES * _CODE)[rows] & ring
    flat_in, ramp_in = (m[rows] & smooth for m in _banding_maps(y_in, strip.scale))
    flat_out, ramp_out = (m[rows] & smooth for m in _banding_maps(y_out, strip.scale))
    crush_in = (y_in < 0.5 * _CODE)[rows] & near_black
    crush_out = (y_out < 0.5 * _CODE)[rows] & near_black
    hf_in = _high_pass(y_in, strip.scale)[rows] ** 2
    hf_out = _high_pass(y_out, strip.scale)[rows] ** 2

    def sums(values):
        return _block_sum(values.astype(np.float32), block)

    n_ring, n_smooth, n_black, n_quiet = sums(ring), sums(smooth), sums(near_black), sums(quiet)
    pixels = sums(np.ones_like(ring))
    enough = 0.25 * pixels
    with np.errstate(divide="ignore", invalid="ignore"):
        halo = (n_ring >= 0.05 * pixels) & (
            (sums(halo_out) - sums(halo_in)) / np.maximum(n_ring, 1) > rule.halo_increase
        )
        banding = (n_smooth >= enough) & (
            sums(flat_out) / np.maximum(sums(ramp_out), 1) - sums(flat_in) / np.maximum(sums(ramp_in), 1)
            > rule.banding_increase
        )
        crush = (n_black >= enough) & (
            (sums(crush_out) - sums(crush_in)) / np.maximum(n_black, 1) > rule.black_crush_increase
        )
        energy_in = sums(hf_in * quiet) / np.maximum(n_quiet, 1)
        energy_out = sums(hf_out * quiet) / np.maximum(n_quiet, 1)
        texture = (
            (n_quiet >= enough)
            & (energy_out > (0.5 * _CODE) ** 2)
            & (energy_out > texture_limit * np.maximum(energy_in, (0.25 * _CODE) ** 2))
        )
    failed = halo | banding | crush | texture
    if not failed.any():
        return np.zeros((strip.p1 - strip.p0, width), dtype=np.float32), []

    grown = cv2.dilate(failed.astype(np.uint8), np.ones((3, 3), np.uint8))
    ys = (np.arange(strip.p0, strip.p1, dtype=np.float64) + 0.5) / block - 0.5
    xs = (np.arange(width, dtype=np.float64) + 0.5) / block - 0.5
    weight = _bilinear(grown, ys, xs, y_origin=b0)
    records = []
    reasons = {"halo": halo, "banding": banding, "black_crush": crush, "excess_texture": texture}
    counts = {"ring": n_ring, "smooth": n_smooth, "near_black": n_black, "quiet": n_quiet}
    region_for = {"halo": "ring", "banding": "smooth", "black_crush": "near_black", "excess_texture": "quiet"}
    for by, bx in zip(*np.nonzero(failed), strict=True):
        y_abs = (b0 + by) * block
        if not strip.y0 <= y_abs < strip.y1:
            continue  # recorded by the strip whose core contains it
        failed_reasons = [r for r, m in reasons.items() if m[by, bx]]
        records.append(
            {
                "x": int(bx * block),
                "y": int(y_abs),
                "width": int(min(block, width - bx * block)),
                "height": int(min(block, height - y_abs)),
                "reasons": failed_reasons,
                "regions": _block_regions(strip, rows, by, bx, block),
                "pixels_by_check": {r: int(counts[region_for[r]][by, bx]) for r in failed_reasons},
            }
        )
    return np.clip(weight, 0, 1), records


def _block_regions(strip: _Strip, rows: slice, by: int, bx: int, block: int) -> list[str]:
    ys = slice(rows.start + by * block, rows.start + (by + 1) * block)
    xs = slice(bx * block, (bx + 1) * block)
    shares = {name: float(strip.weights[name][ys, xs].mean()) for name in REGION_NAMES}
    return [name for name, share in sorted(shares.items(), key=lambda kv: -kv[1]) if share >= 0.1]


def _block_sum(values: np.ndarray, block: int) -> np.ndarray:
    rows = np.add.reduceat(values, np.arange(0, values.shape[0], block), axis=0)
    return np.add.reduceat(rows, np.arange(0, values.shape[1], block), axis=1)


def _encoded_luma(linear: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(cm.linear_to_srgb(np.clip(linear, 0, 1)) @ LUMA_WEIGHTS)


def _halo_excursion(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """How far each luma lies outside the range of the source samples around it (undershoot is a dark halo)."""
    return np.maximum(low - y, 0) + np.maximum(y - high, 0)


def _high_pass(y: np.ndarray, scale: float) -> np.ndarray:
    return y - _blur(y, max(1.0, scale))


def _banding_maps(y: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel: (locally flat, inside a ramp) along the steeper axis.

    A smooth ramp changes a little at every pixel; a quantised (banded) ramp is
    flat between steps. Dither and noise are never flat. The banding score is the
    flat share of ramp pixels.
    """
    k = max(4, round(8 * scale))
    fine_x = np.abs(np.diff(y, axis=1, append=y[:, -1:]))
    fine_y = np.abs(np.diff(y, axis=0, append=y[-1:]))
    padded = np.pad(y, k, mode="edge")
    coarse_x = np.abs(padded[k:-k, 2 * k :] - padded[k:-k, : -2 * k]) / (2 * k)
    coarse_y = np.abs(padded[2 * k :, k:-k] - padded[: -2 * k, k:-k]) / (2 * k)
    use_x = coarse_x >= coarse_y
    fine = np.where(use_x, fine_x, fine_y)
    coarse = np.where(use_x, coarse_x, coarse_y)
    ramp = (coarse > 0.002 * _CODE) & (coarse < 0.5 * _CODE)
    flat = fine < 0.25 * coarse
    return flat & ramp, ramp


def banding_map(linear: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Banding diagnostic of a working-space image: (flat inside a ramp, ramp) masks of its encoded luma.

    ``scale`` is the output-to-source scale; the banding score is the flat share of ramp pixels
    in the smooth and sky regions.
    """
    return _banding_maps(_encoded_luma(linear), scale)


# --------------------------------------------------------------------------- night metrics (PRD 23.5)


class _Metrics:
    """Single-image night metrics accumulated strip by strip."""

    def __init__(self, scale: float):
        self.scale = scale
        self.n = {
            "pixels": 0,
            "clip_any": 0,
            "clip_all": 0,
            "light": 0,
            "clip_light": 0,
            "ramp": 0,
            "flat": 0,
            "ring": 0,
            "halo": 0,
            "near_black": 0,
            "crushed": 0,
            "shadow": 0,
        }
        self.halo_excursion = 0.0
        self.chroma_sq = 0.0
        self.contrast_sq = dict.fromkeys(REGION_NAMES, 0.0)
        self.contrast_n = dict.fromkeys(REGION_NAMES, 0)

    def add(self, padded: np.ndarray, strip: _Strip, reference: tuple[np.ndarray, np.ndarray]) -> None:
        self.add_core(padded, padded[strip.core], strip, reference)

    def add_core(
        self, padded: np.ndarray, core_image: np.ndarray, strip: _Strip, reference: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Measure ``core_image`` (the strip's core rows); ``padded`` gives filter context around them.

        ``reference`` is the encoded luma range of the 2x2 source samples around each padded
        pixel, after the same pointwise operations as the image, used by the halo check.
        """
        core = strip.core
        context = padded.copy()
        context[core] = core_image
        n = self.n
        n["pixels"] += core_image.shape[0] * core_image.shape[1]
        clipped = core_image >= _CLIP_LEVEL
        n["clip_any"] += int(clipped.any(axis=2).sum())
        n["clip_all"] += int(clipped.all(axis=2).sum())
        light = strip.hard(*_LIGHTS)[core]
        n["light"] += int(light.sum())
        n["clip_light"] += int((clipped.any(axis=2) & light).sum())

        encoded = cm.linear_to_srgb(np.clip(context, 0, 1))
        y = np.ascontiguousarray(encoded @ LUMA_WEIGHTS)
        smooth = strip.hard("smooth_gradient", "sky_atmosphere")[core]
        flat, ramp = _banding_maps(y, self.scale)
        n["ramp"] += int((ramp[core] & smooth).sum())
        n["flat"] += int((flat[core] & smooth).sum())

        ring = strip.hard("ring")[core]
        yc = y[core]
        excursion = _halo_excursion(yc, reference[0][core], reference[1][core])
        n["ring"] += int(ring.sum())
        n["halo"] += int((ring & (excursion > _HALO_CODES * _CODE)).sum())
        self.halo_excursion += float(excursion[ring].sum())

        near_black = strip.hard("near_black")[core]
        n["near_black"] += int(near_black.sum())
        n["crushed"] += int((near_black & (yc < 0.5 * _CODE)).sum())

        shadow = strip.hard("near_black", "low_luminance_textured")[core]
        chroma = encoded - y[..., None]
        for c in range(3):
            residual = (chroma[..., c] - _blur(chroma[..., c], 1.5 * self.scale))[core]
            self.chroma_sq += float(np.square(residual[shadow], dtype=np.float64).sum())
        n["shadow"] += int(shadow.sum())

        detail = (y - _blur(y, 6.0 * self.scale))[core]
        labels = {name: strip.weights[name][core] > 0.5 for name in REGION_NAMES}
        for name, mask in labels.items():
            self.contrast_sq[name] += float(np.square(detail[mask], dtype=np.float64).sum())
            self.contrast_n[name] += int(mask.sum())

    def result(self) -> dict:
        n = self.n

        def ratio(a, b):
            return None if n[b] == 0 else n[a] / n[b]

        return {
            "highlight_clipping_fraction": {
                "any_channel": ratio("clip_any", "pixels"),
                "all_channels": ratio("clip_all", "pixels"),
                "within_light_regions": ratio("clip_light", "light"),
                "note": "share of samples at full scale; a roll-off lowers it without recovering radiance",
            },
            "banding_score": ratio("flat", "ramp"),
            "banding_ramp_pixels": n["ramp"],
            "halo": {
                "fraction": ratio("halo", "ring"),
                "mean_excursion_codes": None if n["ring"] == 0 else self.halo_excursion / n["ring"] / _CODE,
                "ring_pixels": n["ring"],
                "note": "share of ring pixels more than 2 codes outside the luma range of the 2x2 source samples "
                "around them; independent of the halo clamp, which uses the wider 3x3 range",
            },
            "shadow_chroma_noise_codes": None
            if n["shadow"] == 0
            else math.sqrt(self.chroma_sq / (3 * n["shadow"])) / _CODE,
            "black_crush_fraction": ratio("crushed", "near_black"),
            "local_contrast_codes": {
                name: None
                if self.contrast_n[name] == 0
                else math.sqrt(self.contrast_sq[name] / self.contrast_n[name]) / _CODE
                for name in REGION_NAMES
            },
        }


class _Paired:
    """Before/after comparisons: colour difference in protected regions, shadow chroma stability."""

    _BINS = np.linspace(0, 50, 1001)

    def __init__(self):
        self.histogram = np.zeros(len(self._BINS) - 1, dtype=np.int64)
        self.delta_sum = 0.0
        self.chromatic_sum = 0.0
        self.protected = 0
        self.chroma_shift = 0.0
        self.shadow = 0

    def add(self, before_core: np.ndarray, after_core: np.ndarray, strip: _Strip) -> None:
        from skimage.color import deltaE_ciede2000, rgb2lab

        core = strip.core
        protected = strip.hard(*_LIGHTS, "specular_reflection")[core]
        if protected.any():
            a = cm.linear_to_srgb(np.clip(before_core[protected], 0, 1))[:, None, :]
            b = cm.linear_to_srgb(np.clip(after_core[protected], 0, 1))[:, None, :]
            lab_a, lab_b = rgb2lab(a), rgb2lab(b)
            delta = deltaE_ciede2000(lab_a, lab_b).ravel()
            # Hue and chroma only: the highlight roll-off changes lightness by design.
            self.chromatic_sum += float(deltaE_ciede2000(lab_a, lab_b, kL=1e6).sum())
            self.histogram += np.histogram(np.minimum(delta, 49.999), self._BINS)[0]
            self.delta_sum += float(delta.sum())
            self.protected += delta.size
        shadow = strip.hard("near_black", "low_luminance_textured")[core]
        if shadow.any():
            ea = cm.linear_to_srgb(np.clip(before_core[shadow], 0, 1))
            eb = cm.linear_to_srgb(np.clip(after_core[shadow], 0, 1))
            ca = ea - (ea @ LUMA_WEIGHTS)[:, None]
            cb = eb - (eb @ LUMA_WEIGHTS)[:, None]
            self.chroma_shift += float(np.abs(cb - ca).sum())
            self.shadow += int(shadow.sum())

    def result(self) -> dict:
        p95 = None
        if self.protected:
            cumulative = np.cumsum(self.histogram)
            p95 = float(self._BINS[1:][np.searchsorted(cumulative, 0.95 * self.protected)])
        return {
            "protected_color_difference_ciede2000": {
                "mean": None if not self.protected else self.delta_sum / self.protected,
                "p95": p95,
                "chromatic_mean": None if not self.protected else self.chromatic_sum / self.protected,
                "pixels": self.protected,
                "regions": [*_LIGHTS, "specular_reflection"],
            },
            "shadow_chroma_mean_shift_codes": None
            if not self.shadow
            else self.chroma_shift / (3 * self.shadow) / _CODE,
        }


def _ratios(before: _Metrics, after: _Metrics) -> dict:
    a, b = before.result()["local_contrast_codes"], after.result()["local_contrast_codes"]
    return {name: None if not a[name] or b[name] is None else b[name] / a[name] for name in REGION_NAMES}


def night_metrics(
    image: np.ndarray,
    regions: RegionMap,
    source_linear: np.ndarray,
    *,
    params: NightParams | None = None,
    strip_rows: int = 512,
) -> dict:
    """PRD 23.5 metrics of one working-space image (the source, a baseline or a processed result).

    ``params`` are the night parameters the image was processed with (None for
    an unprocessed image), so the halo check compares it with an equally
    processed source.
    """
    height, width = image.shape[:2]
    scale = width / source_linear.shape[1]
    margin = math.ceil(3 * 6.0 * max(scale, 1.0)) + 2 * max(4, round(8 * scale)) + 2
    envelope = _Envelope(source_linear)
    metrics = _Metrics(scale)
    for y0, y1, p0, p1 in _strips(height, strip_rows, margin):
        strip = _Strip(y0, y1, p0, p1, scale, regions.soft_strip(width, height, p0, p1), envelope, (width, height))
        metrics.add(np.ascontiguousarray(image[p0:p1]), strip, strip.footprint(params))
    return metrics.result()


def fallback_map(result: NightResult, width: int, height: int, max_edge: int = 1024) -> np.ndarray:
    """uint8 RGB map of the failure-rule fallback blocks, coloured by reason, for the report."""
    scale = min(1.0, max_edge / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    canvas = np.full((size[1], size[0], 3), 24, dtype=np.uint8)
    colours = {
        "halo": (255, 80, 80),
        "banding": (80, 200, 255),
        "black_crush": (160, 80, 255),
        "excess_texture": (255, 200, 60),
    }
    for record in result.fallbacks:
        colour = colours[record["reasons"][0]]
        x0, y0 = int(record["x"] * scale), int(record["y"] * scale)
        x1 = max(x0 + 1, int((record["x"] + record["width"]) * scale))
        y1 = max(y0 + 1, int((record["y"] + record["height"]) * scale))
        canvas[y0:y1, x0:x1] = colour
    return canvas


# --------------------------------------------------------------------------- synthetic night scene


def make_night_scene(width: int = 3840, height: int = 2160, seed: int = 42) -> np.ndarray:
    """Return an (height, width, 3) uint8 sRGB synthetic night scene.

    Deterministic for a given size and seed. Contains what Layer 5B must
    handle: an 8-bit sky gradient that bands, dark smooth road gradients,
    noisy textured shadows with chroma noise, clipped practical lights with
    coloured glows, lit windows, a saturated neon sign and specular reflections
    on a wet road.
    """
    if width < 64 or height < 64:
        raise ValueError("night scenes must be at least 64 x 64")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    v = yy / (height - 1)
    horizon = 0.62

    # Sky: deep blue at the top to a faint warm glow at the horizon (smooth, quantises into bands).
    t = np.clip(v / horizon, 0, 1) ** 2
    top = np.array([0.0015, 0.0025, 0.010], dtype=np.float32)
    glow = np.array([0.030, 0.020, 0.024], dtype=np.float32)
    linear = top + t[..., None] * (glow - top)

    # Road: a dark smooth gradient below the horizon, brighter towards the viewer.
    road = v >= horizon
    depth = np.clip((v - horizon) / (1 - horizon), 0, 1)
    linear[road] = (0.003 + 0.010 * depth[road])[:, None] * np.array([1.0, 0.95, 0.9], dtype=np.float32)

    # Buildings along the horizon, edge to edge, so the sky stays separate from the road.
    noise_mask = np.zeros((height, width), dtype=bool)
    x = 0
    windows = []
    while x < width:
        w = int(rng.uniform(0.06, 0.14) * width)
        top_v = rng.uniform(0.22, 0.48)
        y0 = int(top_v * height)
        y1 = int(horizon * height)
        tone = rng.uniform(0.003, 0.012)
        tint = np.array([1.0, 0.95, 1.05], dtype=np.float32) * tone
        linear[y0:y1, x : x + w] = tint
        # Floor lines: faint architectural structure.
        for fy in range(y0, y1, max(4, height // 60)):
            linear[fy : fy + max(1, height // 600), x : x + w] *= 1.6
        noise_mask[y0:y1, x : x + w] = True
        step_x, step_y = max(6, width // 90), max(8, height // 40)
        win_w, win_h = max(2, step_x // 2), max(3, step_y // 2)
        for wy in range(y0 + step_y // 2, y1 - step_y, step_y):
            for wx in range(x + step_x // 2, x + w - step_x // 2, step_x):
                if rng.random() < 0.3:
                    windows.append((wx, wy, win_w, win_h, rng.uniform(0.25, 1.4)))
        x += w

    for wx, wy, ww, wh, level in windows:
        linear[wy : wy + wh, wx : wx + ww] = np.array([1.0, 0.62, 0.28], dtype=np.float32) * level

    # Street lights: clipped cores, coloured glows and reflections on the wet road.
    lamps = [
        (0.12, (1.0, 0.45, 0.12)),
        (0.33, (1.0, 0.50, 0.15)),
        (0.55, (0.75, 0.88, 1.0)),
        (0.78, (1.0, 0.45, 0.12)),
        (0.93, (1.0, 0.50, 0.18)),
    ]
    lamp_y = (horizon - 0.08) * height
    for lx, colour in lamps:
        colour = np.array(colour, dtype=np.float32)
        cx = lx * width
        d2 = (xx - cx) ** 2 + (yy - lamp_y) ** 2
        core = max(1.5, 0.003 * width)
        halo = 0.025 * width
        linear += (3.0 * np.exp(-d2 / (2 * core**2)))[..., None] * colour
        linear += (0.10 * np.exp(-d2 / (2 * halo**2)))[..., None] * colour
        pole = (np.abs(xx - cx) < max(1, 0.0012 * width)) & (yy > lamp_y + core * 2) & (v < horizon + 0.12)
        linear[pole] = 0.002
        # Reflection: a vertical streak below the light, rippled by the wet surface.
        streak = road & (np.abs(xx - cx) < 0.02 * width)
        fall = np.exp(-((xx - cx) ** 2) / (2 * (0.006 * width) ** 2)) * np.exp(-(v - horizon) * 3)
        ripple = 0.55 + 0.45 * np.sin(yy * (60.0 / height) * 2 * np.pi) ** 2
        linear[streak] += (0.5 * fall * ripple)[streak][:, None] * colour

    # Neon sign: a saturated magenta frame and cyan bars on one building.
    sx0, sy0 = int(0.42 * width), int(0.40 * height)
    sw, sh = int(0.1 * width), int(0.05 * height)
    thickness = max(1, width // 500)
    magenta = np.array([1.6, 0.08, 0.9], dtype=np.float32)
    cyan = np.array([0.05, 0.9, 1.2], dtype=np.float32)
    linear[sy0 : sy0 + thickness, sx0 : sx0 + sw] = magenta
    linear[sy0 + sh : sy0 + sh + thickness, sx0 : sx0 + sw] = magenta
    linear[sy0 : sy0 + sh, sx0 : sx0 + thickness] = magenta
    linear[sy0 : sy0 + sh + thickness, sx0 + sw : sx0 + sw + thickness] = magenta
    for i in range(3):
        bx = sx0 + (i + 1) * sw // 4
        linear[sy0 + sh // 4 : sy0 + 3 * sh // 4, bx : bx + thickness * 2] = cyan

    encoded = cm.linear_to_srgb(np.clip(linear, 0, 1))
    # Noisy shadows with chroma noise on buildings and the near road; the sky and far road stay clean.
    noise_mask |= road & (depth > 0.45)
    dark = noise_mask & (encoded.max(axis=2) < 0.3)
    luma_noise = rng.normal(0, 1.2 / 255, size=(height, width)).astype(np.float32)
    chroma_noise = rng.normal(0, 1.6 / 255, size=(height, width, 3)).astype(np.float32)
    encoded[dark] += luma_noise[dark][:, None] + chroma_noise[dark]
    return (np.clip(encoded, 0, 1) * 255).round().astype(np.uint8)


def write_night_scene(path: str | Path, width: int = 3840, height: int = 2160, seed: int = 42) -> Path:
    """Write the synthetic night scene as an untagged 8-bit PNG, like most AI-generated images."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(make_night_scene(width, height, seed)).save(path)
    return path
