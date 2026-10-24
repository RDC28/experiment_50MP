"""Notebook 03 configuration (PRD 20 and Phase 4): the universal profile and the presets it selects.

``configs/universal_default.yaml`` holds the shared experiment settings (PRD 20.1), the scene-analysis
switches and provisional thresholds, the super-resolution ladder, the measurable-improvement gates, the
universal tone, colour and detail settings of the common pipeline, the face-protection blend, and overrides
for the specialised human (notebook 01) and night (notebook 02) profiles that the human and night scenes are
handed to.

Default policy (PRD 4.3): the most conservative settings that produce measurable improvement. The SR ladder
holds only the conservative mode by default: notebook 00 runs on the real test images showed the balanced
mode over-sharpening and failing the seam gate on the night scene, and PRD 10.2 keeps it away from faces.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import config as lab_config
from . import human_config
from . import night_processing as night
from .scene_analysis import FACE_DETECTORS, OVERRIDES, SceneThresholds

UNIVERSAL_PROFILE_DEFAULTS = {"profile": "universal", "input_category": "universal"}
SECTIONS = ("analysis", "super_resolution", "improvement", "processing", "face_protection", "human", "night")
HUMAN_PROFILE = "configs/human_default.yaml"
NIGHT_PROFILE = "configs/night_default.yaml"

# The common pipeline's universal tone, colour and detail settings (Layer 5B engine, PRD 20.1 "conservative"):
# no shadow lift, black point or colour change, a low highlight shoulder for clipping protection, debanding,
# low region-aware sharpening that skips lights, sky and smooth gradients, no denoising, grain or bloom, and
# the PRD 12.6 failure rule. The YAML file overrides single keys.
UNIVERSAL_PROCESSING = {
    "shadows": {
        "lift_strength": "none",
        "black_point": 0.0,
        "color_preservation": 1.0,
        "banding_reduction": True,
        "dither_amplitude": 0.5,
        "invented_texture_tolerance": "low",
    },
    "highlights": {
        "protect": True,
        "rolloff_strength": "low",
        "halo_suppression": True,
        "halo_suppression_strength": 1.0,
        "preserve_light_color": True,
    },
    "color": {
        "preserve_source_mood": True,
        "chromatic_adaptation": "off",
        "color_temperature_k": 6504.0,
        "warm_cool_balance": 0.0,
        "saturation_adjustment": "none",
        "saturation_luminance_coupling": 0.0,
        "local_color_contrast": 0.0,
        "gamut_mapping": "preserve_hue",
    },
    "texture": {
        "architecture_detail_strength": "low",
        "sky_detail_strength": "none",
        "atmospheric_detail_strength": "none",
        "atmospheric_softness": 0.0,
        "sharpening_radius": 1.0,
        "sharpening_strength": 0.25,
    },
    "noise": {"denoise_source": "off", "synthetic_grain": "off", "grain_amount": 0.0},
    "optics": {"bloom": "off", "bloom_amount": 0.0},
    "failure_rule": {"enabled": True},
}

# Shared settings handed to the specialised profiles, so one form controls every route.
_ROUTED = (
    "runtime_preset",
    "target_megapixels",
    "resolution_mode",
    "pixel_count_tolerance",
    "dimension_multiple",
    "preserve_aspect_ratio",
    "max_aspect_ratio_error",
    "source_color_space",
    "random_seed",
    "tiff_compression",
    "exr_compression",
    "review_max_edge",
    "crop_size",
    "output_root",
    "reports_root",
    "inputs_root",
)
# The human profile always writes TIFF16, half EXR and review images and sizes quick runs itself.
_NIGHT_ONLY = ("quick_target_megapixels", "export_tiff16", "export_exr", "exr_precision", "export_review", "crop_count")


@dataclass(frozen=True)
class AnalysisParams:
    """PRD 20.1 ``analysis``: classification switches, face observation and provisional thresholds."""

    automatic_scene_classification: bool = True
    manual_override: str = "auto"
    face_detection: bool = True
    face_detector: str = "yunet"
    manual_face_boxes: tuple = ()
    nighttime_detection: bool = True
    thresholds: SceneThresholds = field(default_factory=SceneThresholds)

    def __post_init__(self) -> None:
        for name in ("automatic_scene_classification", "face_detection", "nighttime_detection"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"analysis.{name} must be true or false")
        if self.manual_override not in OVERRIDES:
            raise ValueError(f"analysis.manual_override must be one of {OVERRIDES}, got {self.manual_override!r}")
        if self.face_detector not in FACE_DETECTORS[:2]:
            raise ValueError(f"analysis.face_detector must be yunet or haar, got {self.face_detector!r}")
        for box in self.manual_face_boxes:
            if not isinstance(box, tuple) or len(box) not in (4, 5):
                raise ValueError("analysis.manual_face_boxes entries must be [x, y, width, height(, confidence)]")

    @property
    def detector(self) -> str:
        """The automatic detector that runs: ``none`` when face detection is off."""
        return self.face_detector if self.face_detection else "none"

    @classmethod
    def from_dict(cls, values: dict) -> AnalysisParams:
        values = dict(values or {})
        _reject_unknown("analysis", values, cls)
        thresholds = values.pop("thresholds", None) or {}
        _reject_unknown("analysis.thresholds", thresholds, SceneThresholds)
        if "manual_face_boxes" in values:
            values["manual_face_boxes"] = tuple(tuple(box) for box in values["manual_face_boxes"] or ())
        return cls(**values, thresholds=SceneThresholds(**thresholds))

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["manual_face_boxes"] = [list(box) for box in self.manual_face_boxes]
        return data


@dataclass(frozen=True)
class ImprovementGates:
    """When a candidate counts as a measurable improvement over Lanczos (provisional, PRD 25.3).

    ``min_detail_gain`` is the relative rise in high-frequency energy (variance of the Laplacian of sRGB luma);
    the other three bound what the candidate may cost in source consistency (CIEDE2000 mean, SSIM) and in
    clipped samples of its 16-bit master.
    """

    min_detail_gain: float = 0.03
    max_delta_e_increase: float = 0.5
    max_ssim_drop: float = 0.01
    max_clipping_increase: float = 0.002

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 10:
                raise ValueError(f"improvement.{f.name} must be a number between 0 and 10, got {value!r}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class FaceProtection:
    """The face masks of the common pipeline: each observed face box grown by ``margin`` of its size on every
    side, with a ``feather`` (fraction of its size) ramp into the surrounding processing."""

    margin: float = 0.25
    feather: float = 0.15

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"face_protection.{f.name} must be a number between 0 and 1, got {value!r}")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class UniversalConfig:
    """A validated universal profile; build it with :func:`load_universal_config`."""

    experiment: lab_config.ExperimentConfig
    analysis: AnalysisParams = field(default_factory=AnalysisParams)
    sr_modes: tuple[str, ...] = ("conservative",)
    super_resolution: night.SuperResolutionParams = field(
        default_factory=lambda: night.SuperResolutionParams(mode="conservative")
    )
    improvement: ImprovementGates = field(default_factory=ImprovementGates)
    processing: night.NightParams = field(default_factory=lambda: night.NightParams.from_dict(UNIVERSAL_PROCESSING))
    face_protection: FaceProtection = field(default_factory=FaceProtection)
    human: dict = field(default_factory=lambda: {"profile": HUMAN_PROFILE, "overrides": {}})
    night: dict = field(default_factory=lambda: {"profile": NIGHT_PROFILE, "overrides": {}})

    def __post_init__(self) -> None:
        modes = self.sr_modes
        if not modes or len(set(modes)) != len(modes) or any(m not in night.SR_COMPARISON_MODES for m in modes):
            raise ValueError(f"super_resolution.modes must list conservative and/or balanced once each, got {modes}")
        if list(modes) != sorted(modes, key=night.SR_COMPARISON_MODES.index):
            raise ValueError("super_resolution.modes is a ladder: list the conservative mode before the balanced one")
        if self.super_resolution.mode != modes[0]:
            raise ValueError("super_resolution.mode is the first rung of super_resolution.modes")
        for name in ("human", "night"):
            route = getattr(self, name)
            if set(route) != {"profile", "overrides"} or not isinstance(route["overrides"], dict):
                raise ValueError(f"{name} must be a mapping of profile (path) and overrides (mapping)")
        if not self.experiment.export_tiff16:
            raise ValueError("export.tiff16 must stay true: the final image of every universal run is a 16-bit TIFF")
        settings = {**self.super_resolution.experiment_settings(), "sr_modes": modes}
        if any(getattr(self.experiment, key) != value for key, value in settings.items()):
            object.__setattr__(self, "experiment", self.experiment.replace(**settings))

    @property
    def sr_enabled(self) -> bool:
        return self.super_resolution.enabled

    def human_config(self, project_root: str | Path, *, sr_enabled: bool | None = None) -> human_config.HumanConfig:
        """The notebook 01 profile for a human scene: its file, the shared settings of this profile, then
        ``human.overrides``. Without SR the human pipeline uses its explicit Lanczos stand-in."""
        enabled = self.sr_enabled if sr_enabled is None else sr_enabled
        sr = self.super_resolution
        values = {name: getattr(self.experiment, name) for name in _ROUTED}
        values["crops"] = [list(crop) for crop in self.experiment.crops]
        values.update(
            sr_backend=sr.backend if enabled else "lanczos_stand_in",
            sr_tile_size=0 if sr.tile_size == "auto" else int(sr.tile_size),
            sr_tile_overlap=sr.tile_overlap,
            sr_device=sr.device,
            sr_precision=sr.precision,
            sr_weights_dir=sr.weights_dir,
            preserve_source_structure=sr.preserve_source_structure,
            face_detector=self.analysis.detector,
            manual_face_boxes=[list(box) for box in self.analysis.manual_face_boxes],
        )
        values.update(copy.deepcopy(self.human["overrides"]))
        return human_config.load_config(_profile_path(self.human["profile"], project_root), **values)

    def night_config(self, project_root: str | Path, *, sr_enabled: bool | None = None) -> night.NightConfig:
        """The notebook 02 profile for a night scene: its file, the shared settings of this profile with the
        first SR rung as the night input, then ``night.overrides``."""
        enabled = self.sr_enabled if sr_enabled is None else sr_enabled
        values: dict[str, Any] = {name: getattr(self.experiment, name) for name in (*_ROUTED, *_NIGHT_ONLY)}
        values["crops"] = [list(crop) for crop in self.experiment.crops]
        values["super_resolution"] = {**self.super_resolution.to_dict(), "enabled": enabled}
        for key, value in copy.deepcopy(self.night["overrides"]).items():
            if isinstance(value, dict) and isinstance(values.get(key), dict):
                values[key] = {**values[key], **value}
            else:
                values[key] = value
        return night.load_night_config(_profile_path(self.night["profile"], project_root), **values)

    def to_dict(self) -> dict:
        return {
            **self.experiment.to_dict(),
            "analysis": self.analysis.to_dict(),
            "super_resolution": {
                **{k: v for k, v in self.super_resolution.to_dict().items() if k != "mode"},
                "modes": list(self.sr_modes),
            },
            "improvement": self.improvement.to_dict(),
            "processing": {k: v for k, v in self.processing.to_dict().items() if k != "super_resolution"},
            "face_protection": self.face_protection.to_dict(),
            "human": copy.deepcopy(self.human),
            "night": copy.deepcopy(self.night),
        }


def load_universal_config(path: str | Path | None = None, **overrides: Any) -> UniversalConfig:
    """Load a universal profile such as ``configs/universal_default.yaml``.

    Flat keys and the ``export`` block are validated by :func:`config.load_profile`; a section override
    such as ``analysis={"manual_override": "night"}`` updates single keys of that section.
    """
    experiment, sections = lab_config.load_profile(path, SECTIONS, UNIVERSAL_PROFILE_DEFAULTS, **overrides)
    sr_values = dict(sections["super_resolution"])
    modes = tuple(sr_values.pop("modes", None) or ("conservative",))
    if "mode" in sr_values:
        raise ValueError("super_resolution.mode is not a universal key: list the ladder in super_resolution.modes")
    if any(mode not in night.SR_COMPARISON_MODES for mode in modes):
        raise ValueError(f"super_resolution.modes must list conservative and/or balanced once each, got {list(modes)}")
    _reject_unknown("super_resolution", sr_values, night.SuperResolutionParams, allowed_extra=())
    processing = copy.deepcopy(UNIVERSAL_PROCESSING)
    for name, values in sections["processing"].items():
        if name == "super_resolution":
            raise ValueError("processing.super_resolution is not allowed: SR settings live in super_resolution")
        if not isinstance(values, dict):
            raise ValueError(f"processing.{name} must be a mapping")
        processing.setdefault(name, {}).update(values)
    _reject_unknown("improvement", sections["improvement"], ImprovementGates)
    _reject_unknown("face_protection", sections["face_protection"], FaceProtection)
    routes = {}
    for name, default in (("human", HUMAN_PROFILE), ("night", NIGHT_PROFILE)):
        given = dict(sections[name])
        unknown = sorted(set(given) - {"profile", "overrides"})
        if unknown:
            raise ValueError(f"unknown keys in {name}: {', '.join(unknown)} (expected profile and overrides)")
        routes[name] = {"profile": given.get("profile") or default, "overrides": dict(given.get("overrides") or {})}
    return UniversalConfig(
        experiment,
        AnalysisParams.from_dict(sections["analysis"]),
        modes,
        night.SuperResolutionParams(**sr_values, mode=modes[0]),
        ImprovementGates(**sections["improvement"]),
        night.NightParams.from_dict(processing),
        FaceProtection(**sections["face_protection"]),
        routes["human"],
        routes["night"],
    )


def parameter_rows(config: UniversalConfig) -> list[dict]:
    """Every universal-only setting as (parameter, value, default) rows for the notebook's configuration table;
    the shared experiment settings come from ``config.parameter_table`` and the processing settings from
    ``night_processing.parameter_table``."""
    defaults = load_universal_config()
    rows = []
    for section in ("analysis", "super_resolution", "improvement", "face_protection", "human", "night"):
        current, default = config.to_dict()[section], defaults.to_dict()[section]
        for key, value in _leaves(current):
            rows.append({"parameter": f"{section}.{key}", "current": value, "default": _lookup(default, key)})
    return rows


def _leaves(mapping: dict, prefix: str = ""):
    for key, value in mapping.items():
        if isinstance(value, dict) and value:
            yield from _leaves(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}", value


def _lookup(mapping: dict, dotted: str):
    for part in dotted.split("."):
        if not isinstance(mapping, dict) or part not in mapping:
            return "-"
        mapping = mapping[part]
    return mapping


def _profile_path(path: str | Path, project_root: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(project_root) / path


def _reject_unknown(section: str, values: dict, cls, allowed_extra: tuple = ()) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"{section} must be a mapping")
    known = {f.name for f in dataclasses.fields(cls)} | set(allowed_extra)
    if cls is night.SuperResolutionParams:
        known.discard("mode")
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown keys in {section}: {', '.join(unknown)}")
