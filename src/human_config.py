"""Notebook 01 configuration, PRD 18.1 and 22.2.

The profile mirrors the human/crowd subset of PRD 18.1 and inherits Phase 2
SR device, precision, weights and tiling controls. Lanczos remains an explicitly
labelled interpolation option. Landmark/embedding models, optics and sensor
simulation are not implemented;
no configuration switch claims to enable them. Generative face replacement and
background-face restoration are prohibited. Region strengths are provisional
numeric interpretations of the PRD's low/moderate policies.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentConfig, _param
from .crowd_processing import RegionDetailConfig
from .face_preservation import FaceBox, FaceTierConfig, IdentityThresholds, ManualFaceDetector

_NESTED = {"face_tiers": FaceTierConfig, "identity": IdentityThresholds, "regions": RegionDetailConfig}
_BASE_NAMES = {f.name for f in dataclasses.fields(ExperimentConfig)}


@dataclasses.dataclass(frozen=True)
class HumanConfig(ExperimentConfig):
    """Validated human profile with shared ingest/resolution/export controls.

    Manual boxes are source-pixel [x, y, width, height] rectangles, optionally
    followed by annotation confidence. Bounds are checked against the image at
    detection time. None selects the largest face in the highest source tier.
    """

    profile: str = _param("human_crowd", "Human profile name.", "human_crowd", "Labels provenance.")
    input_category: str = _param("human", "Human input store.", "human", "Selects human folders.")
    sr_backend: str = _param(
        "swinir",
        "Explicit upscaler backend.",
        "swinir | realesrgan | lanczos_stand_in",
        "Select Phase 2 pretrained SR or explicit Lanczos interpolation for comparison.",
        warning="Learned SR synthesizes detail; interpolation adds no recovered detail.",
    )
    general_sr_strength: float = _param(
        1.0,
        "General upscaler blend strength.",
        "finite number 0 to 1",
        "Blend Lanczos baseline (0) with selected upscaler output (1); default 1.",
        warning="High values may retain upscaler artifacts; current Lanczos stand-in adds no learned detail.",
    )
    edge_sharpening: float = _param(
        0.0,
        "Standalone edge sharpening strength.",
        "finite number 0 to 1",
        "Bounded unsharp mask, sigma 1 output pixel, maximum channel change 0.02; default 0.",
        warning="High values risk halos and amplified artifacts; observed face boxes are excluded.",
    )
    skin_texture_smoothing: float = _param(
        0.0,
        "Edge-preserving skin smoothing strength.",
        "finite number 0 to 1",
        "Default 0; bilateral filter (5 pixels, sigma colour 0.05, space 2), skin-weighted; channel change <=0.02.",
        warning="High values erase skin texture; complete face boxes cap blending by their region face strength.",
    )
    grain: float = _param(
        0.0,
        "Seeded synthetic grain amplitude.",
        "finite number 0 to 1",
        "Default 0; achromatic independent output-pixel noise bounded to 0.02, scaled by sqrt(luminance).",
        warning="Simulation adds random texture, never recovered detail; faces excluded. High values obscure detail.",
    )
    tonal_contrast: float = _param(
        0.0,
        "Log-luminance contrast strength.",
        "finite number -0.5 to 0.5",
        "Slope 1+strength around linear mid-grey 0.18; RGB gain bounded to [0.5, 2]; default 0.",
        warning="High magnitudes alter face appearance and may clip exports; identity recovery may undo face changes.",
    )
    colour_balance: tuple = _param(
        (1.0, 1.0, 1.0),
        "Linear RGB colour-balance gains.",
        "three finite numbers, each 0.5 to 1.5",
        "Multiply linear channels; default [1, 1, 1]. Reports mask-weighted skin CIEDE2000 before recovery.",
        warning="Unequal gains shift skin tone; high gains may clip exports. Identity recovery may undo face changes.",
    )
    face_detector: str = _param(
        "yunet",
        "Automatic source-scale face detector.",
        "yunet | haar | none",
        "Merge verified runtime YuNet inference after manual annotations; haar is optional; none disables inference.",
        warning="Automatic detectors can miss faces; confidence is uncalibrated and never identity proof.",
    )
    manual_face_boxes: tuple = _param(
        (),
        "Manual source-pixel face observations.",
        "list of [x, y, width, height] or [x, y, width, height, confidence]",
        "Manual boxes win overlaps; automatic detections append with IoU de-duplication.",
        warning="Annotations and confidence are not identity proof; omitted faces are unprotected.",
    )
    primary_subject_index: int | None = _param(
        None,
        "Primary subject index into manual_face_boxes.",
        "null or integer from 0 to number of faces minus 1",
        "Manual index overrides; null selects highest tier, then largest area, then first observation.",
        warning="Incorrect selection applies primary protection to the wrong person.",
    )
    face_tiers: FaceTierConfig = _param(
        FaceTierConfig(),
        "Source-scale tier thresholds.",
        "mapping",
        "Assigns provisional face protection tiers.",
        warning="To be calibrated experimentally; never infer reliability from output pixel size.",
    )
    identity: IdentityThresholds = _param(
        IdentityThresholds(),
        "Identity consistency and retry thresholds.",
        "mapping",
        "Controls PRD 11.6 recovery.",
        warning="Consistency is not identity verification; unavailable metrics require human review.",
    )
    regions: RegionDetailConfig = _param(
        RegionDetailConfig(),
        "Region-specific simulated detail.",
        "mapping",
        "Controls bounded microcontrast.",
        warning="Simulated enhancement can alter appearance; inspect masks and face comparisons.",
    )

    def __post_init__(self) -> None:
        if self.sr_backend not in ("swinir", "realesrgan", "lanczos_stand_in"):
            raise ValueError("sr_backend must be swinir, realesrgan or lanczos_stand_in")
        shared = {name: getattr(self, name) for name in _BASE_NAMES}
        # Only the human profile admits interpolation. Validate every other shared
        # field through Phase 2 without widening ExperimentConfig's backend choices.
        if self.sr_backend == "lanczos_stand_in":
            shared["sr_backend"] = "swinir"
        ExperimentConfig(**shared)
        if self.profile != "human_crowd" or self.input_category != "human":
            raise ValueError("human profile requires profile=human_crowd and input_category=human")
        for name in ("general_sr_strength", "edge_sharpening", "grain", "skin_texture_smoothing"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be finite numeric in [0, 1]")
        from .human_pipeline import validate_colour_gains

        if (
            isinstance(self.tonal_contrast, bool)
            or not isinstance(self.tonal_contrast, (int, float))
            or not math.isfinite(self.tonal_contrast)
            or not -0.5 <= self.tonal_contrast <= 0.5
        ):
            raise ValueError("tonal_contrast must be finite numeric in [-0.5, 0.5]")
        if not isinstance(self.colour_balance, tuple):
            raise ValueError("colour_balance must be a tuple")
        validate_colour_gains(self.colour_balance)
        if self.face_detector not in ("yunet", "haar", "none"):
            raise ValueError("face_detector must be yunet, haar or none")
        for name, cls in _NESTED.items():
            if not isinstance(getattr(self, name), cls):
                raise ValueError(f"{name} must be {cls.__name__}")
        if not isinstance(self.manual_face_boxes, tuple):
            raise ValueError("manual_face_boxes must be a tuple")
        faces = []
        for box in self.manual_face_boxes:
            if not isinstance(box, tuple) or len(box) not in (4, 5):
                raise ValueError("manual face boxes require 4 coordinates and optional confidence")
            faces.append(FaceBox(*box))
        ManualFaceDetector(tuple(faces), self.primary_subject_index)
        if not self.export_tiff16 or not self.export_exr or not self.export_review or self.exr_precision != "half":
            raise ValueError("human profile requires TIFF16, half EXR and review exports")

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        for name in ("baselines", "sr_modes", "crops", "manual_face_boxes", "colour_balance"):
            data[name] = [list(v) if isinstance(v, tuple) else v for v in data[name]]
        return data

    def replace(self, **overrides: Any) -> HumanConfig:
        return from_dict({**self.to_dict(), **overrides})


def from_dict(values: dict[str, Any]) -> HumanConfig:
    """Reject unknown keys at every level and validate existing building blocks."""
    if not isinstance(values, dict):
        raise ValueError("human configuration must be a mapping")
    unknown = set(values) - {f.name for f in dataclasses.fields(HumanConfig)}
    if unknown:
        raise ValueError(f"unknown human configuration keys: {', '.join(sorted(unknown))}")
    normalized = dict(values)
    for name, cls in _NESTED.items():
        if name not in normalized:
            continue
        value = normalized[name]
        if not isinstance(value, dict) or set(value) - {f.name for f in dataclasses.fields(cls)}:
            raise ValueError(f"{name} must be a mapping of documented keys")
        normalized[name] = cls(**value)
    for name in ("baselines", "sr_modes", "crops", "manual_face_boxes", "colour_balance"):
        if name in normalized:
            value = normalized[name]
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{name} must be a list")
            if name not in ("baselines", "sr_modes", "colour_balance"):
                if any(not isinstance(v, (list, tuple)) for v in value):
                    raise ValueError(f"{name} entries must be lists")
                value = [tuple(v) for v in value]
            normalized[name] = tuple(value)
    return HumanConfig(**normalized)


def load_config(path: str | Path | None = None, **overrides: Any) -> HumanConfig:
    """Apply YAML then keyword overrides; nested mappings replace that section."""
    values = {} if path is None else yaml.safe_load(Path(path).read_text())
    if not isinstance(values, dict):
        raise ValueError("human configuration must be a YAML mapping")
    return from_dict({**values, **overrides})


def parameter_table(config: HumanConfig | None = None) -> list[dict]:
    """Document every leaf key with default, current, range, effect and warning."""
    config = config or HumanConfig()
    rows = []

    def visit(current, default, prefix=""):
        for field in dataclasses.fields(current):
            name = prefix + field.name
            value, initial = getattr(current, field.name), getattr(default, field.name)
            if dataclasses.is_dataclass(value):
                visit(value, initial, name + ".")
            else:
                rows.append({"parameter": name, "default": initial, "current": value, **field.metadata})

    visit(config, HumanConfig())
    return rows
