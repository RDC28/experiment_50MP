"""Experiment configuration (PRD sections 6 and 22.2).

Every parameter declares its default, a description, the accepted values, its
effect and, where relevant, a warning for high-risk settings. The notebook
renders that table with :func:`parameter_table`; YAML files only override
defaults and any unknown key is rejected so a typo never silently falls back.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .color_management import SOURCE_COLOR_SPACES, WORKING_SPACE
from .resolution import RESOLUTION_MODES
from .sr_models import BACKENDS as SR_BACKENDS
from .sr_models import DEVICES as SR_DEVICES
from .sr_models import PRECISIONS as SR_PRECISIONS
from .super_resolution import MODES as SR_MODES

BASELINE_METHODS = ("lanczos", "bicubic")
INPUT_CATEGORIES = ("human", "night", "universal")
RUNTIME_PRESETS = ("quick", "standard", "research")
TIFF_COMPRESSIONS = ("zlib", "none")
EXR_PRECISIONS = ("half", "float")
EXR_COMPRESSIONS = ("zip", "piz", "none")


def _param(
    default: Any,
    description: str,
    accepted: str,
    effect: str,
    *,
    warning: str = "",
    choices: tuple | None = None,
    bounds: tuple[float, float] | None = None,
):
    meta = {
        "description": description,
        "accepted": accepted,
        "effect": effect,
        "warning": warning,
        "choices": choices,
        "bounds": bounds,
    }
    if isinstance(default, (list, tuple)):
        return field(default_factory=lambda: tuple(default), metadata=meta)
    return field(default=default, metadata=meta)


@dataclass(frozen=True)
class ExperimentConfig:
    profile: str = _param(
        "baselines",
        "Name of this configuration, recorded in every manifest.",
        "any non-empty text",
        "Labels outputs and reports.",
    )
    input_category: str = _param(
        "universal",
        "Which input/output folder the run belongs to.",
        "human | night | universal",
        "Selects inputs/<category>/ and outputs/<category>/.",
        choices=INPUT_CATEGORIES,
    )
    runtime_preset: str = _param(
        "standard",
        "PRD 21.6 runtime preset.",
        "quick | standard | research",
        "quick: reduced output size; standard: full target; research: full target plus a "
        "reproducibility re-run of every baseline.",
        warning="research doubles baseline runtime.",
        choices=RUNTIME_PRESETS,
    )
    quick_target_megapixels: float = _param(
        2.0,
        "Output size used by the quick preset.",
        "0.01 to 50",
        "Replaces target_megapixels when runtime_preset is quick.",
        bounds=(0.01, 50.0),
    )
    target_megapixels: float = _param(
        50.0,
        "Desired output pixel count in megapixels.",
        "0.01 to 400",
        "Sets the reconstruction size; memory and storage scale linearly with it.",
        warning="Values above 50 can exhaust Colab RAM.",
        bounds=(0.01, 400.0),
    )
    resolution_mode: str = _param(
        "tolerance",
        "How integer target dimensions are chosen (PRD 3.5, 25.3).",
        "tolerance | exact",
        "tolerance: pixel count within pixel_count_tolerance, aspect ratio as close as possible; "
        "exact: exactly target pixels using a divisor pair.",
        warning="exact mode usually cannot keep the source aspect ratio.",
        choices=RESOLUTION_MODES,
    )
    pixel_count_tolerance: float = _param(
        0.001,
        "Allowed relative pixel-count deviation in tolerance mode.",
        "0 to 0.05",
        "0.001 is the PRD 25.3 gate of +/-0.1%.",
        bounds=(0.0, 0.05),
    )
    dimension_multiple: int = _param(
        2,
        "Output width and height must be multiples of this value.",
        "1, 2, 4, 8 or 16",
        "2 keeps dimensions even, like camera sensors and Bayer mosaics.",
        choices=(1, 2, 4, 8, 16),
    )
    preserve_aspect_ratio: bool = _param(
        True,
        "Prevent geometric distortion.",
        "true | false",
        "Enforces max_aspect_ratio_error in exact mode; tolerance mode always minimises aspect error.",
        warning="false allows visible stretching in exact mode.",
    )
    max_aspect_ratio_error: float = _param(
        0.001,
        "Largest accepted relative aspect-ratio change.",
        "0 to 0.1",
        "Exact mode fails rather than distort beyond this.",
        bounds=(0.0, 0.1),
    )
    source_color_space: str = _param(
        "auto",
        "How source code values are interpreted.",
        "auto | srgb | linear_srgb | display_p3 | adobe_rgb",
        "auto uses the embedded ICC profile, assumes sRGB when untagged and linear Rec.709 for EXR.",
        warning="Overriding a correct embedded profile shifts every colour.",
        choices=SOURCE_COLOR_SPACES,
    )
    working_color_space: str = _param(
        WORKING_SPACE,
        "Linear working space for all processing.",
        "linear_rec709",
        "Interpolation happens in linear light with Rec.709/sRGB primaries.",
        choices=(WORKING_SPACE,),
    )
    processing_precision: str = _param(
        "float32",
        "Numerical precision of the working image.",
        "float32",
        "All processing runs in float32 before final quantisation.",
        choices=("float32",),
    )
    baselines: tuple = _param(
        BASELINE_METHODS,
        "Classical interpolation baselines to run (PRD 9.1).",
        "any of lanczos, bicubic",
        "Each baseline produces its own masters, metrics and crops.",
        choices=BASELINE_METHODS,
    )
    enable_super_resolution: bool = _param(
        False,
        "Run inference-only pretrained super-resolution (PRD 6.1, 10; Phase 2).",
        "true | false",
        "Adds one 50 MP master per sr_modes entry, measured and exported like the baselines.",
        warning="Downloads about 135 MB of weights on first use; full 50 MP SR takes minutes on a GPU, longer on "
        "Apple MPS or the CPU.",
    )
    sr_backend: str = _param(
        "swinir",
        "Pretrained model family (PRD 10.1; evaluation in docs/models.md).",
        "swinir | realesrgan",
        "swinir: SwinIR-M classical (fidelity) and real-world GAN (perceptual), the evaluated default; "
        "realesrgan: about 2.5x faster, smooths fine texture and alters faces more.",
        choices=tuple(SR_BACKENDS),
    )
    sr_modes: tuple = _param(
        ("conservative", "balanced"),
        "PRD 10.2 operating modes to run.",
        "any of conservative, balanced, aggressive",
        "conservative: fidelity network, preferred for faces; balanced: perceptual network with low-frequency "
        "correction; aggressive: perceptual network unconstrained.",
        warning="aggressive is experimental, invents the most and must not be used for identity-critical regions.",
        choices=SR_MODES,
    )
    sr_tile_size: int = _param(
        0,
        "SR tile side in source pixels (PRD 10.4).",
        "0 (auto: memory-aware) or 64 to 2048",
        "Larger tiles mean fewer overlaps (less work) but more device and host memory.",
        warning="Too large a tile is caught as out-of-memory and split automatically, which costs time.",
        bounds=(0, 2048),
    )
    sr_tile_overlap: int = _param(
        64,
        "Overlap between neighbouring SR tiles in source pixels (PRD 6.1, 10.4).",
        "64 to 512, at most half the tile size",
        "The outer quarter next to each tile edge is discarded and the middle half crossfaded, hiding seams.",
        warning="Below 64 the networks' border artefacts (measured up to 16 px deep for SwinIR) reach the image.",
        bounds=(64, 512),
    )
    sr_strength: float = _param(
        1.0,
        "General SR strength (PRD 18.2): share of the network output against Lanczos interpolation.",
        "0 to 1",
        "1 uses the network output as is; lower values blend towards the Lanczos baseline.",
        bounds=(0.0, 1.0),
    )
    preserve_source_structure: bool = _param(
        True,
        "Restrict hallucination with the low-frequency source-consistency correction (PRD 6.1, 10.5).",
        "true | false",
        "Applies each mode's correction strength; false disables the correction for every mode.",
        warning="false lets colour and tone drift away from the source.",
    )
    sr_device: str = _param(
        "auto",
        "Inference device.",
        "auto | cuda | mps | cpu",
        "auto prefers CUDA (Colab), then Apple MPS, then the CPU. Out-of-memory tiles fall back to the CPU.",
        choices=SR_DEVICES,
    )
    sr_precision: str = _param(
        "auto",
        "Inference precision (PRD 21.4).",
        "auto | float32 | float16",
        "auto uses float16 on CUDA when the network supports it, float32 otherwise.",
        choices=SR_PRECISIONS,
    )
    sr_weights_dir: str = _param(
        "",
        "Cache folder for downloaded model weights.",
        "path; empty means <project>/weights",
        "Weights are checksum-verified after download and before every load; point this at Drive to reuse them.",
    )
    export_tiff16: bool = _param(
        True,
        "Write the 16-bit sRGB TIFF display master (PRD 17.1 A).",
        "true | false",
        "Lossless 16-bit TIFF with embedded sRGB ICC profile and provenance.",
    )
    tiff_compression: str = _param(
        "zlib",
        "TIFF compression.",
        "zlib | none",
        "zlib is lossless Deflate with horizontal differencing.",
        choices=TIFF_COMPRESSIONS,
    )
    export_exr: bool = _param(
        True,
        "Write the linear-light OpenEXR master (PRD 17.1 B).",
        "true | false",
        "Keeps interpolation overshoot that the integer TIFF must clip.",
    )
    exr_precision: str = _param(
        "half",
        "EXR channel precision.",
        "half | float",
        "half matches the PRD profiles (exr_half: true); float keeps full float32.",
        choices=EXR_PRECISIONS,
    )
    exr_compression: str = _param(
        "zip",
        "EXR compression (all choices are lossless).",
        "zip | piz | none",
        "zip balances size and speed.",
        choices=EXR_COMPRESSIONS,
    )
    export_review: bool = _param(
        True,
        "Write downscaled full-frame review JPEGs (PRD 17.1 C).",
        "true | false",
        "Adds source and baseline previews to the report; crop comparisons are always written.",
    )
    review_max_edge: int = _param(
        2048,
        "Longest edge of full-frame review images in pixels.",
        "256 to 8192",
        "Larger previews show more detail but grow the report.",
        bounds=(256, 8192),
    )
    save_intermediates: bool = _param(
        True,
        "Save the working-space source as a float32 EXR for ablations.",
        "true | false",
        "Lets later phases start from the exact linearised source.",
    )
    crop_size: int = _param(
        256,
        "Side of automatically chosen evaluation crops, in source pixels.",
        "32 to 2048",
        "Larger crops show more context at lower magnification.",
        bounds=(32, 2048),
    )
    crop_count: int = _param(
        4,
        "Number of automatically chosen evaluation crops.",
        "1 to 12",
        "The centre crop plus the most detailed non-overlapping regions.",
        bounds=(1, 12),
    )
    crops: tuple = _param(
        (),
        "Manual evaluation crops as [x, y, width, height] in source pixels.",
        "list of 4-integer lists",
        "When set, replaces the automatic crop selection.",
    )
    random_seed: int = _param(
        42,
        "Seed for every random process in the run.",
        "any integer",
        "Recorded for reproducibility; Phase 0 and 1 processing is deterministic.",
    )
    output_root: str = _param(
        "",
        "Folder for full-resolution masters.",
        "path; empty means <project>/outputs",
        "Point this at Google Drive to keep masters after the Colab VM resets.",
    )
    reports_root: str = _param(
        "",
        "Folder for reports, kept separate from masters (PRD 21.5).",
        "path; empty means <project>/reports",
        "Reports are small and safe to download or archive.",
    )
    inputs_root: str = _param(
        "",
        "Read-only store for ingested source copies.",
        "path; empty means <project>/inputs",
        "Sources are copied here and made read-only before processing.",
    )

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            _check_type(f, value)
            choices, bounds = f.metadata["choices"], f.metadata["bounds"]
            if isinstance(value, tuple) and choices is not None:
                if not value or len(set(value)) != len(value) or any(v not in choices for v in value):
                    raise ValueError(f"{f.name} must be a non-empty subset of {choices} without repeats, got {value!r}")
            elif choices is not None and value not in choices:
                raise ValueError(f"{f.name} must be one of {choices}, got {value!r}")
            if bounds is not None and not bounds[0] <= value <= bounds[1]:
                raise ValueError(f"{f.name} must be between {bounds[0]} and {bounds[1]}, got {value!r}")
        if not self.profile.strip():
            raise ValueError("profile must not be empty")
        if self.sr_tile_size and self.sr_tile_size < 64:
            raise ValueError(f"sr_tile_size must be 0 (auto) or at least 64, got {self.sr_tile_size}")
        if self.sr_tile_size and self.sr_tile_overlap * 2 > self.sr_tile_size:
            raise ValueError("sr_tile_overlap must be at most half of sr_tile_size")
        for crop in self.crops:
            if len(crop) != 4 or any(int(v) != v or v < 0 for v in crop) or crop[2] < 1 or crop[3] < 1:
                raise ValueError(f"each crop must be [x, y, width, height] with non-negative integers, got {crop!r}")

    @property
    def effective_target_megapixels(self) -> float:
        return self.quick_target_megapixels if self.runtime_preset == "quick" else self.target_megapixels

    def replace(self, **overrides: Any) -> ExperimentConfig:
        return from_dict({**self.to_dict(), **overrides})

    def to_dict(self) -> dict:
        data = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        data["baselines"] = list(self.baselines)
        data["sr_modes"] = list(self.sr_modes)
        data["crops"] = [list(crop) for crop in self.crops]
        return data


def _check_type(f: dataclasses.Field, value: Any) -> None:
    default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
    if isinstance(default, bool):
        ok = isinstance(value, bool)
    elif isinstance(default, int):
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif isinstance(default, float):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    else:
        ok = isinstance(value, type(default))
    if not ok:
        raise ValueError(f"{f.name} must be of type {type(default).__name__}, got {value!r}")


def from_dict(values: dict[str, Any]) -> ExperimentConfig:
    known = {f.name for f in dataclasses.fields(ExperimentConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown configuration keys: {', '.join(unknown)}")
    normalized = dict(values)
    for name in ("baselines", "sr_modes"):
        if name in normalized:
            normalized[name] = tuple(normalized[name])
    if "crops" in normalized:
        normalized["crops"] = tuple(tuple(c) for c in normalized["crops"] or ())
    for name in (
        "target_megapixels",
        "quick_target_megapixels",
        "pixel_count_tolerance",
        "max_aspect_ratio_error",
        "sr_strength",
    ):
        if isinstance(normalized.get(name), int) and not isinstance(normalized.get(name), bool):
            normalized[name] = float(normalized[name])
    for name in ("output_root", "reports_root", "inputs_root", "sr_weights_dir"):
        if normalized.get(name) is None:
            normalized[name] = ""
    return ExperimentConfig(**normalized)


def load_config(path: str | Path | None = None, **overrides: Any) -> ExperimentConfig:
    """Load defaults, then a YAML file (if given), then keyword overrides."""
    values: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        values.update(loaded)
    values.update(overrides)
    return from_dict(values)


def parameter_table(config: ExperimentConfig | None = None) -> list[dict]:
    """Rows of (name, default, current, accepted, description, effect, warning) for display."""
    config = config or ExperimentConfig()
    defaults = ExperimentConfig().to_dict()
    current = config.to_dict()
    return [
        {
            "parameter": f.name,
            "default": defaults[f.name],
            "current": current[f.name],
            "accepted": f.metadata["accepted"],
            "description": f.metadata["description"],
            "effect": f.metadata["effect"],
            "warning": f.metadata["warning"],
        }
        for f in dataclasses.fields(ExperimentConfig)
    ]
