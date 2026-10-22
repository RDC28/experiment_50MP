"""Phase 5 synthetic sensor research configuration (PRD sections 14.4, 14.5, 22.2).

Every parameter declares its default, a description, the accepted values, its
effect and, where relevant, a warning, exactly like the Phase 0/1
:mod:`config`. YAML files only override defaults; unknown keys are rejected.
The sensor parameters keep the PRD 14.4 names.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import sensor_simulation as ss
from .config import _check_type as _check_base_type
from .config import _param


@dataclass(frozen=True)
class SensorResearchConfig:
    profile: str = _param(
        "sensor_research",
        "Name of this configuration, recorded in every manifest.",
        "any non-empty text",
        "Labels outputs and reports.",
    )
    input_category: str = _param(
        "universal",
        "Which output folder the run belongs to.",
        "human | night | universal",
        "Selects outputs/<category>/.",
        choices=("human", "night", "universal"),
    )
    cfa_pattern: str = _param(
        "RGGB",
        "Simulated Bayer layout (PRD 14.4), the 2 x 2 tile read row by row.",
        "RGGB | BGGR | GRBG | GBRG",
        "Which colour each photosite samples.",
        choices=tuple(ss.CFA_PATTERNS),
    )
    adc_bit_depth: int = _param(
        14,
        "Quantisation precision of the simulated ADC.",
        "10 | 12 | 14 | 16",
        "Number of code values between black and white level.",
        choices=ss.ADC_BIT_DEPTHS,
    )
    black_level: int | None = _param(
        None,
        "Simulated sensor offset in code values.",
        "integer below white_level, or empty for 2^(bits-5) (512 at 14 bits)",
        "Keeps negative read noise representable.",
    )
    white_level: int | None = _param(
        None,
        "Simulated saturation code.",
        "integer up to 2^bits - 1, or empty for 2^bits - 1",
        "Signals above it clip, like a saturated photosite.",
    )
    analog_gain: float = _param(
        1.0,
        "Exposure/gain simulation (ISO multiplier over base gain).",
        "1 to 64",
        "Same brightness from 1/gain as many electrons, so relative shot noise grows with gain.",
        warning="High gain with shot noise on gives visibly noisy, lower-information mosaics.",
        bounds=(1.0, 64.0),
    )
    exposure_scale: float = _param(
        0.8,
        "Synthetic exposure scaling: camera-native signal of a working-space white on the channel with the "
        "smallest white-balance multiplier, as a fraction of the ADC range.",
        "0.01 to 1",
        "Lower values leave headroom for values above 1.0 in the master and push shadows toward the black level.",
        bounds=(0.01, 1.0),
    )
    full_well_electrons: float = _param(
        45000.0,
        "Full-well capacity of the hypothetical sensor at base gain.",
        "100 to 10,000,000",
        "Sets the shot-noise level: relative noise at full scale is 1/sqrt(full well / gain).",
        bounds=(100.0, 1e7),
    )
    shot_noise_enabled: bool = _param(
        False,
        "Synthetic Poisson shot noise (PRD 14.4).",
        "true | false",
        "Adds photon noise from the stated hypothetical sensor model.",
        warning="Simulated noise is not evidence of a physical exposure.",
    )
    read_noise_enabled: bool = _param(
        False,
        "Synthetic Gaussian read noise (PRD 14.4).",
        "true | false",
        "Adds signal-independent electronic noise.",
    )
    read_noise_std: float = _param(
        2.5,
        "Read noise amplitude in electrons RMS (input-referred), illustrative for the hypothetical sensor.",
        "0 to 1000",
        "Dominates in deep shadows.",
        bounds=(0.0, 1000.0),
    )
    color_matrix: tuple = _param(
        ss.DEFAULT_COLOR_MATRIX,
        "Explicit assumed camera-to-reference matrix: white-balanced camera RGB -> linear Rec.709. Rows sum to 1.",
        "3 x 3 list of numbers, invertible, rows summing to 1",
        "Its inverse is the simulated camera colour response; the DNG stores the matching ColorMatrix1.",
        warning="An assumption, not a measured camera profile.",
    )
    white_balance_multipliers: tuple = _param(
        ss.DEFAULT_WHITE_BALANCE,
        "Channel gains (R, G, B) that neutralise the assumed D65 illuminant.",
        "three numbers between 0.1 and 10",
        "Camera-native channels are divided by these; the DNG stores 1 / gains as AsShotNeutral.",
    )
    synthetic_raw_label: str = _param(
        ss.SYNTHETIC_RAW_LABEL,
        "Required provenance marker stored in every synthetic output (PRD 14.4, 14.6).",
        "text containing 'synthetic'",
        "Identifies every mosaic, DNG and report as simulated.",
    )
    demosaic_methods: tuple = _param(
        ss.DEMOSAIC_METHODS,
        "Demosaic methods for the E08 round trip.",
        "any of bilinear, malvar",
        "Each develops the mosaic back to RGB and is measured against the input.",
        choices=ss.DEMOSAIC_METHODS,
    )
    libraw_roundtrip: bool = _param(
        True,
        "Also develop the DNG with LibRaw (AHD demosaic) and measure it.",
        "true | false",
        "Shows what a RAW processor makes of the synthetic DNG; needs export_synthetic_dng.",
    )
    export_synthetic_dng: bool = _param(
        False,
        "Write the research DNG (PRD 6.1, 14.6). It must pass structural and LibRaw validation.",
        "true | false",
        "Adds a synthetic mosaic DNG; a file failing validation stops the run.",
        warning="Research output only; it is never camera RAW.",
    )
    export_mosaic_tiff: bool = _param(
        True,
        "Write the simulated CFA data as a 16-bit greyscale TIFF with provenance.",
        "true | false",
        "Keeps the exact mosaic for later analysis without the DNG container.",
    )
    bit_depth_study: bool = _param(
        True,
        "Run the PRD 14.5 / E01 bit-depth experiments on the input.",
        "true | false",
        "Compares 8-bit, 16-bit integer, half and float representations and the mosaic quantisation.",
    )
    bit_depth_write_files: bool = _param(
        True,
        "Write each bit-depth representation to disk to measure real file sizes.",
        "true | false",
        "Adds four full-size files (PNG, TIFF, two EXRs) under the run's bit_depth folder.",
        warning="At 50 MP this writes about 1 GB.",
    )
    edit_stops: float = _param(
        3.0,
        "Strength of the identical tonal edit applied to every representation (exposure push, in stops).",
        "0.5 to 8",
        "Larger pushes expose quantisation in the shadows more strongly.",
        bounds=(0.5, 8.0),
    )
    transform_cycles: int = _param(
        5,
        "Repeated forward/inverse transformations for the information-preservation test.",
        "1 to 100",
        "More cycles accumulate more re-quantisation error in integer representations.",
        bounds=(1, 100),
    )
    metric_tile_size: int = _param(
        1024,
        "Side of the full-resolution tiles sampled for round-trip metrics.",
        "128 to 4096",
        "Images up to (tile x grid)^2 pixels are measured whole.",
        bounds=(128, 4096),
    )
    metric_tile_grid: int = _param(
        3,
        "Tiles per side of the evenly spaced metric sampling grid.",
        "1 to 6",
        "grid^2 tiles are assembled into one sample image for SSIM, edge and delta E metrics.",
        bounds=(1, 6),
    )
    crop_size: int = _param(
        256,
        "Side of the visual comparison crops at full input resolution.",
        "32 to 2048",
        "Crops show the CFA, input and developed images at 100%.",
        bounds=(32, 2048),
    )
    crop_count: int = _param(
        3,
        "Number of visual comparison crops (centre plus the most detailed regions).",
        "1 to 12",
        "More crops cover more scene content.",
        bounds=(1, 12),
    )
    review_max_edge: int = _param(
        2048,
        "Longest edge of full-frame review images.",
        "256 to 8192",
        "Larger previews grow the report.",
        bounds=(256, 8192),
    )
    random_seed: int = _param(
        42,
        "Seed for the shot and read noise generators.",
        "any integer",
        "Same seed, same mosaic.",
    )
    output_root: str = _param(
        "",
        "Folder for mosaics, DNGs and bit-depth files.",
        "path; empty means <project>/outputs",
        "Point this at Google Drive to keep outputs after the Colab VM resets.",
    )
    reports_root: str = _param(
        "",
        "Folder for reports.",
        "path; empty means <project>/reports",
        "Reports are small and safe to download or archive.",
    )

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if f.name in ("black_level", "white_level"):
                if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                    raise ValueError(f"{f.name} must be an integer or empty, got {value!r}")
                continue
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
        self.sensor_model()  # cross-parameter checks (levels, matrix, label)

    def sensor_model(self) -> ss.SensorModel:
        return ss.SensorModel(
            cfa_pattern=self.cfa_pattern,
            adc_bit_depth=self.adc_bit_depth,
            black_level=self.black_level,
            white_level=self.white_level,
            analog_gain=float(self.analog_gain),
            exposure_scale=float(self.exposure_scale),
            full_well_electrons=float(self.full_well_electrons),
            shot_noise_enabled=self.shot_noise_enabled,
            read_noise_enabled=self.read_noise_enabled,
            read_noise_std=float(self.read_noise_std),
            color_matrix=self.color_matrix,
            white_balance_multipliers=self.white_balance_multipliers,
            synthetic_raw_label=self.synthetic_raw_label,
        )

    def replace(self, **overrides: Any) -> SensorResearchConfig:
        return from_dict({**self.to_dict(), **overrides})

    def to_dict(self) -> dict:
        data = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        data["color_matrix"] = [list(row) for row in self.color_matrix]
        for name in ("white_balance_multipliers", "demosaic_methods"):
            data[name] = list(data[name])
        return data


_FLOAT_FIELDS = (
    "analog_gain",
    "exposure_scale",
    "full_well_electrons",
    "read_noise_std",
    "edit_stops",
)


def _check_type(f: dataclasses.Field, value: Any) -> None:
    _check_base_type(f, value)
    if f.name == "color_matrix":
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("color_matrix must be a 3 x 3 list of numbers")
        values = [v for row in value for v in row]
    elif f.name == "white_balance_multipliers":
        values = list(value)
        if len(values) != 3:
            raise ValueError("white_balance_multipliers must hold three numbers")
    else:
        return
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError(f"{f.name} must contain only numbers, got {value!r}")


def from_dict(values: dict[str, Any]) -> SensorResearchConfig:
    known = {f.name for f in dataclasses.fields(SensorResearchConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown configuration keys: {', '.join(unknown)}")
    normalized = dict(values)
    if "color_matrix" in normalized:
        normalized["color_matrix"] = tuple(tuple(float(v) for v in row) for row in normalized["color_matrix"])
    for name in ("white_balance_multipliers", "demosaic_methods"):
        if name in normalized:
            normalized[name] = tuple(normalized[name])
    if "white_balance_multipliers" in normalized:
        normalized["white_balance_multipliers"] = tuple(
            float(v) if isinstance(v, int) and not isinstance(v, bool) else v
            for v in normalized["white_balance_multipliers"]
        )
    for name in _FLOAT_FIELDS:
        if isinstance(normalized.get(name), int) and not isinstance(normalized.get(name), bool):
            normalized[name] = float(normalized[name])
    for name in ("output_root", "reports_root"):
        if normalized.get(name) is None:
            normalized[name] = ""
    return SensorResearchConfig(**normalized)


def load_sensor_config(path: str | Path | None = None, **overrides: Any) -> SensorResearchConfig:
    """Load defaults, then a YAML file (if given), then keyword overrides."""
    values: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        values.update(loaded)
    values.update(overrides)
    return from_dict(values)


def parameter_table(config: SensorResearchConfig | None = None) -> list[dict]:
    """Rows of (name, default, current, accepted, description, effect, warning) for display (PRD 22.2)."""
    config = config or SensorResearchConfig()
    defaults = SensorResearchConfig().to_dict()
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
        for f in dataclasses.fields(SensorResearchConfig)
    ]
