"""Notebook 06 configuration (PRD Phase 6): ablation, controlled reference test, run aggregation, final report.

``configs/evaluation_default.yaml`` names the universal profile whose human, night and universal settings the
evaluation reuses (so an ablation stage runs exactly the processing a notebook 01, 02 or 03 run would), then
four sections:

* ``ablation``: the PRD 31 experiment matrix and the E09 full-pipeline ablation on one input, at the scale a
  full 50 MP run of that input uses (a crop keeps it to minutes), plus the PRD 28.3 SR settings sweep.
* ``reference``: experiment E10, real high-resolution photographs degraded to the source scale and
  reconstructed, the only place reconstruction accuracy can be measured against ground truth.
* ``aggregation``: labels for the source images and the identical crops of the visual review sheet.
* ``report``: the provisional thresholds the final report uses to choose a PRD 33 outcome.

Every threshold here is provisional, to be calibrated experimentally (PRD 25.3).
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import sensor_simulation
from . import universal_config as uc

DEFAULT_PROFILE = "configs/evaluation_default.yaml"
SECTIONS = ("ablation", "reference", "aggregation", "report")
SCENES = ("auto", "human", "night", "general")
DEGRADATIONS = ("area", "camera", "ai_render")
# The captain's real inputs are 6336 x 2688 renders; a 50 MP master of them is a x1.7134 linear enlargement.
CAPTAIN_SOURCE = (6336, 2688)
CAPTAIN_SCALE = round(math.sqrt(50e6 / (CAPTAIN_SOURCE[0] * CAPTAIN_SOURCE[1])), 4)
SWEEP_KEYS = ("tile_size", "tile_overlap", "precision", "device")
SENSOR_KEYS = tuple(f.name for f in dataclasses.fields(sensor_simulation.SensorModel)) + ("demosaic",)


def _number(section: str, name: str, value: Any, low: float, high: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise ValueError(f"{section}.{name} must be a number from {low:g} to {high:g}, got {value!r}")


def _integer(section: str, name: str, value: Any, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{section}.{name} must be an integer from {low} to {high}, got {value!r}")


@dataclass(frozen=True)
class AblationParams:
    """The PRD 31 matrix and E09 ablation on one input (see :mod:`ablation` for every configuration).

    ``crop_size`` is the side of a square source crop processed at the scale a full run of the whole source to
    ``target_megapixels`` uses, so per-pixel behaviour matches the full run while the run takes minutes; 0
    processes the whole source at the full target (one SR pass per mode at full size, as long as notebook 00).
    ``crop_center`` is ``auto`` (the primary face, else the saliency centre of interest) or source ``[x, y]``.
    """

    scene: str = "auto"
    crop_size: int = 1536
    crop_center: str | tuple = "auto"
    target_megapixels: float = 50.0
    configurations: tuple = ()
    grain_amount: float = 0.008
    bloom_amount: float = 0.03
    sensor: dict = field(
        default_factory=lambda: {"shot_noise_enabled": True, "read_noise_enabled": True, "demosaic": "malvar"}
    )
    performance_sweep: tuple = ({"precision": "float16"}, {"tile_size": 256}, {"tile_overlap": 32})
    bit_depth_study: bool = True
    review_crop_count: int = 3
    review_crop_px: int = 256

    def __post_init__(self) -> None:
        from .ablation import CONFIGURATIONS, DEFAULT_MATRIX

        if self.scene not in SCENES:
            raise ValueError(f"ablation.scene must be one of {SCENES}, got {self.scene!r}")
        _integer("ablation", "crop_size", self.crop_size, 0, 20000)
        if 0 < self.crop_size < 64:
            raise ValueError("ablation.crop_size must be 0 (whole source) or at least 64 source pixels")
        centre = self.crop_center
        if not (centre == "auto" or (isinstance(centre, tuple) and len(centre) == 2)):
            raise ValueError(f"ablation.crop_center must be auto or [x, y] source pixels, got {centre!r}")
        _number("ablation", "target_megapixels", self.target_megapixels, 0.01, 400.0)
        if not self.configurations:
            object.__setattr__(self, "configurations", DEFAULT_MATRIX)
        unknown = [name for name in self.configurations if name not in CONFIGURATIONS]
        if unknown:
            raise ValueError(f"unknown ablation configurations {unknown}; known: {sorted(CONFIGURATIONS)}")
        if len(set(self.configurations)) != len(self.configurations):
            raise ValueError("ablation.configurations lists a configuration twice")
        _number("ablation", "grain_amount", self.grain_amount, 0.0, 0.05)
        _number("ablation", "bloom_amount", self.bloom_amount, 0.0, 0.2)
        bad = sorted(set(self.sensor) - set(SENSOR_KEYS))
        if bad:
            raise ValueError(f"unknown ablation.sensor keys {bad}; known: {list(SENSOR_KEYS)}")
        self.sensor_model()  # validates the sensor parameters
        for entry in self.performance_sweep:
            if not isinstance(entry, dict) or not entry or set(entry) - set(SWEEP_KEYS):
                raise ValueError(f"ablation.performance_sweep entries are mappings of {SWEEP_KEYS}, got {entry!r}")
        if not isinstance(self.bit_depth_study, bool):
            raise ValueError("ablation.bit_depth_study must be true or false")
        _integer("ablation", "review_crop_count", self.review_crop_count, 1, 8)
        _integer("ablation", "review_crop_px", self.review_crop_px, 32, 1024)

    def sensor_model(self) -> sensor_simulation.SensorModel:
        values = {k: v for k, v in self.sensor.items() if k != "demosaic"}
        for key in ("color_matrix", "white_balance_multipliers"):
            if key in values:
                values[key] = tuple(tuple(row) if isinstance(row, list) else row for row in values[key])
        return sensor_simulation.SensorModel(**values)

    @property
    def demosaic(self) -> str:
        method = self.sensor.get("demosaic", "malvar")
        if method not in sensor_simulation.DEMOSAIC_METHODS:
            raise ValueError(f"ablation.sensor.demosaic must be one of {sensor_simulation.DEMOSAIC_METHODS}")
        return method

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["configurations"] = list(self.configurations)
        data["crop_center"] = self.crop_center if self.crop_center == "auto" else list(self.crop_center)
        data["performance_sweep"] = [dict(entry) for entry in self.performance_sweep]
        return data


@dataclass(frozen=True)
class ReferenceParams:
    """Experiment E10 (see :mod:`reference_test`).

    ``scales`` are linear enlargement factors: each reference (centre-cropped to ``crop_megapixels``, 0 keeps
    the whole photograph) is degraded to 1/scale of its size by every degradation model and reconstructed back
    to the reference size. The default scale is the one a 50 MP master of the captain's 6336 x 2688 inputs uses.
    """

    references: tuple = ("portrait_holi", "night_forest_path")
    degradations: tuple = DEGRADATIONS
    scales: tuple = (CAPTAIN_SCALE,)
    crop_megapixels: float = 12.0
    configurations: tuple = (
        "bicubic",
        "lanczos",
        "sr_conservative",
        "sr_balanced",
        "specialized",
        "full",
        "full_no_sensor",
        "full_no_grain",
    )
    optical_blur_sigma: float = 0.5
    jpeg_quality: int = 95
    ai_smoothing: float = 0.5
    ai_sharpening: float = 0.4
    cache_dir: str = ""

    def __post_init__(self) -> None:
        from .ablation import CONFIGURATIONS
        from .reference_test import REFERENCES

        if not self.references:
            raise ValueError("reference.references must name at least one reference photograph")
        for ref in self.references:
            if isinstance(ref, str):
                if ref not in REFERENCES:
                    raise ValueError(f"unknown reference {ref!r}; known: {sorted(REFERENCES)}")
            elif not (isinstance(ref, dict) and {"name", "title", "sha1", "scene"} <= set(ref)):
                raise ValueError("a custom reference is a mapping with name, title (File:...), sha1 and scene")
        bad = [d for d in self.degradations if d not in DEGRADATIONS]
        if not self.degradations or bad:
            raise ValueError(f"reference.degradations must list some of {DEGRADATIONS}, got {list(self.degradations)}")
        if not self.scales:
            raise ValueError("reference.scales must list at least one scale")
        for scale in self.scales:
            _number("reference", "scales", scale, 1.05, 8.0)
        _number("reference", "crop_megapixels", self.crop_megapixels, 0.0, 400.0)
        if 0 < self.crop_megapixels < 0.25:
            raise ValueError("reference.crop_megapixels must be 0 (whole photograph) or at least 0.25")
        unknown = [name for name in self.configurations if name not in CONFIGURATIONS]
        if not self.configurations or unknown:
            raise ValueError(f"reference.configurations must name known configurations, unknown: {unknown}")
        if "lanczos" not in self.configurations:
            raise ValueError("reference.configurations must include lanczos, the baseline every gain is measured from")
        _number("reference", "optical_blur_sigma", self.optical_blur_sigma, 0.0, 3.0)
        _integer("reference", "jpeg_quality", self.jpeg_quality, 50, 100)
        _number("reference", "ai_smoothing", self.ai_smoothing, 0.0, 1.0)
        _number("reference", "ai_sharpening", self.ai_sharpening, 0.0, 2.0)

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        for key in ("references", "degradations", "scales", "configurations"):
            data[key] = list(getattr(self, key))
        return data


@dataclass(frozen=True)
class AggregationParams:
    """How collected runs are labelled and compared.

    ``labels`` maps a source SHA-256 prefix to a readable name. ``review_crops`` are normalised
    ``[x, y, width, height]`` boxes (fractions of the frame) cut identically from every run of one source; when
    empty, the crops the runs themselves recorded are used, at most ``review_crop_count`` of them.
    """

    labels: dict = field(default_factory=dict)
    review_crops: tuple = ()
    review_crop_count: int = 4
    review_crop_px: int = 256

    def __post_init__(self) -> None:
        for key, value in self.labels.items():
            if not isinstance(key, str) or len(key) < 8 or not isinstance(value, str):
                raise ValueError("aggregation.labels maps SHA-256 prefixes (8+ hex digits) to names")
        for box in self.review_crops:
            if len(box) != 4 or not all(0 <= v <= 1 for v in box) or box[0] + box[2] > 1 or box[1] + box[3] > 1:
                raise ValueError(f"aggregation.review_crops boxes are [x, y, width, height] fractions, got {box!r}")
        _integer("aggregation", "review_crop_count", self.review_crop_count, 1, 12)
        _integer("aggregation", "review_crop_px", self.review_crop_px, 32, 1024)

    def label(self, sha256: str) -> str | None:
        return next((name for prefix, name in self.labels.items() if sha256.startswith(prefix)), None)

    def to_dict(self) -> dict:
        return {**dataclasses.asdict(self), "review_crops": [list(b) for b in self.review_crops]}


@dataclass(frozen=True)
class ReportParams:
    """Provisional PRD 33 outcome thresholds, applied to the E10 reference test (gains over Lanczos).

    Outcome A needs the best reconstruction to beat Lanczos against ground truth by ``a_min_psnr_gain_db`` and
    to bring the above-source-Nyquist error down to ``a_max_hf_nmse`` (1 means no better than adding no detail),
    plus a blind visual review that prefers it. Outcome C applies when the gain is below ``c_max_psnr_gain_db``
    and the above-Nyquist error stays at or above ``c_min_hf_nmse``. Anything between is Outcome B.
    ``quality_tolerance_db`` treats configurations within that PSNR of the best as equal quality when choosing
    the best quality-to-runtime configuration.
    """

    a_min_psnr_gain_db: float = 1.0
    a_max_hf_nmse: float = 0.7
    c_max_psnr_gain_db: float = 0.25
    c_min_hf_nmse: float = 0.95
    quality_tolerance_db: float = 0.1

    def __post_init__(self) -> None:
        _number("report", "a_min_psnr_gain_db", self.a_min_psnr_gain_db, 0.0, 20.0)
        _number("report", "a_max_hf_nmse", self.a_max_hf_nmse, 0.0, 2.0)
        _number("report", "c_max_psnr_gain_db", self.c_max_psnr_gain_db, 0.0, 20.0)
        _number("report", "c_min_hf_nmse", self.c_min_hf_nmse, 0.0, 2.0)
        _number("report", "quality_tolerance_db", self.quality_tolerance_db, 0.0, 5.0)
        if self.c_max_psnr_gain_db > self.a_min_psnr_gain_db or self.a_max_hf_nmse > self.c_min_hf_nmse:
            raise ValueError("report thresholds overlap: outcome C must sit strictly below outcome A")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class EvaluationConfig:
    """A validated evaluation profile; build it with :func:`load_evaluation_config`."""

    universal: uc.UniversalConfig
    ablation: AblationParams = field(default_factory=AblationParams)
    reference: ReferenceParams = field(default_factory=ReferenceParams)
    aggregation: AggregationParams = field(default_factory=AggregationParams)
    report: ReportParams = field(default_factory=ReportParams)
    universal_profile: str = "configs/universal_default.yaml"

    @property
    def experiment(self):
        return self.universal.experiment

    def to_dict(self) -> dict:
        return {
            "universal_profile": self.universal_profile,
            "universal": self.universal.to_dict(),
            "ablation": self.ablation.to_dict(),
            "reference": self.reference.to_dict(),
            "aggregation": self.aggregation.to_dict(),
            "report": self.report.to_dict(),
        }


def load_evaluation_config(
    path: str | Path | None = None, project_root: str | Path | None = None, **overrides: Any
) -> EvaluationConfig:
    """Load an evaluation profile such as ``configs/evaluation_default.yaml``.

    ``universal_profile`` (relative to ``project_root``, by default the repository) and ``universal_overrides``
    build the universal profile; a section override such as ``ablation={"crop_size": 1024}`` updates single
    keys of that section, and ``universal_overrides`` merges into the file's.
    """
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
    values: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must hold a mapping")
        values = loaded
    values = copy.deepcopy(values)
    for key, value in overrides.items():
        if key in (*SECTIONS, "universal_overrides") and isinstance(value, dict):
            values[key] = {**(values.get(key) or {}), **value}
        else:
            values[key] = value
    unknown = sorted(set(values) - {*SECTIONS, "universal_profile", "universal_overrides"})
    if unknown:
        raise ValueError(f"unknown evaluation keys: {', '.join(unknown)}")
    profile = values.get("universal_profile") or EvaluationConfig.universal_profile
    profile_path = Path(profile) if Path(profile).is_absolute() else root / profile
    universal = uc.load_universal_config(profile_path, **(values.get("universal_overrides") or {}))
    sections = {}
    for name, cls in (
        ("ablation", AblationParams),
        ("reference", ReferenceParams),
        ("aggregation", AggregationParams),
        ("report", ReportParams),
    ):
        given = values.get(name) or {}
        if not isinstance(given, dict):
            raise ValueError(f"{name} must be a mapping")
        known = {f.name for f in dataclasses.fields(cls)}
        bad = sorted(set(given) - known)
        if bad:
            raise ValueError(f"unknown keys in {name}: {', '.join(bad)}")
        sections[name] = cls(**{k: _tuples(k, v) for k, v in given.items()})
    return EvaluationConfig(universal, **sections, universal_profile=str(profile))


def parameter_rows(config: EvaluationConfig) -> list[dict]:
    """Every evaluation setting as (parameter, value, default) rows for the notebook's configuration table."""
    defaults = EvaluationConfig(config.universal).to_dict()
    current = config.to_dict()
    rows = []
    for section in SECTIONS:
        for key, value in current[section].items():
            rows.append({"parameter": f"{section}.{key}", "current": value, "default": defaults[section].get(key)})
    return rows


def _tuples(key: str, value: Any) -> Any:
    """YAML lists become the tuples the frozen dataclasses hold (mappings stay mappings)."""
    if key in ("sensor", "labels"):
        return dict(value or {})
    if key == "crop_center" and isinstance(value, list):
        return tuple(value)
    if key == "review_crops":
        return tuple(tuple(box) for box in value or ())
    if key == "performance_sweep":
        return tuple(dict(entry) for entry in value or ())
    if isinstance(value, list):
        return tuple(value)
    return value
