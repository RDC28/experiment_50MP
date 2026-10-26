"""Typed configuration for PRD_v3 required controls and toggle acceptance rules."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Literal, get_args, get_type_hints


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated, JSON-compatible experiment settings; construction performs no I/O.

    Choice values without PRD alternatives stay restricted to the documented mode.
    Calibration paths declare assets for later validation by CameraProfileManager.
    """

    TARGET_PROFILE: Literal["SONY_A1II_LIKE", "CINEMA_50MP_VIRTUAL", "VIRTUAL_16BIT_SENSOR"] = "SONY_A1II_LIKE"
    OUTPUT_GEOMETRY: Literal["SONY_A1II_NATIVE", "CINEMA_50MP_VIRTUAL"] = "SONY_A1II_NATIVE"
    SENSOR_FIT_MODE: Literal["OUTPAINT_VERTICAL", "CROP_HORIZONTAL", "FIT_WITH_SENSOR_MARGINS"] = "OUTPAINT_VERTICAL"
    SR_BACKEND: Literal["SWINIR", "HAT", "REAL_ESRGAN", "SUPIR", "LANCZOS"] = "SWINIR"
    SR_MODE: Literal["FIDELITY"] = "FIDELITY"
    GENERATIVE_DETAIL_STRENGTH: float = 0.0
    HDR_ENABLED: bool = True
    HDR_BACKEND: Literal["DETERMINISTIC", "EXPANDDIFF", "VID2HDR"] = "DETERMINISTIC"
    HDR_HIGHLIGHT_EV: float = 3.0
    HDR_SHADOW_EV: float = 3.0
    INVERSE_ISP: Literal["PHYSICS"] = "PHYSICS"
    USE_RAW_FLOW: bool = False
    USE_SPIRALDIFF: bool = False
    CAMERA_PROFILE_SOURCE: Literal["IMAGE_ONLY", "REFERENCE_RAW", "FULL_CALIBRATION"] = "IMAGE_ONLY"
    SPECTRAL_MODE: Literal["MATRIX_ONLY", "REFERENCE_PROFILE", "CALIBRATED_CURVES"] = "MATRIX_ONLY"
    EXPOSURE_MODE: Literal["POSTERIOR", "MANUAL", "PROFILE_DEFAULT"] = "POSTERIOR"
    ADC_BITS: int = 14
    EXPERIMENTAL_16BIT_SENSOR: bool = False
    SHOT_NOISE: bool = True
    READ_NOISE: bool = True
    PRNU: bool = False
    DSNU: bool = False
    ROW_FPN: bool = False
    COLUMN_FPN: bool = False
    DARK_CURRENT: bool = False
    HOT_PIXELS: bool = False
    DEAD_PIXELS: bool = False
    OPTICAL_SIMULATION: bool = False
    DISTORTION: bool = False
    VIGNETTING: bool = False
    CHROMATIC_ABERRATION: bool = False
    DIFFRACTION: bool = False
    DEFOCUS: bool = False
    DNG_STORAGE_BITS: int = 16
    DNG_COMPRESSION: Literal["UNCOMPRESSED", "LOSSLESS"] = "UNCOMPRESSED"
    GENERATE_SYNTHETIC_EXIF: bool = True
    FABRICATE_UNOBSERVABLE_EXIF: bool = False
    WRITE_DNG: bool = True
    WRITE_TIFF16: bool = True
    WRITE_EXR: bool = True
    WRITE_METADATA_JSON: bool = True
    WRITE_SONY_ARW_EMULATION_JSON: bool = True
    WRITE_VALIDATION_REPORT: bool = True
    WRITE_C2PA: bool = True
    SCENE_SEED: int = 42
    SENSOR_NOISE_SEED: int = 42001
    SENSOR_BODY_SEED: int = 42002
    N_SENSOR_REALISATIONS: int = 1
    FAST_DEV_RUN: bool = False
    INPUT_MODE: Literal["UPLOAD", "PATH", "DRIVE"] = "UPLOAD"
    INPUT_PATH: str | None = None
    DRIVE_MOUNT_PATH: str = "/content/drive"
    CALIBRATION_ASSETS: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        for field in fields(self):
            value = getattr(self, field.name)
            hint = hints[field.name]
            choices = get_args(hint) if getattr(hint, "__origin__", None) is Literal else ()
            if choices and value not in choices:
                raise ValueError(f"{field.name} must be one of {choices}")
            if hint is bool and type(value) is not bool:
                raise ValueError(f"{field.name} must be a boolean")
            if hint is int and (type(value) is not int or value < 0):
                raise ValueError(f"{field.name} must be a nonnegative integer")
            if hint is float and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f"{field.name} must be a finite number")
        if not 0 <= self.GENERATIVE_DETAIL_STRENGTH <= 1:
            raise ValueError("GENERATIVE_DETAIL_STRENGTH must be in [0, 1]")
        if self.HDR_HIGHLIGHT_EV < 0 or self.HDR_SHADOW_EV < 0:
            raise ValueError("HDR EV values must be nonnegative")
        if self.ADC_BITS not in (14, 16):
            raise ValueError("ADC_BITS must be 14 or 16")
        if self.DNG_STORAGE_BITS != 16:
            raise ValueError("DNG_STORAGE_BITS must be 16")
        if self.N_SENSOR_REALISATIONS < 1:
            raise ValueError("N_SENSOR_REALISATIONS must be positive")
        if self.INPUT_PATH is not None and (not isinstance(self.INPUT_PATH, str) or not self.INPUT_PATH.strip()):
            raise ValueError("INPUT_PATH must be a nonempty string or None")
        if self.INPUT_MODE in ("PATH", "DRIVE") and self.INPUT_PATH is None:
            raise ValueError("PATH and DRIVE input require INPUT_PATH")
        if not isinstance(self.DRIVE_MOUNT_PATH, str) or not self.DRIVE_MOUNT_PATH.strip():
            raise ValueError("DRIVE_MOUNT_PATH must be a nonempty string")
        if not isinstance(self.CALIBRATION_ASSETS, tuple) or any(
            not isinstance(path, str) or not path.strip() for path in self.CALIBRATION_ASSETS
        ):
            raise ValueError("CALIBRATION_ASSETS must be a tuple of nonempty paths")
        if self.SPECTRAL_MODE == "CALIBRATED_CURVES" and not self.CALIBRATION_ASSETS:
            raise ValueError("CALIBRATED_CURVES requires CALIBRATION_ASSETS")
        if self.OUTPUT_GEOMETRY == "CINEMA_50MP_VIRTUAL":
            object.__setattr__(self, "TARGET_PROFILE", "CINEMA_50MP_VIRTUAL")
        if self.EXPERIMENTAL_16BIT_SENSOR or self.ADC_BITS == 16:
            object.__setattr__(self, "ADC_BITS", 16)
            object.__setattr__(self, "TARGET_PROFILE", "VIRTUAL_16BIT_SENSOR")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-native settings for config.json."""
        result = asdict(self)
        result["CALIBRATION_ASSETS"] = list(self.CALIBRATION_ASSETS)
        return result

    @classmethod
    def from_dict(cls, settings: dict[str, object]) -> ExperimentConfig:
        """Restore validated settings, rejecting unknown configuration keys."""
        values = dict(settings)
        if isinstance(values.get("CALIBRATION_ASSETS"), list):
            values["CALIBRATION_ASSETS"] = tuple(values["CALIBRATION_ASSETS"])
        return cls(**values)
