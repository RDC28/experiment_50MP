"""Immutable synthetic camera state shared by all source-resolution runs.

These priors describe a hypothetical exposure, never recovered Sony electronics.
No images, weights or frame-sized arrays are needed to construct this state.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Literal

FieldMode = Literal["AUTO", "ESTIMATE", "FABRICATE", "USER"]
Provenance = Literal["OBSERVED", "ESTIMATED", "DERIVED", "PRIOR_SAMPLED", "SYNTHETIC", "USER_SET", "UNSPECIFIED"]


@dataclass(frozen=True)
class CameraField:
    """JSON-ready registry record with explicit epistemic status."""

    name: str
    value: object
    units: str | None
    mode: FieldMode
    provenance: Provenance
    confidence: float | None
    posterior: dict | None
    method: str
    validation: str

    def __post_init__(self) -> None:
        if self.mode not in ("AUTO", "ESTIMATE", "FABRICATE", "USER"):
            raise ValueError("Unknown field mode")
        if self.provenance not in (
            "OBSERVED",
            "ESTIMATED",
            "DERIVED",
            "PRIOR_SAMPLED",
            "SYNTHETIC",
            "USER_SET",
            "UNSPECIFIED",
        ):
            raise ValueError("Unknown provenance")
        if not self.name or not self.method or not self.validation:
            raise ValueError("Field name, method and validation are required")
        if self.confidence is not None and (
            isinstance(self.confidence, bool) or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("Confidence must be null or a finite probability")
        json.dumps(asdict(self), allow_nan=False)


@dataclass(frozen=True)
class CameraHypothesis:
    """One seeded camera draw; tuples and scalars prevent cross-run mutation.

    ISO determines analogue gain relative to base ISO 100. Conversion gain is
    derived at base ISO so full well maps to the available ADC range. Higher ISO
    consequently saturates sooner. Digital gain is unity. NoiseProfile describes
    normalized black-subtracted temporal signal before clipping, excluding FPN.
    """

    seed: int = 42
    adc_bits: int = 14
    black_level: int = 512
    full_well_e: float = 52000.0
    read_noise_e: float = 3.0
    iso: int = 100
    shutter_s: float = 1 / 125
    aperture: float = 5.6
    cfa_pattern: str = "RGGB"
    shot_noise: bool = True
    read_noise: bool = True

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be a uint64 integer")
        if type(self.adc_bits) is not int or self.adc_bits != 14:
            raise ValueError("This Sony-like hypothesis requires 14-bit ADC in uint16 storage")
        if type(self.black_level) is not int or not 0 <= self.black_level < self.white_level:
            raise ValueError("Black level must fit below white level")
        for name in ("full_well_e", "read_noise_e", "shutter_s", "aperture"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name != "read_noise_e" and value == 0):
                raise ValueError(f"{name} outside physical domain")
        if type(self.iso) is not int or not 100 <= self.iso <= 32000:
            raise ValueError("ISO must lie in the synthetic standard range 100..32000")
        if self.cfa_pattern not in ("RGGB", "BGGR", "GRBG", "GBRG"):
            raise ValueError("Expected a Bayer CFA pattern")
        if type(self.shot_noise) is not bool or type(self.read_noise) is not bool:
            raise ValueError("Noise switches must be booleans")

    @classmethod
    def sample(cls, seed: int = 42) -> CameraHypothesis:
        """Sample declared electronics priors once, without consuming global RNG.

        Exposure defaults remain explicit hypotheses, not inferred EXIF. Full
        well uses a truncated log-normal prior; read noise a uniform prior.
        """
        cls(seed=seed)  # Validate before creating the local generator.
        rng = random.Random(seed)
        full_well = rng.lognormvariate(math.log(52000), 0.2)
        while not 20000 <= full_well <= 100000:
            full_well = rng.lognormvariate(math.log(52000), 0.2)
        return cls(seed=seed, full_well_e=full_well, read_noise_e=rng.uniform(1, 5))

    @property
    def white_level(self) -> int:
        return (1 << self.adc_bits) - 1

    @property
    def conversion_gain_e_per_dn(self) -> float:
        return self.full_well_e / (self.white_level - self.black_level)

    @property
    def analogue_gain(self) -> float:
        return self.iso / 100

    @property
    def saturation_e(self) -> float:
        return self.full_well_e / self.analogue_gain

    @property
    def noise_profile(self) -> tuple[float, ...] | None:
        if not self.shot_noise:
            return None  # DNG requires a positive shot slope; never fabricate it.
        scale = self.analogue_gain / self.full_well_e
        read_sigma = self.read_noise_e if self.read_noise else 0.0
        offset = (read_sigma * scale) ** 2 + 1 / (12 * (self.white_level - self.black_level) ** 2)
        return (scale, offset) * 3

    def stream_seed(self, stage: str) -> int:
        """Stable independent stream ID, deliberately independent of input size."""
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("A nonempty stage name is required")
        digest = hashlib.sha256(f"prd-v5:{self.seed}:{stage}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def registry(self) -> dict[str, dict]:
        """Return fresh JSON records; callers cannot mutate the hypothesis.

        The initial MVP subset is intentionally not the complete PRD registry.
        No statistical validation is implied by arithmetic validation rules.
        """
        result = {}

        def add(name, value, units, provenance, method, validation, mode="AUTO"):
            result[name] = asdict(CameraField(name, value, units, mode, provenance, None, None, method, validation))

        for name, units in (
            ("seed", None),
            ("adc_bits", "bits"),
            ("black_level", "DN"),
            ("full_well_e", "electrons"),
            ("read_noise_e", "electrons RMS"),
            ("iso", "ISO"),
            ("shutter_s", "s"),
            ("aperture", "f-number"),
            ("cfa_pattern", None),
            ("shot_noise", None),
            ("read_noise", None),
        ):
            add(
                name,
                getattr(self, name),
                units,
                "SYNTHETIC",
                "Explicit synthetic hypothesis; not measured",
                "CameraHypothesis physical-domain checks",
            )
        for name, value, units in (
            ("white_level", self.white_level, "DN"),
            ("conversion_gain_e_per_dn", self.conversion_gain_e_per_dn, "electrons/DN"),
            ("analogue_gain", self.analogue_gain, "ratio"),
            ("saturation_e", self.saturation_e, "electrons"),
            ("noise_profile", self.noise_profile, "normalized signal variance"),
        ):
            add(
                name,
                value,
                units,
                "DERIVED",
                "Derived from shared ADC/electron hypothesis",
                "Arithmetic consistency only; photon-transfer validation pending",
            )
        for name in ("gps", "serial_number", "timestamp", "shutter_count"):
            add(
                name,
                None,
                None,
                "UNSPECIFIED",
                "Unknown; user-only identity",
                "Must remain null without explicit user input",
                mode="USER",
            )
        return result
