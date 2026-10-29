"""Strict JSON configuration for the implemented analytic MVP only."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from prd_v5.analytic import AnalyticISP
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.pipeline import run_analytic


@dataclass(frozen=True)
class AnalyticRunConfig:
    """Portable explicit hypotheses, with conservative local memory admission.

    This is not the full PRD panel and never requests or substitutes learned
    inference. Paths are supplied at execution, outside the reusable config.
    """

    schema_version: int = 1
    camera: CameraHypothesis = field(default_factory=CameraHypothesis)
    isp: AnalyticISP = field(default_factory=AnalyticISP)
    target_megapixels: float = 50.0
    assume_unprofiled_srgb: bool = False
    strip_rows: int = 64
    max_decode_bytes: int = 256 * 1024 * 1024
    max_develop_bytes: int = 512 * 1024 * 1024

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported analytic configuration schema_version")
        if not isinstance(self.camera, CameraHypothesis) or not isinstance(self.isp, AnalyticISP):
            raise ValueError("Expected validated camera and ISP hypotheses")
        if (
            type(self.target_megapixels) not in (int, float)
            or not math.isfinite(self.target_megapixels)
            or self.target_megapixels <= 0
        ):
            raise ValueError("target_megapixels must be positive and finite")
        if type(self.assume_unprofiled_srgb) is not bool:
            raise ValueError("assume_unprofiled_srgb must be boolean")
        for name in ("strip_rows", "max_decode_bytes", "max_develop_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_json(self) -> str:
        """Serialize actual hypotheses rather than only their random seed."""
        return json.dumps(asdict(self), indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> AnalyticRunConfig:
        """Reject duplicate, unknown and nonfinite fields instead of ignoring them."""

        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate configuration field: {key}")
                result[key] = value
            return result

        def nonfinite(value):
            raise ValueError(f"Nonfinite JSON number: {value}")

        values = json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        try:
            for name, constructor in (("camera", CameraHypothesis), ("isp", AnalyticISP)):
                if name in values:
                    if not isinstance(values[name], dict):
                        raise ValueError(f"{name} must be an object")
                    values[name] = constructor(**values[name])
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid analytic configuration fields: {error}") from error

    def run(self, source_path: str | Path, output_dir: str | Path) -> dict:
        """Execute the bounded analytic pipeline with exactly these settings."""
        return run_analytic(
            source_path,
            output_dir,
            self.camera,
            self.isp,
            target_megapixels=self.target_megapixels,
            assume_unprofiled_srgb=self.assume_unprofiled_srgb,
            strip_rows=self.strip_rows,
            max_decode_bytes=self.max_decode_bytes,
            max_develop_bytes=self.max_develop_bytes,
        )
