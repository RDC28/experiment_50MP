"""Strict JSON configuration for the implemented analytic MVP only."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from prd_v5.analytic import AnalyticISP
from prd_v5.fidelity import SRCheckpoint
from prd_v5.geometry import GeometryPlanner
from prd_v5.hypothesis import CameraField, CameraHypothesis
from prd_v5.learned import ResidualAdapter
from prd_v5.optics import Vignetting
from prd_v5.sensor import SensorEffects


@dataclass(frozen=True)
class AnalyticRunConfig:
    """Portable explicit hypotheses, with conservative local memory admission.

    This is not the full PRD panel and never requests or substitutes learned
    inference. Paths are supplied at execution, outside the reusable config.
    """

    sensor_effects: SensorEffects = field(default_factory=SensorEffects)
    schema_version: int = 1
    camera: CameraHypothesis = field(default_factory=CameraHypothesis)
    isp: AnalyticISP = field(default_factory=AnalyticISP)
    geometry_mode: str = "PRESERVE_SOURCE_ASPECT_50MP"
    native_fit_policy: str = "LETTERBOX_SENSOR_MARGIN"
    target_megapixels: float = 50.0
    assume_unprofiled_srgb: bool = False
    strip_rows: int = 64
    max_decode_bytes: int = 256 * 1024 * 1024
    max_develop_bytes: int = 512 * 1024 * 1024
    vignetting: Vignetting = field(default_factory=Vignetting)
    accept_experimental_optics: bool = False

    def __post_init__(self):
        if not isinstance(self.vignetting, Vignetting):
            raise ValueError("Expected validated Vignetting")
        if type(self.accept_experimental_optics) is not bool:
            raise ValueError("accept_experimental_optics must be boolean")
        if self.vignetting.corner_strength and not self.accept_experimental_optics:
            raise ValueError("Explicit experimental acceptance is required for optics")
        if not isinstance(self.sensor_effects, SensorEffects):
            raise ValueError("Expected validated SensorEffects")
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
        # Reuse the allocation-free planner as the authoritative geometry domain.
        GeometryPlanner().plan(
            32,
            32,
            mode=self.geometry_mode,
            native_fit_policy=self.native_fit_policy,
            target_megapixels=self.target_megapixels,
        )
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
            for name, constructor in (
                ("camera", CameraHypothesis),
                ("isp", AnalyticISP),
                ("sensor_effects", SensorEffects),
                ("vignetting", Vignetting),
            ):
                if name in values:
                    if not isinstance(values[name], dict):
                        raise ValueError(f"{name} must be an object")
                    values[name] = constructor(**values[name])
            return cls(**values)
        except TypeError as error:
            raise ValueError(f"Invalid analytic configuration fields: {error}") from error

    def run(
        self,
        source_path: str | Path,
        output_dir: str | Path,
        *,
        camera_selections: tuple[CameraField, ...] = (),
        isp_selections: tuple[CameraField, ...] = (),
        sr_checkpoint: SRCheckpoint | None = None,
        inverse_mode: str = "ANALYTIC",
        residual_adapter: ResidualAdapter | None = None,
    ) -> dict:
        """Execute the bounded analytic pipeline with exactly these settings."""
        from prd_v5.pipeline import run_analytic

        return run_analytic(
            source_path,
            output_dir,
            self.camera,
            self.isp,
            sensor_effects=self.sensor_effects,
            vignetting=self.vignetting,
            accept_experimental_optics=self.accept_experimental_optics,
            camera_selections=camera_selections,
            isp_selections=isp_selections,
            sr_checkpoint=sr_checkpoint,
            inverse_mode=inverse_mode,
            residual_adapter=residual_adapter,
            target_megapixels=self.target_megapixels,
            geometry_mode=self.geometry_mode,
            native_fit_policy=self.native_fit_policy,
            assume_unprofiled_srgb=self.assume_unprofiled_srgb,
            strip_rows=self.strip_rows,
            max_decode_bytes=self.max_decode_bytes,
            max_develop_bytes=self.max_develop_bytes,
        )
