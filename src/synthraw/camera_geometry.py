"""Optional calibration and explicit lens hypotheses for PRD_v3's camera milestone."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from synthraw.camera_profile import CameraProfile
from synthraw.provenance import FieldRecord, Provenance


@dataclass
class CameraGeometryEstimate:
    """Geometry records plus an auditable optional-model fallback reason."""

    records: dict[str, FieldRecord]
    backend: str
    fallback_reason: str | None


class CameraGeometryEstimator:
    """Infer a rectilinear lens hypothesis, never identify a real captured lens.

    A caller-owned lazy GeoCalib adapter returns horizontal FOV in degrees.
    It must use user-provided or runtime-approved assets; this class never loads
    models or downloads weights. FOV must describe the final sensor framing.
    """

    def estimate(
        self,
        profile: CameraProfile,
        *,
        manual_fov_degrees: float | None = None,
        profile_fov_degrees: float = 60.0,
        maximum_aperture: float = 2.8,
        use_geocalib: bool = False,
        geocalib_adapter: Callable[[Any], float] | None = None,
        image: Any = None,
    ) -> CameraGeometryEstimate:
        """Prefer explicit manual FOV, then optional GeoCalib, then profile FOV.

        Adapters translate their installed GeoCalib version's output to a scalar
        horizontal FOV. Expected availability/runtime failures fall back with a
        reason; programming errors propagate instead of being silently hidden.
        """
        if type(use_geocalib) is not bool:
            raise ValueError("use_geocalib must be boolean")
        self._positive(maximum_aperture, "maximum_aperture")
        self._fov(profile_fov_degrees)
        if manual_fov_degrees is not None:
            self._fov(manual_fov_degrees)
        width = profile.camera["sensor_width_mm"].value
        height = profile.camera["sensor_height_mm"].value
        self._positive(width, "sensor_width_mm")
        self._positive(height, "sensor_height_mm")
        fov, backend, reason = profile_fov_degrees, "PROFILE_DEFAULT", None
        if manual_fov_degrees is not None:
            fov, backend = manual_fov_degrees, "MANUAL"
        elif use_geocalib:
            if geocalib_adapter is None or image is None:
                reason = "GeoCalib requires a configured lazy adapter and an image; using profile FOV"
            else:
                try:
                    fov = geocalib_adapter(image)
                    self._fov(fov)
                    backend = "GEOCALIB"
                except (ImportError, OSError, RuntimeError, ValueError) as exc:
                    fov = profile_fov_degrees
                    reason = f"GeoCalib unavailable or invalid ({type(exc).__name__}: {exc}); using profile FOV"
        inferred = backend == "GEOCALIB"
        provenance = Provenance.INFERRED if inferred else Provenance.FABRICATED
        method = f"{backend} horizontal rectilinear FOV hypothesis; not lens identification"
        focal = float(width) / (2 * math.tan(math.radians(fov) / 2))
        equivalent = focal * math.hypot(36, 24) / math.hypot(width, height)
        dependencies = ["horizontal_fov_degrees", "sensor_width_mm", "sensor_height_mm"]
        records = {}
        for name, value, unit in (
            ("horizontal_fov_degrees", float(fov), "degrees"),
            ("FocalLength", focal, "mm"),
            ("FocalLengthIn35mmFormat", max(1, round(equivalent)), "mm"),
        ):
            records[name] = FieldRecord(
                name,
                value,
                provenance,
                method,
                unit=unit,
                dependencies=dependencies if name != "horizontal_fov_degrees" else [],
                user_override=backend == "MANUAL",
            )
        records["LensInfo"] = FieldRecord(
            "LensInfo",
            [focal, focal, float(maximum_aperture), float(maximum_aperture)],
            Provenance.FABRICATED,
            "Virtual fixed-focal lens and selected maximum-aperture hypothesis; not lens identification",
            dependencies=["FocalLength"],
            disclaimer="Aperture and lens identity cannot be recovered from FOV alone.",
        )
        return CameraGeometryEstimate(records, backend, reason)

    @staticmethod
    def _positive(value: float, name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    @classmethod
    def _fov(cls, value: float) -> None:
        cls._positive(value, "horizontal FOV")
        if value >= 180:
            raise ValueError("horizontal FOV must be less than 180 degrees")
