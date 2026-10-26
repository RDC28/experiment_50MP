"""Built-in and user-supplied reference profiles for PRD_v3's camera and sensor specification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthraw.config import ExperimentConfig
from synthraw.geometry import GeometryPlan, TargetGeometryPlanner
from synthraw.provenance import FieldRecord, Provenance, ProvenanceWriter

SENSOR_PRIOR_DISCLAIMER = "Not measured from a Sony Alpha 1 II sensor."
PROFILE_NAMES = ("SONY_A1II_LIKE", "CINEMA_50MP_VIRTUAL", "VIRTUAL_16BIT_SENSOR")


@dataclass
class CameraProfile:
    """Independent camera envelope and sensor records, never capture metadata."""

    camera: dict[str, FieldRecord]
    sensor: dict[str, FieldRecord]

    def write(self, directory: str | Path) -> tuple[Path, Path]:
        """Write both PRD profile sidecars using the shared provenance schema."""
        root = Path(directory)
        writer = ProvenanceWriter()
        return (
            writer.write(list(self.camera.values()), root / "camera_profile.json"),
            writer.write(list(self.sensor.values()), root / "sensor_profile.json"),
        )


class CameraProfileManager:
    """Construct IMAGE_ONLY priors without loading RAWs or allocating sensor arrays."""

    def build(
        self,
        name: str = "SONY_A1II_LIKE",
        *,
        source: str | None = None,
        config: ExperimentConfig | None = None,
        geometry: GeometryPlan | None = None,
        reference_raw: str | Path | None = None,
    ) -> CameraProfile:
        """Return a fresh profile; virtual geometries never claim native Sony status."""
        if config is not None:
            name = config.TARGET_PROFILE
        source = source or (config.CAMERA_PROFILE_SOURCE if config is not None else "IMAGE_ONLY")
        if name not in PROFILE_NAMES:
            raise ValueError(f"Unknown camera profile: {name}")
        if source not in ("IMAGE_ONLY", "REFERENCE_RAW", "FULL_CALIBRATION"):
            raise ValueError("Supported profile sources are IMAGE_ONLY, REFERENCE_RAW and FULL_CALIBRATION")
        reference_raw = reference_raw or (config.REFERENCE_RAW_PATH if config is not None else None)
        if source == "FULL_CALIBRATION" and reference_raw is None:
            raise ValueError("FULL_CALIBRATION requires REFERENCE_RAW_PATH")
        if source == "REFERENCE_RAW" and reference_raw is None:
            raise ValueError("REFERENCE_RAW requires a user-supplied reference_raw file")
        if source == "IMAGE_ONLY" and reference_raw is not None:
            raise ValueError("Select REFERENCE_RAW to use a reference_raw file")
        virtual = name != "SONY_A1II_LIKE"
        bits = 16 if name == "VIRTUAL_16BIT_SENSOR" else 14
        # Legacy name-only calls retain the PRD example source geometry.
        target = geometry or TargetGeometryPlanner().plan(
            2048,
            868,
            config
            or ExperimentConfig(
                OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL" if name == "CINEMA_50MP_VIRTUAL" else "SONY_A1II_NATIVE",
                ADC_BITS=bits,
            ),
        )
        width, height = target.width, target.height
        camera: dict[str, FieldRecord] = {}
        sensor: dict[str, FieldRecord] = {}

        def add(
            records: dict[str, FieldRecord],
            field: str,
            value: Any,
            unit: str | None = None,
            fabricated: bool = False,
            disclaimer: str | None = None,
        ) -> None:
            records[field] = FieldRecord(
                field,
                value,
                Provenance.FABRICATED if fabricated else Provenance.DETERMINISTIC,
                "PRD_v3 simulation prior" if fabricated else "PRD_v3 profile definition",
                unit=unit,
                disclaimer=disclaimer,
            )

        for field, value in {
            "profile_name": name,
            "profile_source": "IMAGE_ONLY",
            "Make": "Synthetic",
            "Model": name,
            "native_sony_geometry": not virtual,
            "iso_standard_range": [100, 32000],
            "iso_expanded_range": [50, 102400],
            "colour_temperature_range": [2500, 9900],
            "mechanical_shutter_min_seconds": 1 / 8000,
            "electronic_shutter_min_seconds": 1 / 32000,
        }.items():
            add(camera, field, value)
        add(camera, "sensor_width_mm", 35.9, "mm", fabricated=virtual)
        add(camera, "sensor_height_mm", 24.0, "mm", fabricated=virtual)
        for field, value in {
            "sensor_width": width,
            "sensor_height": height,
            "active_area": list(target.active_area),
            "default_crop_origin": list(target.default_crop_origin),
            "default_crop_size": list(target.default_crop_size),
            "adc_bits": bits,
            "dng_storage_bits": 16,
            "WhiteLevel": (1 << bits) - 1,
        }.items():
            add(sensor, field, value)
        add(sensor, "cfa_pattern", "RGGB", fabricated=True)
        add(sensor, "BlackLevel", 512, "DN", fabricated=True)
        # Each interval and its selected default retain the scientific disclaimer.
        for field, value, interval, unit in (
            ("full_well_electrons", 52000.0, [20000.0, 100000.0], "electrons"),
            ("read_noise_electrons", 3.0, [0.3, 10.0], "electrons RMS"),
            ("prnu_sigma", 0.01, [0.0, 0.03], "fraction"),
            ("dsnu_sigma_dn", 1.0, [0.0, 64.0], "DN"),
        ):
            add(sensor, field, value, unit, fabricated=True, disclaimer=SENSOR_PRIOR_DISCLAIMER)
            add(sensor, f"{field}_range", interval, unit, fabricated=True, disclaimer=SENSOR_PRIOR_DISCLAIMER)
        profile = CameraProfile(camera, sensor)
        if source in ("REFERENCE_RAW", "FULL_CALIBRATION"):
            self._read_reference(profile, Path(reference_raw))
            if geometry is not None:
                # Reference measurements describe calibration, not the output raster.
                for field, value in {
                    "sensor_width": width,
                    "sensor_height": height,
                    "active_area": list(target.active_area),
                    "default_crop_origin": list(target.default_crop_origin),
                    "default_crop_size": list(target.default_crop_size),
                }.items():
                    add(sensor, field, value)
        if source == "FULL_CALIBRATION":
            from synthraw.calibration import calibrate_sensor

            calibrate_sensor(profile, config or ExperimentConfig(), Path(reference_raw))
        return profile

    @staticmethod
    def _read_reference(profile: CameraProfile, path: Path) -> None:
        """Read exposed decoder measurements only; retain unmeasured PRD priors."""
        if not path.is_file():
            raise FileNotFoundError(path)
        import numpy as np
        import rawpy

        def measured(records: dict[str, FieldRecord], field: str, value: Any) -> None:
            records[field] = FieldRecord(
                field,
                value,
                Provenance.MEASURED,
                "User-supplied reference RAW via LibRaw/rawpy",
                dependencies=[str(path)],
                disclaimer="Measured from the supplied reference file, not from the synthetic exposure.",
            )

        with rawpy.imread(str(path)) as raw:
            pattern = raw.raw_pattern
            if pattern is None or pattern.shape != (2, 2):
                raise ValueError("Reference RAW must expose a 2x2 Bayer CFA")
            phase = "".join(chr(raw.color_desc[int(index)]) for index in pattern.ravel())
            if phase not in ("RGGB", "BGGR", "GRBG", "GBRG"):
                raise ValueError("Reference RAW must expose an RGB Bayer CFA")
            levels = np.asarray(raw.black_level_per_channel, dtype=float)
            white = float(raw.white_level)
            wb = np.asarray(raw.camera_whitebalance, dtype=float)
            if not np.isfinite(levels).all() or np.any(levels < 0) or not np.all(levels < white):
                raise ValueError("Reference RAW has invalid black/white levels")
            if wb.size < 3 or not np.isfinite(wb[:3]).all() or np.any(wb[:3] <= 0):
                raise ValueError("Reference RAW has no valid camera white balance")
            for field, value in {
                "sensor_width": raw.sizes.raw_width,
                "sensor_height": raw.sizes.raw_height,
                "active_area": [
                    raw.sizes.top_margin,
                    raw.sizes.left_margin,
                    raw.sizes.top_margin + raw.sizes.height,
                    raw.sizes.left_margin + raw.sizes.width,
                ],
                "default_crop_origin": [0, 0],
                "default_crop_size": [raw.sizes.width, raw.sizes.height],
                "cfa_pattern": phase,
                "BlackLevel": levels.tolist(),
                "WhiteLevel": raw.white_level,
            }.items():
                measured(profile.sensor, field, value)
            measured(profile.camera, "camera_whitebalance", wb.tolist())
            measured(profile.camera, "AsShotNeutral", (wb[1] / wb[:3]).tolist())
            # These are LibRaw transforms, not the original dual-illuminant DNG tags.
            for field, matrix in (
                ("libraw_color_matrix", raw.color_matrix),
                ("libraw_rgb_xyz_matrix", raw.rgb_xyz_matrix),
            ):
                matrix = np.asarray(matrix)
                if np.isfinite(matrix).all() and np.any(matrix):
                    measured(profile.camera, field, matrix.tolist())
        profile.camera["profile_source"] = FieldRecord(
            "profile_source", "REFERENCE_RAW", Provenance.DETERMINISTIC, "Explicit user profile selection"
        )
