"""Built-in IMAGE_ONLY profiles for PRD_v3's camera and sensor specification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    def build(self, name: str = "SONY_A1II_LIKE") -> CameraProfile:
        """Return a fresh profile; virtual geometries never claim native Sony status."""
        if name not in PROFILE_NAMES:
            raise ValueError(f"Unknown camera profile: {name}")
        virtual = name != "SONY_A1II_LIKE"
        bits = 16 if name == "VIRTUAL_16BIT_SENSOR" else 14
        width, height = (10872, 4608) if name == "CINEMA_50MP_VIRTUAL" else (8640, 5760)
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
            "active_area": [0, 0, height, width],
            "default_crop_origin": [0, 0],
            "default_crop_size": [width, height],
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
        return CameraProfile(camera, sensor)
