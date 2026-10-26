"""Sidecar-only Sony emulation following PRD_v3's Sony family policy."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from synthraw.config import ExperimentConfig
from synthraw.provenance import FieldRecord, Provenance


class SonyARWEmulationBuilder:
    """Populate a discovered catalogue without creating Sony MakerNote blocks."""

    def build(
        self,
        registry: dict[str, Any],
        config: ExperimentConfig,
        *,
        width: int,
        height: int,
        inputs: Mapping[str, FieldRecord] | None = None,
        arw_compression_emulation: str = "UNCOMPRESSED",
    ) -> dict[str, Any]:
        """Retain tag identity, provenance and explicit reasons for all missing values.

        Inputs carry upstream exposure, lens, WB or user metadata. No camera
        settings are inferred here. Optical flags describe the simulated pixels.
        """
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError("width and height must be positive integers")
        if arw_compression_emulation not in {"UNCOMPRESSED", "LOSSLESS", "COMPRESSED"}:
            raise ValueError("Unsupported ARW compression emulation")
        inputs = inputs or {}
        names = {entry["field_name"] for entry in registry["fields"]}
        if inputs.keys() - names:
            raise ValueError("Input fields must exist in the registry")
        fixed = {
            "FileFormat": [5, 0, 0, 0],
            "SonyImageWidthMax": width,
            "SonyImageHeightMax": height,
            "SonyRawImageSize": [width, height],
            "AspectRatio": [width, height],
            "HighISONoiseReduction": "Off",
            "LongExposureNoiseReduction": "Off",
            "DistortionCorrection": config.OPTICAL_SIMULATION and config.DISTORTION,
            "VignettingCorrection": config.OPTICAL_SIMULATION and config.VIGNETTING,
            "ChromaticAberrationCorrection": config.OPTICAL_SIMULATION and config.CHROMATIC_ABERRATION,
        }
        for name, record in inputs.items():
            record.validate()
            if name != record.field:
                raise ValueError("Input key must match FieldRecord.field")
            lower = name.lower()
            if name in fixed:
                raise ValueError(f"{name} is determined by the simulation")
            if "serial" in lower or "shuttercount" in lower or name == "SonyModelID":
                raise ValueError(f"{name} requires a verified identity policy and cannot be supplied")
            if any(token in lower for token in ("date", "timeoriginal", "gps")) and not record.user_override:
                raise ValueError(f"{name} requires explicit user input")
            if "temperature" in lower or "battery" in lower:
                if record.provenance != Provenance.FABRICATED:
                    raise ValueError(f"{name} must have fabricated provenance")
                if not config.FABRICATE_UNOBSERVABLE_EXIF:
                    raise ValueError(f"{name} requires FABRICATE_UNOBSERVABLE_EXIF")
            if name.endswith("CorrParams"):
                raise ValueError("Correction arrays require an optical simulator export policy")
        result = copy.deepcopy(registry)
        for entry in result["fields"]:
            name = entry["field_name"]
            entry.update(
                implementation_state="UNKNOWN",
                generated_value=None,
                provenance=None,
                confidence=None,
                unsupported_reason="No verified value or emulation policy is available",
            )
            if name in fixed:
                # ARW identity is a fictional sidecar claim, not a physical format.
                provenance = Provenance.FABRICATED if name == "FileFormat" else Provenance.DETERMINISTIC
                record = FieldRecord(name, fixed[name], provenance, "PRD Sony sidecar simulation policy")
            elif name in inputs:
                record = inputs[name]
            else:
                lower = name.lower()
                if "serial" in lower or "shuttercount" in lower:
                    entry["unsupported_reason"] = "Body identity remains null; no realistic identity is fabricated"
                elif any(token in lower for token in ("date", "timeoriginal", "gps")):
                    entry["unsupported_reason"] = "Requires explicit user input; no date or location inference"
                elif name.endswith("CorrParams"):
                    entry["implementation_state"] = "UNSUPPORTED"
                    entry["unsupported_reason"] = "No optical simulator correction array has been exported"
                elif "temperature" in lower or "battery" in lower:
                    entry["unsupported_reason"] = "No fabricated value supplied; never inferred from rendered pixels"
                continue
            entry.update(
                implementation_state="EMULATED",
                generated_value=record.value,
                provenance=record.provenance.value,
                confidence=record.confidence,
                unsupported_reason=None,
                field_record=record.to_dict(),
            )
        result.update(
            synthetic=True,
            target_profile=config.TARGET_PROFILE,
            ARW_COMPRESSION_EMULATION=arw_compression_emulation,
            disclaimer=(
                "Synthetic sensor reconstruction of a hypothetical exposure; no physical camera capture occurred."
            ),
        )
        # Validate the complete sidecar before returning, including arbitrary input values.
        json.dumps(result, allow_nan=False)
        return result

    def write(self, path: str | Path, registry: dict[str, Any], config: ExperimentConfig, **kwargs: Any) -> Path:
        """Write sony_arw_emulation.json; never embed proprietary binary metadata."""
        payload = json.dumps(self.build(registry, config, **kwargs), indent=2, allow_nan=False)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n", encoding="utf-8")
        return destination
