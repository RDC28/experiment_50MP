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
        optical_records: Mapping[str, Mapping[str, FieldRecord]] | None = None,
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
        }
        optical = self._optical_fields(config, optical_records)
        for name, record in inputs.items():
            record.validate()
            if name != record.field:
                raise ValueError("Input key must match FieldRecord.field")
            lower = name.lower()
            if name in fixed or name in optical:
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
            elif name in optical:
                record = optical[name]
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
                    entry["unsupported_reason"] = "Optical stage disabled or no applied simulator records supplied"
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

    @staticmethod
    def _optical_fields(
        config: ExperimentConfig,
        stages: Mapping[str, Mapping[str, FieldRecord]] | None,
    ) -> dict[str, FieldRecord]:
        """Export PRD lens-family values from applied stages, never guessed Sony encodings."""
        result = {}
        families = (
            ("distortion", "Distortion", config.DISTORTION, ("k1", "k2")),
            ("vignetting", "Vignetting", config.VIGNETTING, ("corner_loss_ev", "power")),
            ("chromatic", "ChromaticAberration", config.CHROMATIC_ABERRATION, ("red_scale", "blue_scale")),
        )
        for stage, family, toggle, coefficients in families:
            records = (stages or {}).get(stage)
            enabled = False
            if records is not None:
                for name, record in records.items():
                    record.validate()
                    if record.field != name:
                        raise ValueError("Optical record key must match FieldRecord.field")
                if "enabled" not in records or type(records["enabled"].value) is not bool:
                    raise ValueError("Optical stage requires a boolean enabled record")
                enabled = records["enabled"].value
                if enabled != (config.OPTICAL_SIMULATION and toggle):
                    raise ValueError("Applied optical records disagree with config")
            flag = family + "Correction"
            result[flag] = FieldRecord(flag, enabled, Provenance.DETERMINISTIC, "Applied LensSimulator stage state")
            if not enabled:
                continue
            parameters = [records[name] for name in coefficients]
            # These are synthetic model coefficients, not proprietary Sony binary arrays.
            name = family + "CorrParams"
            result[name] = FieldRecord(
                name,
                [record.value for record in parameters],
                Provenance.FABRICATED,
                "Applied LensSimulator coefficients in order: " + ", ".join(coefficients),
                dependencies=[f"optics.{stage}.{key}" for key in coefficients],
                disclaimer="Synthetic forward-model parameters; not Sony MakerNote encoding or lens calibration",
                alternatives=[{"parameter": key, "field_record": records[key].to_dict()} for key in coefficients],
            )
        return result

    def write(self, path: str | Path, registry: dict[str, Any], config: ExperimentConfig, **kwargs: Any) -> Path:
        """Write sony_arw_emulation.json; never embed proprietary binary metadata."""
        payload = json.dumps(self.build(registry, config, **kwargs), indent=2, allow_nan=False)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n", encoding="utf-8")
        return destination
