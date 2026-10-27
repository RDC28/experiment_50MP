"""Sony tag discovery for PRD_v3's Sony ARW-emulation registry contract."""

from __future__ import annotations

import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# Seed coverage is deliberately a fallback, never an exhaustive Sony catalogue.
_SEED_FIELDS = (
    "FileFormat SonyRawFileType SonyRawImageSize Quality Quality2 SonyModelID "
    "SonyISO BaseISO StopsAboveBaseISO SonyExposureTime2 ExposureTime SonyFNumber MaxApertureValue "
    "ShutterType SonyImageWidthMax SonyImageHeightMax AspectRatio WhiteBalance ColorTemperature "
    "WhiteBalanceShift CreativeStyle PictureEffect Sharpness DynamicRangeOptimizer HDR "
    "HighISONoiseReduction LongExposureNoiseReduction LensFormat LensMount LensType LensType2 LensType3 "
    "LensSpec EMountVersion DistortionCorrection VignettingCorrection ChromaticAberrationCorrection "
    "DistortionCorrParams VignettingCorrParams ChromaticAberrationCorrParams FocusMode AFAreaMode "
    "AFPointSelected AFStatus FocusDistance ReleaseMode SequenceNumber SelfTimer CameraTemperature "
    "SensorTemperature BoardTemperature BatteryLevel InternalSerialNumber SerialNumber ShutterCount "
    "FirmwareVersion SonyDateTime DateTimeOriginal GPSLatitude GPSLongitude GPSAltitude"
).split()


def _entry(name: str, table: str, tag_id: str | None, data_type: str | None, writable: bool | None) -> dict[str, Any]:
    return {
        "field_name": name,
        "table": table,
        "tag_id": tag_id,
        "exiftool_type": data_type,
        "writable": writable,
        "enum_values": {},
        "implementation_state": "UNKNOWN",
        "generated_value": None,
        "provenance": None,
        "confidence": None,
        "unsupported_reason": "No emulation policy has been applied to this field",
    }


class SonyMetadataRegistry:
    """Discover Sony fields without writing MakerNotes or inventing tag definitions."""

    def parse_catalogue(self, xml: str | bytes) -> dict[str, Any]:
        """Parse ExifTool -listx; retain table/tag identity to avoid name collisions."""
        root = ET.fromstring(xml)
        if root.tag != "taginfo":
            raise ValueError("Expected an ExifTool taginfo catalogue")
        entries = []
        for table in root.findall("table"):
            table_name = table.get("name", "")
            identity = " ".join((table_name, table.get("g0", ""), table.get("g1", ""))).lower()
            sony_table = "sony" in identity or "arw" in identity
            for tag in table.findall("tag"):
                name = tag.get("name")
                if not name or not (sony_table or name.startswith("Sony") or name.startswith("ARW")):
                    continue
                writable = tag.get("writable")
                if writable not in (None, "true", "false"):
                    raise ValueError(f"Invalid writable state for {name}: {writable}")
                entry = _entry(
                    name,
                    table_name,
                    tag.get("id"),
                    tag.get("type"),
                    None if writable is None else writable == "true",
                )
                for value in tag.findall("values/key"):
                    key = value.get("id")
                    label = value.find("val[@lang='en']")
                    if label is None:
                        label = value.find("val")
                    if key is not None and label is not None:
                        entry["enum_values"][key] = "".join(label.itertext())
                entries.append(entry)
        if not entries:
            raise ValueError("ExifTool catalogue contains no Sony/ARW fields")
        return {"source": "exiftool-listx", "seed_only": False, "fallback_reason": None, "fields": entries}

    def build(self) -> dict[str, Any]:
        """Query installed ExifTool; otherwise return the bundled seed marked with the reason it was used."""
        executable = shutil.which("exiftool")
        if executable is None:
            return self.seed("ExifTool is not installed")
        try:
            version = subprocess.run([executable, "-ver"], capture_output=True, check=True, timeout=30)
            result = subprocess.run([executable, "-listx"], capture_output=True, check=True, timeout=120)
            registry = self.parse_catalogue(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError, ET.ParseError) as error:
            return self.seed(f"ExifTool catalogue unavailable: {type(error).__name__}: {error}")
        registry["exiftool_version"] = version.stdout.decode("ascii", "replace").strip()
        return registry

    @staticmethod
    def seed(reason: str) -> dict[str, Any]:
        """The bundled fallback seed; ``seed_only`` and ``fallback_reason`` say it is not a full catalogue."""
        return {
            "source": "bundled-seed",
            "seed_only": True,
            "fallback_reason": reason,
            "exiftool_version": None,
            "fields": [_entry(name, "seed", None, None, None) for name in _SEED_FIELDS],
        }

    def write(self, path: str | Path) -> Path:
        """Write sony_tag_registry.json with explicit unknown values and strict JSON."""
        payload = json.dumps(self.build(), indent=2, allow_nan=False)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n", encoding="utf-8")
        return destination
