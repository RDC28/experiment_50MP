"""Sony tag discovery for PRD_v3's Sony ARW-emulation registry contract."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# Pinned current ExifTool release (https://exiftool.org/checksums.txt), unpacked and run with the system Perl.
EXIFTOOL_VERSION = "13.59"
EXIFTOOL_SHA256 = "668ea3acececb7235fbd0f4900e72d5f12c9b07e5c778fd36cb1e9b5828fd65a"
EXIFTOOL_URL = f"https://sourceforge.net/projects/exiftool/files/Image-ExifTool-{EXIFTOOL_VERSION}.tar.gz/download"
# Set by install_exiftool when the pinned release cannot be installed; ExifTool then counts as absent.
EXIFTOOL_UNAVAILABLE_ENV = "SYNTHRAW_EXIFTOOL_UNAVAILABLE"

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
        """Query installed ExifTool; only an absent ExifTool returns the bundled seed, marked with the reason.

        An installed ExifTool that fails (non-zero exit, timeout, bad XML, no Sony fields) raises.
        """
        unavailable = os.environ.get(EXIFTOOL_UNAVAILABLE_ENV)
        executable = None if unavailable else shutil.which("exiftool")
        if executable is None:
            return self.seed(unavailable or "ExifTool is not installed")
        version = subprocess.run([executable, "-ver"], capture_output=True, check=True, timeout=30)
        result = subprocess.run([executable, "-listx"], capture_output=True, check=True, timeout=120)
        registry = self.parse_catalogue(result.stdout)
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


def install_exiftool(directory: str | Path) -> Path | None:
    """Install the pinned ExifTool release into ``directory`` and put it first on ``PATH``.

    The download must match ``EXIFTOOL_SHA256`` before it is unpacked. If the download or the verification
    fails, ExifTool counts as absent: this warns, returns None and keeps the reason in
    ``EXIFTOOL_UNAVAILABLE_ENV``, so ``SonyMetadataRegistry.build`` records it with the bundled seed.
    """
    root = Path(directory)
    try:
        with urllib.request.urlopen(EXIFTOOL_URL, timeout=120) as response:
            archive = response.read()
        digest = hashlib.sha256(archive).hexdigest()
        if digest != EXIFTOOL_SHA256:
            raise ValueError(f"sha256 {digest} does not match the pinned {EXIFTOOL_SHA256}")
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            tar.extractall(root, filter="data")
    except (OSError, ValueError, tarfile.TarError) as error:
        reason = f"Pinned ExifTool {EXIFTOOL_VERSION} could not be installed: {type(error).__name__}: {error}"
        os.environ[EXIFTOOL_UNAVAILABLE_ENV] = reason
        warnings.warn(f"{reason}; the Sony tag registry uses the bundled seed.", stacklevel=2)
        return None
    os.environ.pop(EXIFTOOL_UNAVAILABLE_ENV, None)
    bin_dir = root / f"Image-ExifTool-{EXIFTOOL_VERSION}"
    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    return bin_dir / "exiftool"
