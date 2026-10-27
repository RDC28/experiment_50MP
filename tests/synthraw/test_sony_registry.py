"""Tiny, offline Sony catalogue fixtures for PRD_v3 metadata discovery."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from synthraw.sony_registry import SonyMetadataRegistry

FIXTURE = Path(__file__).parent / "fixtures" / "exiftool_sony_listx.xml"


def test_parse_catalogue_preserves_types_enums_and_duplicate_names():
    registry = SonyMetadataRegistry().parse_catalogue(FIXTURE.read_bytes())
    assert registry["seed_only"] is False
    fields = registry["fields"]
    assert len(fields) == 6
    assert fields[0]["enum_values"] == {"5 0 0 0": "ARW 5.0"}
    assert fields[0]["exiftool_type"] == "int8u"
    assert fields[0]["writable"] is True
    assert fields[1]["writable"] is False
    quality = [field for field in fields if field["field_name"] == "Quality"]
    assert len(quality) == 2
    assert quality[0]["table"] != quality[1]["table"]
    assert quality[1]["writable"] is None
    assert all(field["unsupported_reason"] and field["generated_value"] is None for field in fields)


def test_seed_covers_prd_families_and_serializes(tmp_path):
    with patch("synthraw.sony_registry.shutil.which", return_value=None):
        path = SonyMetadataRegistry().write(tmp_path / "metadata" / "sony_tag_registry.json")
    registry = json.loads(path.read_text())
    assert registry["seed_only"] is True
    fields = {field["field_name"]: field for field in registry["fields"]}
    required = {
        "FileFormat",
        "SonyRawFileType",
        "SonyRawImageSize",
        "Quality",
        "Quality2",
        "SonyModelID",
        "SonyISO",
        "BaseISO",
        "StopsAboveBaseISO",
        "SonyExposureTime2",
        "ExposureTime",
        "SonyFNumber",
        "ShutterType",
        "SonyImageWidthMax",
        "SonyImageHeightMax",
        "LensFormat",
        "LensMount",
        "LensType",
        "LensType2",
        "LensType3",
        "FocusMode",
        "BatteryLevel",
        "InternalSerialNumber",
        "ShutterCount",
        "FirmwareVersion",
        "GPSLatitude",
        "DistortionCorrParams",
        "HighISONoiseReduction",
    }
    assert required <= fields.keys()
    assert all(field["exiftool_type"] is None and field["provenance"] is None for field in fields.values())


def test_installed_exiftool_is_queried_without_shell():
    def run(command, **kwargs):
        assert kwargs == {"capture_output": True, "check": True, "timeout": 30 if command[1] == "-ver" else 120}
        return SimpleNamespace(stdout=b"13.36\n" if command[1] == "-ver" else FIXTURE.read_bytes())

    with (
        patch("synthraw.sony_registry.shutil.which", return_value="/bin/exiftool"),
        patch("synthraw.sony_registry.subprocess.run", side_effect=run) as mocked,
    ):
        registry = SonyMetadataRegistry().build()
    assert registry["seed_only"] is False and registry["fallback_reason"] is None
    assert registry["source"] == "exiftool-listx" and registry["exiftool_version"] == "13.36"
    assert [call.args[0] for call in mocked.call_args_list] == [["/bin/exiftool", "-ver"], ["/bin/exiftool", "-listx"]]


def test_missing_exiftool_records_why_the_seed_is_used():
    with patch("synthraw.sony_registry.shutil.which", return_value=None):
        registry = SonyMetadataRegistry().build()
    assert registry["seed_only"] is True and registry["source"] == "bundled-seed"
    assert registry["fallback_reason"] == "ExifTool is not installed" and registry["exiftool_version"] is None


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.CalledProcessError(1, "exiftool"),
        subprocess.TimeoutExpired("exiftool", 120),
        PermissionError("exiftool"),
        SimpleNamespace(stdout=b"<taginfo/>"),
        SimpleNamespace(stdout=b"not xml"),
    ],
)
def test_exiftool_failure_falls_back_to_the_seed_with_the_reason_recorded(failure):
    def run(command, **kwargs):
        if command[1] == "-ver":
            return SimpleNamespace(stdout=b"13.36\n")
        if isinstance(failure, BaseException):
            raise failure
        return failure

    with (
        patch("synthraw.sony_registry.shutil.which", return_value="/bin/exiftool"),
        patch("synthraw.sony_registry.subprocess.run", side_effect=run),
    ):
        registry = SonyMetadataRegistry().build()
    assert registry["seed_only"] is True and registry["source"] == "bundled-seed"
    assert registry["fallback_reason"].startswith("ExifTool catalogue unavailable: ")
    assert registry["fields"] and json.dumps(registry, allow_nan=False)


@pytest.mark.parametrize(
    "xml", ["<other/>", "<taginfo/>", '<taginfo><table name="Sony"><tag name="X" writable="yes"/></table></taginfo>']
)
def test_invalid_catalogues_are_rejected(xml):
    with pytest.raises(ValueError):
        SonyMetadataRegistry().parse_catalogue(xml)
