"""Tiny, offline Sony catalogue fixtures for PRD_v3 metadata discovery."""

import hashlib
import http.client
import io
import json
import os
import subprocess
import tarfile
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from synthraw import sony_registry
from synthraw.sony_registry import (
    EXIFTOOL_UNAVAILABLE_ENV,
    EXIFTOOL_URL,
    EXIFTOOL_VERSION,
    SonyMetadataRegistry,
    install_exiftool,
)

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
    "stage, failure, error",
    [
        *[
            (stage, failure, type(failure))
            for stage in ("-ver", "-listx")
            for failure in (
                subprocess.CalledProcessError(1, "exiftool"),
                subprocess.TimeoutExpired("exiftool", 120),
                PermissionError("exiftool"),
            )
        ],
        ("-listx", SimpleNamespace(stdout=b"<taginfo/>"), ValueError),
        ("-listx", SimpleNamespace(stdout=b"not xml"), ET.ParseError),
    ],
)
def test_exiftool_failure_does_not_silently_downgrade_to_seed(stage, failure, error):
    def run(command, **kwargs):
        if command[1] == stage:
            if isinstance(failure, BaseException):
                raise failure
            return failure
        return SimpleNamespace(stdout=b"13.36\n" if command[1] == "-ver" else FIXTURE.read_bytes())

    with (
        patch("synthraw.sony_registry.shutil.which", return_value="/bin/exiftool"),
        patch("synthraw.sony_registry.subprocess.run", side_effect=run),
        pytest.raises(error),
    ):
        SonyMetadataRegistry().build()


def _release(script):
    """A gzip tarball laid out like the ExifTool release, with ``script`` as its exiftool."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        data = script.encode()
        member = tarfile.TarInfo(f"Image-ExifTool-{EXIFTOOL_VERSION}/exiftool")
        member.size, member.mode = len(data), 0o755
        tar.addfile(member, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def environment(monkeypatch):
    """Restore PATH and the install-failure reason after each install test."""
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))
    monkeypatch.setenv(EXIFTOOL_UNAVAILABLE_ENV, "")
    monkeypatch.delenv(EXIFTOOL_UNAVAILABLE_ENV)
    return monkeypatch


def test_install_exiftool_verifies_the_pinned_release_and_the_registry_uses_it(environment, tmp_path):
    archive = _release(f'#!/bin/sh\nif [ "$1" = -ver ]; then echo {EXIFTOOL_VERSION}; else cat "{FIXTURE}"; fi\n')
    requested = []

    def urlopen(url, timeout):
        requested.append(url)
        return io.BytesIO(archive)

    environment.setattr(sony_registry.urllib.request, "urlopen", urlopen)
    environment.setattr(sony_registry, "EXIFTOOL_SHA256", hashlib.sha256(archive).hexdigest())
    executable = install_exiftool(tmp_path / "exiftool")
    assert requested == [EXIFTOOL_URL]
    assert executable == tmp_path / "exiftool" / f"Image-ExifTool-{EXIFTOOL_VERSION}" / "exiftool"
    assert os.environ["PATH"].split(os.pathsep)[0] == str(executable.parent)
    assert EXIFTOOL_UNAVAILABLE_ENV not in os.environ
    registry = SonyMetadataRegistry().build()
    assert registry["source"] == "exiftool-listx" and registry["seed_only"] is False
    assert registry["exiftool_version"] == EXIFTOOL_VERSION and len(registry["fields"]) == 6


def _tampered(url, timeout):
    return io.BytesIO(_release("#!/bin/sh\n"))


def _offline(url, timeout):
    raise urllib.error.URLError("offline")


class _Truncated(io.BytesIO):
    def read(self, *args):
        raise http.client.IncompleteRead(b"partial", 7_900_000)


def _truncated(url, timeout):
    return _Truncated()


def _malformed(url, timeout):
    raise http.client.BadStatusLine("garbage")


@pytest.mark.parametrize(
    "urlopen, reason",
    [(_tampered, "sha256 "), (_offline, "URLError"), (_truncated, "IncompleteRead"), (_malformed, "BadStatusLine")],
    ids=["hash-mismatch", "download-failure", "truncated-download", "malformed-response"],
)
def test_failed_exiftool_install_counts_as_absent_with_the_reason_recorded(environment, tmp_path, urlopen, reason):
    path = os.environ["PATH"]
    environment.setattr(sony_registry.urllib.request, "urlopen", urlopen)
    with pytest.warns(UserWarning, match="bundled seed"):
        assert install_exiftool(tmp_path / "exiftool") is None
    assert not (tmp_path / "exiftool").exists() and os.environ["PATH"] == path
    # Any other ExifTool on PATH is not used either.
    with (
        patch("synthraw.sony_registry.shutil.which", return_value="/bin/exiftool"),
        patch("synthraw.sony_registry.subprocess.run", side_effect=AssertionError("ExifTool must not run")),
    ):
        registry = SonyMetadataRegistry().build()
    assert registry["seed_only"] is True and registry["source"] == "bundled-seed"
    assert registry["fallback_reason"].startswith(f"Pinned ExifTool {EXIFTOOL_VERSION} could not be installed: ")
    assert reason in registry["fallback_reason"]


@pytest.mark.parametrize(
    "xml", ["<other/>", "<taginfo/>", '<taginfo><table name="Sony"><tag name="X" writable="yes"/></table></taginfo>']
)
def test_invalid_catalogues_are_rejected(xml):
    with pytest.raises(ValueError):
        SonyMetadataRegistry().parse_catalogue(xml)
