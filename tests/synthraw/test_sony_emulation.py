"""Offline sidecar contract tests for PRD_v3 Sony emulation."""

import copy
import json
from unittest.mock import patch

import pytest

from synthraw.config import ExperimentConfig
from synthraw.provenance import FieldRecord, Provenance
from synthraw.sony_emulation import SonyARWEmulationBuilder
from synthraw.sony_registry import SonyMetadataRegistry


@pytest.fixture
def registry():
    with patch("synthraw.sony_registry.shutil.which", return_value=None):
        return SonyMetadataRegistry().build()


def fields(result):
    return {entry["field_name"]: entry for entry in result["fields"]}


def test_complete_sidecar_roundtrip_preserves_catalogue_and_identity(registry, tmp_path):
    original = copy.deepcopy(registry)
    registry["fields"].append({**registry["fields"][0], "table": "second-table"})
    path = SonyARWEmulationBuilder().write(
        tmp_path / "sony_arw_emulation.json", registry, ExperimentConfig(), width=64, height=48
    )
    result = json.loads(path.read_text())
    assert registry["fields"][:-1] == original["fields"]
    assert len(result["fields"]) == len(registry["fields"])
    assert result["fields"][-1]["table"] == "second-table"
    values = fields(result)
    assert values["FileFormat"]["generated_value"] == [5, 0, 0, 0]
    assert values["FileFormat"]["provenance"] == "F"
    assert values["SonyRawImageSize"]["generated_value"] == [64, 48]
    assert values["SonyImageWidthMax"]["provenance"] == "D"
    assert result["ARW_COMPRESSION_EMULATION"] == "UNCOMPRESSED"
    for name in ("SonyModelID", "SerialNumber", "InternalSerialNumber", "ShutterCount", "GPSLatitude"):
        assert values[name]["generated_value"] is None
        assert values[name]["unsupported_reason"]
    for entry in result["fields"]:
        assert entry["generated_value"] is not None or entry["unsupported_reason"]


@pytest.mark.parametrize("optical,distortion", [(False, True), (True, False), (True, True)])
def test_corrections_and_nr_reflect_simulation(registry, optical, distortion):
    config = ExperimentConfig(OPTICAL_SIMULATION=optical, DISTORTION=distortion)
    values = fields(SonyARWEmulationBuilder().build(registry, config, width=64, height=48))
    assert values["DistortionCorrection"]["generated_value"] is (optical and distortion)
    assert values["VignettingCorrection"]["generated_value"] is False
    assert values["HighISONoiseReduction"]["generated_value"] == "Off"
    assert values["LongExposureNoiseReduction"]["generated_value"] == "Off"
    assert values["DistortionCorrParams"]["implementation_state"] == "UNSUPPORTED"


def test_exposure_and_user_date_preserve_provenance(registry):
    inputs = {
        "SonyISO": FieldRecord("SonyISO", 100, Provenance.INFERRED, "exposure posterior", confidence=0.2),
        "DateTimeOriginal": FieldRecord(
            "DateTimeOriginal", "2026:01:01 00:00:00", Provenance.MEASURED, "user input", user_override=True
        ),
    }
    result = SonyARWEmulationBuilder().build(registry, ExperimentConfig(), width=64, height=48, inputs=inputs)
    values = fields(result)
    assert values["SonyISO"]["confidence"] == 0.2
    assert values["DateTimeOriginal"]["field_record"]["user_override"] is True


@pytest.mark.parametrize(
    "name", ["SonyModelID", "SerialNumber", "ShutterCount", "GPSLatitude", "SonyDateTime", "DistortionCorrection"]
)
def test_forbidden_inputs_are_rejected(registry, name):
    record = FieldRecord(name, 123, Provenance.INFERRED, "test")
    with pytest.raises(ValueError):
        SonyARWEmulationBuilder().build(registry, ExperimentConfig(), width=64, height=48, inputs={name: record})


def test_unobservable_fabrication_is_explicit_and_never_confident(registry):
    builder = SonyARWEmulationBuilder()
    record = FieldRecord("BatteryLevel", 75, Provenance.FABRICATED, "fictional battery")
    with pytest.raises(ValueError, match="FABRICATE_UNOBSERVABLE_EXIF"):
        builder.build(registry, ExperimentConfig(), width=64, height=48, inputs={record.field: record})
    config = ExperimentConfig(FABRICATE_UNOBSERVABLE_EXIF=True)
    result = builder.build(registry, config, width=64, height=48, inputs={record.field: record})
    assert fields(result)["BatteryLevel"]["confidence"] is None
    record.provenance = Provenance.INFERRED
    with pytest.raises(ValueError, match="fabricated provenance"):
        builder.build(registry, config, width=64, height=48, inputs={record.field: record})


@pytest.mark.parametrize("width,height", [(0, 48), (64, -1), (True, 48)])
def test_invalid_dimensions(registry, width, height):
    with pytest.raises(ValueError):
        SonyARWEmulationBuilder().build(registry, ExperimentConfig(), width=width, height=height)
