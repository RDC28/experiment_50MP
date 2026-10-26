import json

import pytest

from synthraw.provenance import FieldRecord, Provenance, ProvenanceWriter


@pytest.mark.parametrize("provenance", list(Provenance))
def test_record_json_round_trip(provenance):
    record = FieldRecord("ImageWidth", 8640, provenance, "geometry", unit="pixels")
    payload = json.loads(json.dumps(record.to_dict()))
    assert payload["provenance"] == provenance.value
    assert FieldRecord.from_dict(payload) == record
    assert set(payload) == {
        "field",
        "value",
        "unit",
        "provenance",
        "method",
        "confidence",
        "alternatives",
        "dependencies",
        "user_override",
        "disclaimer",
    }


def test_prd_inferred_example_and_fabricated_disclaimer(tmp_path):
    exposure = FieldRecord(
        "ExposureTime",
        0.008,
        Provenance.INFERRED,
        "joint_exposure_posterior_v2",
        unit="s",
        confidence=0.42,
        alternatives=[{"value": 0.004, "weight": 0.26}],
        dependencies=["FNumber", "PhotographicSensitivity", "scene_luminance_scale"],
    )
    fabricated = FieldRecord(
        "full_well_electrons",
        52000,
        Provenance.FABRICATED,
        "virtual_a1ii_profile_prior",
        unit="electrons",
        disclaimer="Not measured from a Sony Alpha 1 II sensor.",
    )
    destination = tmp_path / "metadata" / "provenance.json"
    assert ProvenanceWriter().write([exposure, fabricated], destination) == destination
    records = json.loads(destination.read_text())
    assert records == [exposure.to_dict(), fabricated.to_dict()]
    assert records[1]["confidence"] is None
    assert records[1]["disclaimer"] == fabricated.disclaimer


@pytest.mark.parametrize("confidence", [0, 0.5, 1])
def test_fabricated_confidence_rejected(confidence):
    with pytest.raises(ValueError, match="Fabricated"):
        FieldRecord("noise", 1, Provenance.FABRICATED, "simulation", confidence=confidence)


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_confidence_rejected(confidence):
    with pytest.raises(ValueError, match="confidence"):
        FieldRecord("exposure", 1, Provenance.INFERRED, "estimate", confidence=confidence)


def test_unknown_provenance_rejected():
    with pytest.raises(ValueError):
        FieldRecord.from_dict({"field": "width", "value": 2, "provenance": "X", "method": "geometry"})


def test_writer_revalidates_mutated_records(tmp_path):
    record = FieldRecord("noise", 1, Provenance.FABRICATED, "simulation")
    record.confidence = 0.9
    with pytest.raises(ValueError, match="Fabricated"):
        ProvenanceWriter().write([record], tmp_path / "invalid.json")
    assert not (tmp_path / "invalid.json").exists()


def test_nonfinite_values_do_not_write_partial_output(tmp_path):
    destination = tmp_path / "metadata.json"
    destination.write_text("previous output")
    record = FieldRecord("value", float("nan"), Provenance.INFERRED, "estimate")
    with pytest.raises(ValueError):
        ProvenanceWriter().write([record], destination)
    assert destination.read_text() == "previous output"


def test_mutable_defaults_are_independent():
    first = FieldRecord("width", 2, Provenance.DETERMINISTIC, "geometry")
    second = FieldRecord("height", 2, Provenance.DETERMINISTIC, "geometry")
    first.dependencies.append("source")
    assert second.dependencies == []
