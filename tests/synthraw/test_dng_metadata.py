"""Tiny, allocation-free checks for the PRD_v3 core DNG field contract."""

from dataclasses import replace

import pytest

from synthraw.config import ExperimentConfig
from synthraw.dng_metadata import CFA_PATTERNS, DISCLOSURE, DNGFieldRecord, DNGMetadataBuilder
from synthraw.geometry import TargetGeometryPlanner
from synthraw.provenance import Provenance


@pytest.fixture
def geometry():
    plan = TargetGeometryPlanner().plan(64, 48)
    return replace(
        plan, width=64, height=48, active_area=(0, 0, 48, 64), default_crop_origin=(0, 0), default_crop_size=(64, 48)
    )


@pytest.mark.parametrize("phase", CFA_PATTERNS)
@pytest.mark.parametrize("bits", (14, 16))
def test_core_contract(geometry, phase, bits):
    config = ExperimentConfig(ADC_BITS=bits)
    fields = DNGMetadataBuilder().build(geometry, config, cfa_phase=phase)
    assert fields["CFAPattern"].value == CFA_PATTERNS[phase]
    assert fields["WhiteLevel"].value == 2**bits - 1
    assert fields["BitsPerSample"].value == 16
    assert fields["Make"].value == "Synthetic"
    assert fields["ImageDescription"].value == DISCLOSURE
    assert fields["DNGBackwardVersion"].value == (1, 3, 0, 0)
    assert fields["ColorMatrix1"].provenance == Provenance.FABRICATED
    assert fields["AsShotNeutral"].provenance == Provenance.FABRICATED
    assert fields["NoiseProfile"].provenance == Provenance.DETERMINISTIC
    assert all(record.to_dict()["dng_type"] for record in fields.values())
    assert "VIRTUAL_16BIT_SENSOR" in fields["Model"].value if bits == 16 else "Sony" in fields["Model"].value


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cfa_phase": "RGBG"},
        {"black_level": 16383},
        {"black_level": -1},
        {"white_level": 16384},
        {"white_level": 1.5},
        {"black_level": float("nan")},
        {"color_matrix1": ((1, 0, 0), (1, 0, 0), (0, 0, 1))},
        {"color_matrix2": ((float("inf"), 0, 0), (0, 1, 0), (0, 0, 1))},
        {"color_matrix1": (1, 2, 3)},
        {"as_shot_neutral": (0, 1, 1)},
        {"noise_profile": (0, 1)},
        {"noise_profile": (1, -1)},
        {"noise_profile": (1, 0, 1, 0)},
        {"software_version": ""},
    ],
)
def test_invalid_inputs(geometry, kwargs):
    with pytest.raises(ValueError):
        DNGMetadataBuilder().build(geometry, **kwargs)


@pytest.mark.parametrize(
    "name,value",
    [
        ("Make", "SONY"),
        ("ImageDescription", "camera capture"),
        ("BitsPerSample", 14),
        ("NoiseReductionApplied", 1),
        ("DefaultCropSize", (65, 48)),
        ("ActiveArea", (0, 0, 49, 64)),
        ("DefaultCropOrigin", (-1, 0)),
        ("BaselineNoise", 0),
        ("LinearResponseLimit", 1.1),
        ("BaselineExposure", -6),
    ],
)
def test_mutated_records_revalidated(geometry, name, value):
    builder = DNGMetadataBuilder()
    fields = builder.build(geometry)
    fields[name].value = value
    with pytest.raises(ValueError):
        builder.validate(fields, ExperimentConfig())


@pytest.mark.parametrize(
    "name,value,user",
    [
        ("MakerNote", "binary", True),
        ("DateTimeOriginal", "2026:01:01 00:00:00", False),
        ("GPSLatitude", (1, 2, 3), False),
        ("BodySerialNumber", "1234567", True),
    ],
)
def test_prohibited_metadata(geometry, name, value, user):
    builder = DNGMetadataBuilder()
    fields = builder.build(geometry)
    fields[name] = DNGFieldRecord(
        name,
        value,
        Provenance.FABRICATED,
        "user input",
        user_override=user,
        dng_type="ASCII" if isinstance(value, str) else "RATIONAL[3]",
    )
    with pytest.raises(ValueError):
        builder.validate(fields, ExperimentConfig())


def test_crop_relative_to_active_area(geometry):
    geometry = replace(geometry, active_area=(4, 6, 44, 60), default_crop_origin=(2, 2), default_crop_size=(50, 36))
    fields = DNGMetadataBuilder().build(geometry)
    assert fields["ActiveArea"].value == (4, 6, 44, 60)


def test_virtual_cinema_identity(geometry):
    fields = DNGMetadataBuilder().build(geometry, ExperimentConfig(OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL"))
    assert fields["UniqueCameraModel"].value == "Synthetic Virtual Cinema 50MP v1"
