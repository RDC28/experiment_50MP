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
    # A bare 64 x 48 sensor: the planned 50 MP content box and outpaint bands do not apply to it.
    return replace(
        plan,
        width=64,
        height=48,
        content_box=(0.0, 0.0, 64.0, 48.0),
        outpaint_bands=(),
        active_area=(0, 0, 48, 64),
        default_crop_origin=(0, 0),
        default_crop_size=(64, 48),
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


def exposure_record(name, value, provenance=Provenance.INFERRED):
    from synthraw.provenance import FieldRecord

    return FieldRecord(name, value, provenance, "manual hypothetical exposure", user_override=True)


def test_exposure_provenance_and_apex_derivation(geometry):
    exposure = {
        name: exposure_record(name, value)
        for name, value in (("ExposureTime", 1 / 128), ("FNumber", 4), ("PhotographicSensitivity", 100))
    }
    exposure["ExposureTime"].confidence = 0.6
    exposure["ExposureTime"].alternatives = [{"value": 1 / 64}]
    fields = DNGMetadataBuilder().build(geometry, exposure=exposure)
    assert fields["ShutterSpeedValue"].value == 7
    assert fields["ApertureValue"].value == 4
    for name in exposure:
        expected = exposure[name].to_dict()
        assert {key: fields[name].to_dict()[key] for key in expected} == expected
    assert fields["ShutterSpeedValue"].dependencies == ["ExposureTime"]
    assert fields["ApertureValue"].provenance == Provenance.DETERMINISTIC
    fields["ExposureTime"].alternatives.append({"value": 1 / 32})
    assert len(exposure["ExposureTime"].alternatives) == 1


@pytest.mark.parametrize(
    "name,value",
    [
        ("ExposureTime", 0),
        ("ExposureTime", -1),
        ("ExposureTime", 1 / 64000),
        ("ExposureTime", float("nan")),
        ("ExposureTime", True),
        ("FNumber", 0.5),
        ("FNumber", float("inf")),
        ("PhotographicSensitivity", 49),
        ("PhotographicSensitivity", 102401),
        ("PhotographicSensitivity", 100.5),
        ("ExposureBiasValue", 6),
        ("ExposureBiasValue", [0]),
        ("ShutterSpeedValue", 7),
    ],
)
def test_invalid_exposure_inputs(geometry, name, value):
    with pytest.raises(ValueError):
        DNGMetadataBuilder().build(geometry, exposure={name: exposure_record(name, value)})


@pytest.mark.parametrize("iso", (50, 102400))
def test_expanded_iso_limits(geometry, iso):
    fields = DNGMetadataBuilder().build(
        geometry, exposure={"PhotographicSensitivity": exposure_record("PhotographicSensitivity", iso)}
    )
    assert fields["PhotographicSensitivity"].value == iso


@pytest.mark.parametrize("change", ("source", "derived", "missing", "provenance", "type"))
def test_exposure_dependencies_revalidated(geometry, change):
    builder = DNGMetadataBuilder()
    fields = builder.build(geometry, exposure={"ExposureTime": exposure_record("ExposureTime", 1 / 128)})
    if change == "source":
        fields["ExposureTime"].value = 1 / 64
    elif change == "derived":
        fields["ShutterSpeedValue"].value = 8
    elif change == "missing":
        del fields["ExposureTime"]
    elif change == "provenance":
        fields["ExposureTime"].provenance = Provenance.MEASURED
    else:
        fields["ExposureTime"].dng_type = "SHORT"
    with pytest.raises(ValueError):
        builder.validate(fields, ExperimentConfig())


def test_lens_metadata_roundtrip(geometry, tmp_path):
    import numpy as np
    import rawpy

    from synthraw.dng_writer import DNGWriter

    lens = {
        name: exposure_record(name, value, provenance)
        for name, value, provenance in (
            ("LensInfo", (24, 70, 2.8, 4), Provenance.MEASURED),
            ("FocalLength", 50, Provenance.INFERRED),
            # tifffile's canonical name for EXIF FocalLengthIn35mmFormat.
            ("FocalLengthIn35mmFilm", 50, Provenance.DETERMINISTIC),
            ("LensMake", "Synthetic", Provenance.FABRICATED),
            ("LensModel", "Hypothetical 24-70mm", Provenance.FABRICATED),
        )
    }
    fields = DNGMetadataBuilder().build(geometry, lens=lens)
    for name, source in lens.items():
        expected = source.to_dict()
        assert {key: fields[name].to_dict()[key] for key in expected} == expected
    mosaic = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) + 512
    path = DNGWriter().write(tmp_path / "lens.dng", mosaic, fields)
    with rawpy.imread(str(path)) as raw:
        assert np.array_equal(raw.raw_image, mosaic)


@pytest.mark.parametrize(
    "name,value,provenance",
    [
        ("LensInfo", (70, 24, 4, 2.8), Provenance.MEASURED),
        ("LensInfo", (24, 70, 0, 4), Provenance.INFERRED),
        ("LensInfo", (24, 70, 2.8), Provenance.FABRICATED),
        ("LensInfo", (24, 70, float("nan"), 4), Provenance.MEASURED),
        ("LensInfo", ("24", "70", "2.8", "4"), Provenance.MEASURED),
        ("FocalLength", True, Provenance.INFERRED),
        ("FocalLength", -50, Provenance.INFERRED),
        ("FocalLength", 50, Provenance.FABRICATED),
        ("FocalLengthIn35mmFilm", 65536, Provenance.DETERMINISTIC),
        ("FocalLengthIn35mmFilm", 50.5, Provenance.INFERRED),
        ("LensMake", "", Provenance.FABRICATED),
        ("LensModel", "Lens\x00hidden", Provenance.INFERRED),
        ("LensModel", "Lens", Provenance.DETERMINISTIC),
        ("UnknownLensField", 50, Provenance.INFERRED),
    ],
)
def test_invalid_lens_metadata(geometry, name, value, provenance):
    with pytest.raises(ValueError):
        DNGMetadataBuilder().build(geometry, lens={name: exposure_record(name, value, provenance)})


def test_mutated_lens_range_rejected(geometry):
    builder = DNGMetadataBuilder()
    fields = builder.build(
        geometry,
        lens={
            "LensInfo": exposure_record("LensInfo", (24, 70, 2.8, 4)),
            "FocalLength": exposure_record("FocalLength", 50),
        },
    )
    fields["FocalLength"].value = 85
    with pytest.raises(ValueError, match="outside selected lens range"):
        builder.validate(fields, ExperimentConfig())


def test_without_synthetic_exif_only_user_capture_fields_are_accepted(geometry):
    from synthraw.provenance import FieldRecord

    config = ExperimentConfig(GENERATE_SYNTHETIC_EXIF=False)
    plain = DNGMetadataBuilder().build(geometry, config)
    assert not {"ExposureTime", "FNumber", "PhotographicSensitivity", "ShutterSpeedValue"} & plain.keys()
    inferred = {"ExposureTime": FieldRecord("ExposureTime", 1 / 128, Provenance.INFERRED, "exposure posterior")}
    with pytest.raises(ValueError, match="GENERATE_SYNTHETIC_EXIF"):
        DNGMetadataBuilder().build(geometry, config, exposure=inferred)
    lens = {"FocalLength": FieldRecord("FocalLength", 35.0, Provenance.INFERRED, "FOV hypothesis")}
    with pytest.raises(ValueError, match="GENERATE_SYNTHETIC_EXIF"):
        DNGMetadataBuilder().build(geometry, config, lens=lens)
    user = {"ExposureTime": exposure_record("ExposureTime", 1 / 128)}
    fields = DNGMetadataBuilder().build(geometry, config, exposure=user)
    assert fields["ExposureTime"].user_override and fields["ShutterSpeedValue"].value == 7
    assert DNGMetadataBuilder().build(geometry, exposure=inferred)["ExposureTime"].value == 1 / 128


def test_outpaint_content_box_is_the_default_user_crop():
    """Regression: 36% of an outpainted frame was invented bands with nothing in the DNG saying so."""
    config = ExperimentConfig(OUTPUT_GEOMETRY="SONY_A1II_NATIVE", SENSOR_FIT_MODE="OUTPAINT_VERTICAL")
    geometry = TargetGeometryPlanner().plan(6336, 2688, config)
    assert geometry.outpaint_bands and geometry.default_crop_size == (8640, 5760)
    fields = DNGMetadataBuilder().build(geometry, config)
    top, left, bottom, right = fields["DefaultUserCrop"].value
    # The 8640 x 3665.45 content box, rounded inward to whole Bayer tiles.
    assert (left, right) == (0, 1)
    assert top * 5760 == pytest.approx(1048) and bottom * 5760 == pytest.approx(4712)
    assert geometry.content_box[1] <= top * 5760 and bottom * 5760 <= geometry.content_box[3]


@pytest.mark.parametrize("mode", ["CROP_HORIZONTAL", "FIT_WITH_SENSOR_MARGINS"])
def test_frames_without_outpaint_bands_have_no_user_crop(mode):
    config = ExperimentConfig(OUTPUT_GEOMETRY="SONY_A1II_NATIVE", SENSOR_FIT_MODE=mode)
    geometry = TargetGeometryPlanner().plan(6336, 2688, config)
    assert "DefaultUserCrop" not in DNGMetadataBuilder().build(geometry, config)
