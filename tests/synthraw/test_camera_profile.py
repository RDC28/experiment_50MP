"""Tiny, allocation-free checks of PRD_v3 built-in camera priors."""

import json

import pytest

from synthraw.camera_profile import (
    ILCE1M2_SAMPLE_DISCLAIMER,
    ILCE1M2_XYZ_TO_CAMERA_D65,
    PROFILE_NAMES,
    SENSOR_PRIOR_DISCLAIMER,
    CameraProfileManager,
)
from synthraw.provenance import FieldRecord, Provenance


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_profile_geometry_precision_and_identity(name):
    profile = CameraProfileManager().build(name)
    camera, sensor = profile.camera, profile.sensor
    assert camera["Make"].value == "Synthetic"
    assert camera["Model"].value == name
    assert camera["native_sony_geometry"].value == (name == "SONY_A1II_LIKE")
    expected = (10872, 4608) if name == "CINEMA_50MP_VIRTUAL" else (8640, 5760)
    assert (sensor["sensor_width"].value, sensor["sensor_height"].value) == expected
    bits = 16 if name == "VIRTUAL_16BIT_SENSOR" else 14
    assert sensor["adc_bits"].value == bits
    assert sensor["WhiteLevel"].value == 2**bits - 1
    assert sensor["dng_storage_bits"].value == 16
    assert 0 <= sensor["BlackLevel"].value < sensor["WhiteLevel"].value
    assert all(isinstance(record, FieldRecord) for record in (*camera.values(), *sensor.values()))


def test_default_operating_envelope():
    camera = CameraProfileManager().build().camera
    assert camera["profile_source"].value == "IMAGE_ONLY"
    for key, value in {
        "sensor_width_mm": 35.9,
        "sensor_height_mm": 24.0,
        "iso_standard_range": [100, 32000],
        "iso_expanded_range": [50, 102400],
        "colour_temperature_range": [2500, 9900],
        "mechanical_shutter_min_seconds": 1 / 8000,
        "electronic_shutter_min_seconds": 1 / 32000,
    }.items():
        assert camera[key].value == value


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_sensor_priors_are_fabricated_and_disclaimed(name):
    sensor = CameraProfileManager().build(name).sensor
    for field in ("full_well_electrons", "read_noise_electrons", "prnu_sigma", "dsnu_sigma_dn"):
        record, interval = sensor[field], sensor[f"{field}_range"]
        assert interval.value[0] <= record.value <= interval.value[1]
        # The Sony-like read noise is a single-frame bound from the ILCE-1M2 sample, not a pure prior.
        sampled = name == "SONY_A1II_LIKE" and field == "read_noise_electrons"
        for item in (interval,) if sampled else (record, interval):
            assert item.provenance == Provenance.FABRICATED
            assert item.confidence is None
            assert item.disclaimer == SENSOR_PRIOR_DISCLAIMER
        if sampled:
            assert record.provenance == Provenance.INFERRED
            assert record.disclaimer == ILCE1M2_SAMPLE_DISCLAIMER


def test_sony_like_profile_carries_the_measured_ilce1m2_sample():
    profile = CameraProfileManager().build("SONY_A1II_LIKE")
    camera, sensor = profile.camera, profile.sensor
    assert camera["xyz_to_camera_d65"].value == ILCE1M2_XYZ_TO_CAMERA_D65
    neutral = camera["reference_as_shot_neutral"].value
    assert neutral[0] < 1 and neutral[2] < 1
    for record in (camera["xyz_to_camera_d65"], sensor["conversion_gain_e_per_dn"]):
        assert record.provenance == Provenance.MEASURED
        assert record.disclaimer == ILCE1M2_SAMPLE_DISCLAIMER
    # Full well lands at the ADC ceiling with the measured conversion gain.
    span = sensor["WhiteLevel"].value - sensor["BlackLevel"].value
    assert sensor["full_well_electrons"].value == pytest.approx(
        span * sensor["conversion_gain_e_per_dn"].value, rel=0.01
    )
    assert camera["lens_distortion_k1"].value > 0
    virtual = CameraProfileManager().build("CINEMA_50MP_VIRTUAL")
    assert "xyz_to_camera_d65" not in virtual.camera


def test_sidecars_round_trip_and_fresh_profiles(tmp_path):
    manager = CameraProfileManager()
    profile = manager.build()
    paths = profile.write(tmp_path / "profiles")
    assert [path.name for path in paths] == ["camera_profile.json", "sensor_profile.json"]
    for path, records in zip(paths, (profile.camera, profile.sensor), strict=True):
        decoded = [FieldRecord.from_dict(item) for item in json.loads(path.read_text())]
        assert decoded == list(records.values())
    profile.camera["iso_standard_range"].value[0] = 999
    assert manager.build().camera["iso_standard_range"].value == [100, 32000]


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match="Unknown camera profile"):
        CameraProfileManager().build("SONY")


@pytest.mark.parametrize("phase", ("RGGB", "BGGR", "GRBG", "GBRG"))
def test_reference_raw_real_decoder_round_trip(tmp_path, phase):
    from dataclasses import replace

    import numpy as np

    from synthraw.dng_metadata import DNGMetadataBuilder
    from synthraw.dng_writer import DNGWriter
    from synthraw.geometry import TargetGeometryPlanner

    geometry = replace(
        TargetGeometryPlanner().plan(64, 48),
        outpaint_bands=(),  # The bare test sensor has no planned outpaint bands.
        width=64,
        height=48,
        active_area=(0, 0, 48, 64),
        default_crop_origin=(0, 0),
        default_crop_size=(64, 48),
    )
    records = DNGMetadataBuilder().build(geometry, cfa_phase=phase, as_shot_neutral=(0.5, 1, 0.75))
    path = DNGWriter().write(tmp_path / "reference.dng", np.full((48, 64), 1024, dtype=np.uint16), records)
    profile = CameraProfileManager().build(source="REFERENCE_RAW", reference_raw=path)
    assert profile.sensor["cfa_pattern"].value == phase
    assert profile.sensor["sensor_width"].value == 64
    assert profile.sensor["sensor_height"].value == 48
    assert profile.sensor["WhiteLevel"].value == 16383
    assert profile.sensor["BlackLevel"].value == 512.0
    assert profile.sensor["black_level_per_channel"].value == [512] * 4
    assert profile.camera["reference_as_shot_neutral"].value == pytest.approx([0.5, 1, 0.75])
    assert "AsShotNeutral" not in profile.camera
    assert profile.camera["reference_raw"].value == str(path)
    assert profile.camera["profile_source"].value == "REFERENCE_RAW"
    assert "libraw_rgb_xyz_matrix" not in profile.camera
    assert profile.camera["Make"].value == "Synthetic"
    assert profile.sensor["read_noise_electrons"].provenance != Provenance.MEASURED
    for field in ("camera_whitebalance", "reference_as_shot_neutral", "libraw_color_matrix"):
        record = profile.camera[field]
        assert record.provenance == Provenance.MEASURED
        assert record.dependencies == ["reference_raw"]
    assert all(
        record.provenance == Provenance.MEASURED
        for record in profile.sensor.values()
        if record.field
        in ("cfa_pattern", "BlackLevel", "black_level_per_channel", "WhiteLevel", "sensor_width", "active_area")
    )
    for output in profile.write(tmp_path / "sidecars"):
        assert json.loads(output.read_text())


def test_reference_requires_explicit_file_and_mode(tmp_path):
    manager = CameraProfileManager()
    with pytest.raises(ValueError, match="requires"):
        manager.build(source="REFERENCE_RAW")
    with pytest.raises(ValueError, match="Select REFERENCE_RAW"):
        manager.build(reference_raw=tmp_path / "unused.raw")
    with pytest.raises(FileNotFoundError):
        manager.build(source="REFERENCE_RAW", reference_raw=tmp_path / "missing.raw")
    with pytest.raises(ValueError, match="REFERENCE_RAW_PATH"):
        manager.build(source="FULL_CALIBRATION")


def test_invalid_reference_fails_without_synthetic_fallback(tmp_path):
    path = tmp_path / "invalid.raw"
    path.write_bytes(b"not a RAW")
    import rawpy

    with pytest.raises(rawpy.LibRawError):
        CameraProfileManager().build(source="REFERENCE_RAW", reference_raw=path)


@pytest.mark.parametrize("aspect", [(4, 3), (1, 1), (2048, 868)])
@pytest.mark.parametrize("settings", [{}, {"ADC_BITS": 16}, {"EXPERIMENTAL_16BIT_SENSOR": True}])
def test_configured_profile_matches_planned_export(tmp_path, aspect, settings):
    """Exercise notebook profile -> sidecars -> DNG using a tiny export."""
    from dataclasses import replace

    import numpy as np
    import tifffile

    from synthraw.config import ExperimentConfig
    from synthraw.dng_metadata import DNGMetadataBuilder
    from synthraw.dng_writer import DNGWriter
    from synthraw.geometry import TargetGeometryPlanner

    config = ExperimentConfig(OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL", **settings)
    geometry = TargetGeometryPlanner().plan(*aspect, config)
    profile = CameraProfileManager().build(config=config, geometry=geometry)
    assert profile.camera["Model"].value == config.TARGET_PROFILE
    assert not profile.camera["native_sony_geometry"].value
    for field, expected in {
        "sensor_width": geometry.width,
        "sensor_height": geometry.height,
        "active_area": list(geometry.active_area),
        "default_crop_origin": list(geometry.default_crop_origin),
        "default_crop_size": list(geometry.default_crop_size),
        "adc_bits": config.ADC_BITS,
    }.items():
        assert profile.sensor[field].value == expected
    profile.write(tmp_path)
    # No full-resolution allocation: only the arithmetic plan above uses 50 MP.
    tiny = replace(geometry, width=64, height=48, active_area=(0, 0, 48, 64), default_crop_size=(64, 48))
    records = DNGMetadataBuilder().build(tiny, config)
    path = DNGWriter().write(tmp_path / "synthetic.dng", np.full((48, 64), 1024, dtype=np.uint16), records, config)
    with tifffile.TiffFile(path) as raw:
        tags = raw.pages[0].tags
        assert "virtual" in tags[272].value.lower()
        assert "virtual" in tags[50708].value.lower()
        assert tags[50717].value == profile.sensor["WhiteLevel"].value


@pytest.mark.parametrize("settings", [{}, {"ADC_BITS": 16}, {"EXPERIMENTAL_16BIT_SENSOR": True}])
def test_native_config_identity_and_margin_geometry(settings):
    from synthraw.config import ExperimentConfig
    from synthraw.geometry import TargetGeometryPlanner

    config = ExperimentConfig(SENSOR_FIT_MODE="FIT_WITH_SENSOR_MARGINS", **settings)
    geometry = TargetGeometryPlanner().plan(16, 9, config)
    profile = CameraProfileManager().build(config=config, geometry=geometry)
    assert profile.camera["Model"].value == config.TARGET_PROFILE
    assert profile.camera["native_sony_geometry"].value == (config.ADC_BITS == 14)
    assert profile.sensor["adc_bits"].value == config.ADC_BITS
    assert profile.sensor["default_crop_origin"].value == list(geometry.default_crop_origin)
    assert profile.sensor["default_crop_size"].value == list(geometry.default_crop_size)
