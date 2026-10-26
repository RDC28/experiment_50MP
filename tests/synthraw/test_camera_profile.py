"""Tiny, allocation-free checks of PRD_v3 built-in camera priors."""

import json

import pytest

from synthraw.camera_profile import PROFILE_NAMES, SENSOR_PRIOR_DISCLAIMER, CameraProfileManager
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
        for item in (record, interval):
            assert item.provenance == Provenance.FABRICATED
            assert item.confidence is None
            assert item.disclaimer == SENSOR_PRIOR_DISCLAIMER


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
