"""User-path calibration regression through actual tiny DNGs and LibRaw."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from synthraw.camera_profile import CameraProfileManager
from synthraw.config import ExperimentConfig
from synthraw.dng_metadata import DNGMetadataBuilder
from synthraw.dng_writer import DNGWriter
from synthraw.geometry import TargetGeometryPlanner
from synthraw.provenance import Provenance


@pytest.fixture
def calibration(tmp_path):
    rng = np.random.default_rng(12)
    shape = (128, 128)
    geometry = replace(
        TargetGeometryPlanner().plan(128, 128),
        outpaint_bands=(),  # The bare test sensor has no planned outpaint bands.
        width=128,
        height=128,
        active_area=(0, 0, 128, 128),
        default_crop_size=(128, 128),
    )
    metadata = DNGMetadataBuilder().build(geometry)
    writer = DNGWriter()

    def save(name, values):
        return str(writer.write(tmp_path / name, np.rint(values).astype(np.uint16), metadata))

    offset = rng.normal(0, 8, shape)
    response = rng.normal(1, 0.03, shape)
    darks = tuple(save(f"dark{i}.dng", 512 + offset + rng.normal(0, 3, shape)) for i in range(8))
    flats = tuple(
        save(f"flat{level}_{i}.dng", 512 + offset + rng.poisson(level * response * 2) / 2 + rng.normal(0, 3, shape))
        for level in (1000, 4000)
        for i in range(2)
    )
    return ExperimentConfig(
        CAMERA_PROFILE_SOURCE="FULL_CALIBRATION",
        REFERENCE_RAW_PATH=darks[0],
        DARK_FRAME_PATHS=darks,
        FLAT_FIELD_PATHS=flats,
    )


def test_full_calibration_from_user_dngs(calibration, tmp_path):
    profile = CameraProfileManager().build(config=calibration)
    assert profile.camera["profile_source"].value == "FULL_CALIBRATION"
    for field, expected, tolerance in (
        ("BlackLevel", 512, 0.01),
        ("conversion_gain_e_per_dn", 2, 0.08),
        ("read_noise_electrons", 6, 0.08),
        ("dsnu_sigma_dn", 8, 0.05),
        ("prnu_sigma", 0.03, 0.08),
    ):
        record = profile.sensor[field]
        assert record.value == pytest.approx(expected, rel=tolerance)
        assert record.provenance == Provenance.MEASURED
        assert record.dependencies
    assert profile.sensor["full_well_electrons"].provenance == Provenance.FABRICATED
    profile.write(tmp_path / "sidecars")
    assert ExperimentConfig.from_dict(calibration.to_dict()) == calibration


def test_missing_calibration_assets_are_named():
    with pytest.raises(ValueError, match="REFERENCE_RAW_PATH"):
        CameraProfileManager().build(config=ExperimentConfig(CAMERA_PROFILE_SOURCE="FULL_CALIBRATION"))


def test_unreadable_calibration_names_path(calibration, tmp_path):
    bad = tmp_path / "bad.dng"
    bad.write_text("not a RAW")
    with pytest.raises(ValueError, match="bad.dng"):
        CameraProfileManager().build(config=replace(calibration, DARK_FRAME_PATHS=(str(bad),) * 2))


def test_flat_pairs_require_signal_range(calibration):
    with pytest.raises(ValueError, match="distinct signal levels"):
        CameraProfileManager().build(config=replace(calibration, FLAT_FIELD_PATHS=calibration.FLAT_FIELD_PATHS[:2] * 2))


@pytest.mark.parametrize("field,paths", [("DARK_FRAME_PATHS", ()), ("FLAT_FIELD_PATHS", ())])
def test_missing_frame_groups(calibration, field, paths):
    with pytest.raises(ValueError, match=field):
        CameraProfileManager().build(config=replace(calibration, **{field: paths}))


def test_missing_frame_names_path(calibration, tmp_path):
    missing = str(tmp_path / "absent.dng")
    with pytest.raises(FileNotFoundError, match="absent.dng"):
        CameraProfileManager().build(config=replace(calibration, DARK_FRAME_PATHS=(missing,) * 2))


def test_clipped_flats_rejected(calibration, tmp_path):
    import tifffile

    bad = tmp_path / "clipped.dng"
    # Preserve valid DNG metadata while changing only one pixel through the mapped raster.
    bad.write_bytes(Path(calibration.FLAT_FIELD_PATHS[0]).read_bytes())
    mapped = tifffile.memmap(bad, mode="r+")
    mapped[0, 0] = 16383
    mapped.flush()
    mapped._mmap.close()
    with pytest.raises(ValueError, match="Clipped.*clipped.dng"):
        CameraProfileManager().build(config=replace(calibration, FLAT_FIELD_PATHS=(str(bad),) * 4))
