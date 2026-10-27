"""Tiny structural round trips for PRD_v3 uncompressed CFA export."""

from dataclasses import replace

import numpy as np
import pytest
import tifffile

from synthraw.config import ExperimentConfig
from synthraw.dng_metadata import CFA_PATTERNS, DNGMetadataBuilder
from synthraw.dng_writer import DNGWriter
from synthraw.geometry import TargetGeometryPlanner


@pytest.fixture
def fixture_data():
    geometry = replace(
        TargetGeometryPlanner().plan(64, 48),
        width=64,
        height=48,
        active_area=(0, 0, 48, 64),
        default_crop_origin=(0, 0),
        default_crop_size=(64, 48),
    )
    return geometry, np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) + 512


@pytest.mark.parametrize("phase", CFA_PATTERNS)
@pytest.mark.parametrize("bits", (14, 16))
def test_stripped_round_trip(tmp_path, fixture_data, phase, bits):
    geometry, mosaic = fixture_data
    config = ExperimentConfig(ADC_BITS=bits)
    records = DNGMetadataBuilder().build(geometry, config, cfa_phase=phase, as_shot_neutral=(0.5, 1, 0.75))
    # A noncontiguous view exercises bounded strip packing.
    mosaic = np.repeat(mosaic, 2, axis=1)[:, ::2]
    path = DNGWriter().write(tmp_path / "gradient.dng", mosaic, records, config, rows_per_strip=7)
    with tifffile.TiffFile(path) as container:
        page = container.pages[0]
        assert len(page.dataoffsets) == 7
        assert page.compression == 1
        assert np.array_equal(page.asarray(), mosaic)
        assert page.tags["Make"].value == "Synthetic"
        assert page.tags["ColorMatrix1"].dtype.name == "SRATIONAL"
        assert page.tags["BlackLevel"].dtype.name == "RATIONAL"


@pytest.mark.parametrize("mutation", ("dtype", "shape", "range", "identity", "strip"))
def test_invalid_input_does_not_create_file(tmp_path, fixture_data, mutation):
    geometry, mosaic = fixture_data
    records = DNGMetadataBuilder().build(geometry)
    kwargs = {}
    if mutation == "dtype":
        mosaic = mosaic.astype(np.float32)
    elif mutation == "shape":
        mosaic = mosaic[:-2]
    elif mutation == "range":
        mosaic[0, 0] = 16384
    elif mutation == "identity":
        records["Make"].value = "SONY"
    else:
        kwargs["rows_per_strip"] = 0
    path = tmp_path / "invalid.dng"
    with pytest.raises(ValueError):
        DNGWriter().write(path, mosaic, records, **kwargs)
    assert not path.exists()


def test_readback_detects_corrupted_pixels(tmp_path, fixture_data):
    geometry, mosaic = fixture_data
    records = DNGMetadataBuilder().build(geometry)
    writer = DNGWriter()
    path = writer.write(tmp_path / "corrupt.dng", mosaic, records)
    mapped = tifffile.memmap(path, mode="r+")
    mapped[0, 0] += 1
    mapped.flush()
    mapped._mmap.close()
    with pytest.raises(ValueError, match="pixels differ"):
        writer.validate_structure(path, mosaic, records)
