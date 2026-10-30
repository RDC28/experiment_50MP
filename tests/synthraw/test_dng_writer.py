"""Tiny structural round trips for PRD_v3 uncompressed and lossless JPEG CFA export."""

from dataclasses import replace

import numpy as np
import pytest
import rawpy
import tifffile

from synthraw.config import ExperimentConfig
from synthraw.dng_metadata import CFA_PATTERNS, DNGMetadataBuilder
from synthraw.dng_writer import DNGWriter, lossless_tile_shape
from synthraw.geometry import TargetGeometryPlanner
from synthraw.libraw_validator import LibRawValidator


@pytest.fixture
def fixture_data():
    geometry = replace(
        TargetGeometryPlanner().plan(64, 48),
        outpaint_bands=(),  # The bare test sensor has no planned outpaint bands.
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


def _geometry(width, height):
    return replace(
        TargetGeometryPlanner().plan(width, height),
        outpaint_bands=(),  # The bare test sensor has no planned outpaint bands.
        width=width,
        height=height,
        active_area=(0, 0, height, width),
        default_crop_origin=(0, 0),
        default_crop_size=(width, height),
    )


@pytest.mark.parametrize(
    ("shape", "tile"),
    (((48, 64), (48, 64)), ((50, 70), (48, 64)), ((300, 518), (256, 256)), ((512, 256), (256, 256))),
)
def test_lossless_tile_shape_is_a_multiple_of_16_within_the_image(shape, tile):
    assert lossless_tile_shape(shape) == tile


def test_lossless_rejects_images_smaller_than_one_tile_unit():
    with pytest.raises(ValueError, match="16 x 16"):
        lossless_tile_shape((14, 64))


@pytest.mark.parametrize("phase", CFA_PATTERNS)
@pytest.mark.parametrize("bits", (14, 16))
@pytest.mark.parametrize("shape", ((48, 64), (50, 70), (300, 518)))
def test_lossless_dng_reopens_in_libraw_bit_for_bit_with_the_uncompressed_mosaic(tmp_path, phase, bits, shape):
    """PRD optional lossless mode: LJ92 tiles must decode, in LibRaw, to exactly the uncompressed mosaic."""
    height, width = shape
    geometry = _geometry(width, height)
    rng = np.random.default_rng(height * width + bits)
    mosaic = rng.integers(512, (1 << bits) - 1, shape, dtype=np.uint16)
    paths = {}
    for compression in ("UNCOMPRESSED", "LOSSLESS"):
        config = ExperimentConfig(ADC_BITS=bits, DNG_COMPRESSION=compression)
        records = DNGMetadataBuilder().build(geometry, config, cfa_phase=phase, as_shot_neutral=(0.5, 1, 0.75))
        paths[compression] = DNGWriter().write(tmp_path / f"{compression}.dng", mosaic, records, config)
        LibRawValidator().validate(paths[compression], mosaic, records, develop=True)
    with tifffile.TiffFile(paths["LOSSLESS"]) as container:
        page = container.pages[0]
        assert page.compression == 7 and page.is_tiled and page.bitspersample == 16
        assert (page.tilelength, page.tilewidth) == lossless_tile_shape(shape)
        assert page.tags["Compression"].value == 7 and page.tags["Make"].value == "Synthetic"
        assert np.array_equal(page.asarray(), mosaic)
    with rawpy.imread(str(paths["LOSSLESS"])) as lossless, rawpy.imread(str(paths["UNCOMPRESSED"])) as plain:
        assert lossless.raw_image.dtype == np.uint16
        assert np.array_equal(lossless.raw_image, plain.raw_image)
        assert np.array_equal(lossless.raw_image, mosaic)


def test_lossless_dng_is_smaller_than_uncompressed_for_a_smooth_mosaic(tmp_path):
    geometry = _geometry(512, 256)
    rows, columns = np.mgrid[0:256, 0:512]
    mosaic = (512 + 20 * rows + 10 * columns).astype(np.uint16)
    sizes = {}
    for compression in ("UNCOMPRESSED", "LOSSLESS"):
        config = ExperimentConfig(DNG_COMPRESSION=compression)
        records = DNGMetadataBuilder().build(geometry, config)
        sizes[compression] = DNGWriter().write(tmp_path / f"{compression}.dng", mosaic, records, config).stat().st_size
    assert sizes["LOSSLESS"] < sizes["UNCOMPRESSED"] / 2


def test_lossless_readback_detects_a_corrupted_tile(tmp_path):
    geometry = _geometry(512, 256)
    mosaic = np.random.default_rng(4).integers(512, 16383, (256, 512), dtype=np.uint16)
    config = ExperimentConfig(DNG_COMPRESSION="LOSSLESS")
    records = DNGMetadataBuilder().build(geometry, config)
    writer = DNGWriter()
    path = writer.write(tmp_path / "corrupt.dng", mosaic, records, config)
    with tifffile.TiffFile(path) as container:
        page = container.pages[0]
        offset = page.dataoffsets[1] + page.databytecounts[1] // 2
    data = bytearray(path.read_bytes())
    data[offset] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="pixels differ"):
        writer.validate_structure(path, mosaic, records)


def test_compression_record_must_match_the_configured_mode(fixture_data):
    geometry, mosaic = fixture_data
    records = DNGMetadataBuilder().build(geometry)
    assert records["Compression"].value == 1
    with pytest.raises(ValueError, match="Compression"):
        DNGWriter().write("unused.dng", mosaic, records, ExperimentConfig(DNG_COMPRESSION="LOSSLESS"))
    assert DNGMetadataBuilder().build(geometry, ExperimentConfig(DNG_COMPRESSION="LOSSLESS"))["Compression"].value == 7
