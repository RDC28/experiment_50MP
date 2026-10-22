import json

import cv2
import numpy as np
import OpenEXR
import pytest
import tifffile
from PIL import Image

from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import exporters

METADATA = {"run_id": "test", "notice": "synthetic", "software": "synthetic_raw_lab test"}


def sample_uint16(height=37, width=53):
    data = np.random.default_rng(0).integers(0, 65536, size=(height, width, 3), dtype=np.uint16)
    data[0, 0] = (1, 2, 3)  # low-order bits that an 8-bit round trip would destroy
    return data


@pytest.mark.parametrize("compression", ["zlib", "none"])
def test_tiff16_roundtrip(tmp_path, compression):
    data = sample_uint16()
    result = exporters.write_tiff16(tmp_path / "m.tif", data, METADATA, compression=compression)

    assert result.valid and all(result.checks.values())
    assert (result.width, result.height, result.sample_type, result.channels) == (53, 37, "uint16", 3)
    with tifffile.TiffFile(result.path) as tif:
        page = tif.pages[0]
        assert page.bitspersample == 16 and page.samplesperpixel == 3
        assert bytes(page.tags[34675].value) == cm.srgb_icc_profile()
        assert json.loads(page.description) == METADATA
        assert page.compression == (
            tifffile.COMPRESSION.ADOBE_DEFLATE if compression == "zlib" else tifffile.COMPRESSION.NONE
        )
        assert np.array_equal(page.asarray(), data)
    # An independent decoder sees the same 16-bit values in the same channel order.
    other = cv2.imread(str(result.path), cv2.IMREAD_UNCHANGED)
    assert other.dtype == np.uint16
    assert np.array_equal(cv2.cvtColor(other, cv2.COLOR_BGR2RGB), data)
    assert result.size_bytes == result.path.stat().st_size


def test_tiff16_rejects_wrong_input(tmp_path):
    with pytest.raises(ValueError):
        exporters.write_tiff16(tmp_path / "m.tif", np.zeros((4, 4, 3), np.uint8), METADATA)
    with pytest.raises(ValueError):
        exporters.write_tiff16(tmp_path / "m.tif", np.zeros((4, 4, 3), np.uint16), METADATA, compression="lzw")


@pytest.mark.parametrize(("precision", "dtype"), [("half", np.float16), ("float", np.float32)])
def test_exr_roundtrip(tmp_path, precision, dtype):
    linear = np.random.default_rng(1).normal(0.4, 0.5, size=(29, 41, 3)).astype(np.float32)  # includes <0 and >1
    attributes = {"comments": "synthetic", "colorSpaceDescription": exporters.EXR_COLOR_DESCRIPTION}
    result = exporters.write_exr(tmp_path / "m.exr", linear, attributes, precision=precision, compression="zip")

    assert result.valid
    assert (result.width, result.height, result.sample_type) == (41, 29, np.dtype(dtype).name)
    with OpenEXR.File(str(result.path), separate_channels=True) as exr:
        header = exr.header()
        channels = exr.channels()
        assert sorted(channels) == ["B", "G", "R"]
        for index, name in enumerate("RGB"):
            plane = channels[name].pixels
            assert plane.dtype == dtype and plane.shape == (29, 41)
            assert np.array_equal(plane, linear[..., index].astype(dtype))
        assert np.allclose(header["chromaticities"], exporters.REC709_CHROMATICITIES, atol=1e-6)
        assert header["comments"] == "synthetic"
        assert "not calibrated scene-referred" in header["colorSpaceDescription"]
        assert header["compression"] == OpenEXR.ZIP_COMPRESSION


def test_exr_rejects_bad_arguments(tmp_path):
    linear = np.zeros((4, 4, 3), np.float32)
    with pytest.raises(ValueError):
        exporters.write_exr(tmp_path / "m.exr", linear, {}, precision="double")
    with pytest.raises(ValueError):
        exporters.write_exr(tmp_path / "m.exr", linear, {}, compression="dwaa")
    with pytest.raises(ValueError):
        exporters.write_exr(tmp_path / "m.exr", linear, {"count": 3})


def test_failed_check_raises(tmp_path):
    result = exporters.ExportResult(tmp_path / "x", "k", 1, 1, "uint16", 3, "NONE", "", "", 0, {"decodes": False})
    assert not result.valid
    with pytest.raises(exporters.ExportValidationError, match="decodes"):
        exporters._raise_if_invalid(result)


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
def test_review_images_embed_srgb(tmp_path, suffix):
    rgb = np.random.default_rng(2).integers(0, 256, size=(20, 30, 3), dtype=np.uint8)
    path = exporters.write_review_image(tmp_path / f"r{suffix}", rgb)
    with Image.open(path) as image:
        assert image.size == (30, 20) and image.mode == "RGB"
        assert image.info["icc_profile"] == cm.srgb_icc_profile()
        if suffix == ".png":
            assert np.array_equal(np.asarray(image), rgb)


def test_sidecar(tmp_path):
    result = exporters.write_tiff16(tmp_path / "m.tif", sample_uint16(8, 8), METADATA)
    sidecar = exporters.write_sidecar(result, {"run_id": "test"})
    record = json.loads(sidecar.read_text())
    assert record["export"]["pixel_sha256"] == result.pixel_sha256
    assert record["provenance"] == {"run_id": "test"}
