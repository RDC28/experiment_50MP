"""Tiny real DNG development checks for the PRD_v3 master export."""

import io

import numpy as np
import pytest
import rawpy
import tifffile
from PIL import ImageCms
from test_raw_gate import tiny_raw

from synthraw.exporters import ADOBE_GAMMA, TIFFWriter, adobe_rgb_icc
from synthraw.libraw_validator import HardValidationError


def test_icc_is_accepted_by_colour_management():
    profile = ImageCms.ImageCmsProfile(io.BytesIO(adobe_rgb_icc()))
    assert "Adobe RGB" in ImageCms.getProfileDescription(profile)
    ImageCms.buildTransformFromOpenProfiles(profile, ImageCms.createProfile("sRGB"), "RGB", "RGB")


def test_developed_master_matches_independent_raw_path(tmp_path):
    dng, _, _ = tiny_raw(tmp_path)
    with rawpy.imread(str(dng)) as raw:
        expected = raw.postprocess(
            output_bps=16,
            output_color=rawpy.ColorSpace.Adobe,
            gamma=(ADOBE_GAMMA, 0),
            use_camera_wb=True,
            no_auto_bright=True,
            user_flip=0,
        )
    path = TIFFWriter().write(tmp_path / "developed_50mp_16bit.tif", dng, rows_per_strip=7)
    with tifffile.TiffFile(path) as container:
        page = container.pages[0]
        assert page.dtype == np.uint16
        assert page.shape == (48, 64, 3)
        assert len(page.dataoffsets) == 7
        assert page.tags[34675].value == adobe_rgb_icc()
        np.testing.assert_array_equal(page.asarray(), expected)
    mapped = tifffile.memmap(path, mode="r+")
    mapped[0, 0, 0] ^= 1
    mapped.flush()
    mapped._mmap.close()
    with pytest.raises(HardValidationError, match="pixels differ"):
        TIFFWriter.validate(path, expected, adobe_rgb_icc())


def test_bad_raw_does_not_write_successful_master(tmp_path):
    source = tmp_path / "bad.dng"
    source.write_bytes(b"invalid")
    destination = tmp_path / "master.tif"
    with pytest.raises(HardValidationError, match="RAW development failed"):
        TIFFWriter().write(destination, source)
    assert not destination.exists()


@pytest.mark.parametrize("rows", (0, -1, True, 1.5))
def test_bad_strip_size(tmp_path, rows):
    with pytest.raises(ValueError, match="positive integer"):
        TIFFWriter().write(tmp_path / "master.tif", tmp_path / "unused.dng", rows_per_strip=rows)


def test_exr_preserves_unclamped_hdr(tmp_path):
    import OpenEXR

    from synthraw.exporters import EXRWriter

    rgb = np.linspace(-0.2, 12, 48 * 20 * 3, dtype=np.float32).reshape(20, 48, 3)
    original = rgb.copy()
    path = EXRWriter().write(tmp_path / "masters/scene_linear_50mp.exr", rgb)
    with OpenEXR.File(str(path), separate_channels=True) as exr:
        for i, channel in enumerate("RGB"):
            np.testing.assert_array_equal(exr.channels()[channel].pixels, original[..., i])
        np.testing.assert_allclose(exr.header()["chromaticities"], (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290))
    np.testing.assert_array_equal(rgb, original)


def test_preview_is_bounded_tagged_srgb(tmp_path):
    from PIL import Image

    from synthraw.exporters import PreviewWriter

    rgb = np.full((20, 48, 3), (-0.1, 0.18, 4), dtype=np.float32)
    path = PreviewWriter().write(tmp_path / "masters/preview_8bit.png", rgb, max_side=24)
    with Image.open(path) as image:
        assert image.size == (24, 10)
        assert image.mode == "RGB"
        np.testing.assert_array_equal(np.asarray(image)[0, 0], [0, 118, 255])
        profile = ImageCms.ImageCmsProfile(io.BytesIO(image.info["icc_profile"]))
        assert "sRGB" in ImageCms.getProfileDescription(profile)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 2)),
        np.zeros((0, 2, 3)),
        np.ones((2, 2, 3), dtype=np.uint8),
        np.full((2, 2, 3), np.nan),
        np.full((2, 2, 3), 1e100),
    ],
)
def test_master_export_rejects_invalid_radiance(tmp_path, bad):
    from synthraw.exporters import EXRWriter, PreviewWriter

    for writer in (EXRWriter(), PreviewWriter()):
        destination = tmp_path / "invalid"
        with pytest.raises(ValueError):
            writer.write(destination, bad)
        assert not destination.exists()


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_preview_rejects_invalid_bound(tmp_path, size):
    from synthraw.exporters import PreviewWriter

    with pytest.raises(ValueError, match="positive integer"):
        PreviewWriter().write(tmp_path / "preview.png", np.zeros((2, 2, 3)), max_side=size)
