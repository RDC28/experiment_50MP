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
