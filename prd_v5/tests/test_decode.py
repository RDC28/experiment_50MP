"""Tiny end-user JPEG/PNG decode fixtures."""

import json

import numpy as np
import pytest
from PIL import Image, ImageCms

from prd_v5.decode import decode_source


def output(tmp_path):
    return np.memmap(tmp_path / "pixels.bin", mode="w+", dtype=np.float32, shape=(2, 3, 3))


@pytest.mark.parametrize("extension", ["png", "jpg"])
def test_decode_file_to_disk_srgb(tmp_path, extension):
    path = tmp_path / f"source.{extension}"
    Image.new("RGB", (3, 2), (0, 128, 255)).save(path)
    dest = output(tmp_path)
    report = decode_source(path, dest, assume_unprofiled_srgb=True, strip_rows=1)
    with Image.open(path) as expected:
        np.testing.assert_allclose(dest, np.asarray(expected).astype(np.float32) / 255)
    assert report["effective_colour_policy"] == "assumed_srgb"
    if extension == "png":
        assert report["srgb_pixels_at_zero"] == report["srgb_pixels_at_one"] == 6
    json.dumps(report, allow_nan=False)


def test_icc_transform_and_strip_invariance(tmp_path):
    path = tmp_path / "icc.png"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGB", (3, 2), (40, 80, 120)).save(path, icc_profile=profile)
    dest = output(tmp_path)
    report = decode_source(path, dest, strip_rows=1)
    first = dest.copy()
    decode_source(path, dest, strip_rows=2)
    np.testing.assert_array_equal(first, dest)
    np.testing.assert_allclose(dest[0, 0], np.array([40, 80, 120]) / 255, atol=1e-7)
    assert report["effective_colour_policy"] == "icc_to_srgb"


def test_policy_and_budget_fail_before_loading(tmp_path, monkeypatch):
    path = tmp_path / "source.png"
    Image.new("RGB", (3, 2)).save(path)
    dest = output(tmp_path)
    dest[:] = -1

    def forbidden(*args, **kwargs):
        pytest.fail("must reject before decoding")

    monkeypatch.setattr(Image.Image, "load", forbidden)
    with pytest.raises(ValueError, match="colour policy"):
        decode_source(path, dest)
    with pytest.raises(ValueError, match="memory budget"):
        decode_source(path, dest, assume_unprofiled_srgb=True, max_decode_bytes=47)
    assert np.all(dest == -1)


@pytest.mark.parametrize("kwargs", [{"strip_rows": True}, {"max_decode_bytes": 0}])
def test_invalid_resource_parameters(tmp_path, kwargs):
    with pytest.raises(ValueError, match="positive integer"):
        decode_source(tmp_path / "absent.png", output(tmp_path), **kwargs)


def test_invalid_destination_and_profile(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (3, 2)).save(path)
    with pytest.raises(ValueError, match="memmap"):
        decode_source(path, np.zeros((2, 3, 3)), assume_unprofiled_srgb=True)
    Image.new("RGB", (3, 2)).save(path, icc_profile=b"invalid")
    with pytest.raises(ValueError, match="blocked_invalid_icc"):
        decode_source(path, output(tmp_path))
