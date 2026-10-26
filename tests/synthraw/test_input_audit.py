"""Tiny file-based source audits, without model or full-resolution allocations."""

import hashlib
from dataclasses import asdict

import numpy as np
import pytest
from PIL import Image, ImageCms

from synthraw.input_audit import InputInspector


def test_rgb_observations_and_clipping(tmp_path):
    path = tmp_path / "source.png"
    pixels = np.array([[[0, 254, 100], [1, 255, 100]], [[2, 253, 100], [255, 0, 100]]], dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    audit = InputInspector().inspect(path)
    assert (audit.width, audit.height, audit.mode, audit.bit_depth) == (2, 2, "RGB", 8)
    assert audit.aspect_ratio == 1
    assert audit.near_black_fractions == (0.5, 0.25, 0)
    assert audit.near_white_fractions == (0.25, 0.5, 0)
    assert audit.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit.resolution_class == "OTHER"
    assert not audit.icc_present and not audit.exif_present
    assert audit.to_dict() == asdict(audit)


def test_icc_and_exif(tmp_path):
    path = tmp_path / "source.jpg"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    exif = Image.Exif()
    exif[271] = "Source maker"
    Image.new("RGB", (4, 3), (128, 128, 128)).save(path, icc_profile=profile, exif=exif)
    audit = InputInspector().inspect(path)
    assert audit.image_format == "JPEG"
    assert audit.icc_present and "sRGB" in audit.icc_description
    assert audit.exif_present
    assert audit.aspect_ratio == 4 / 3


def test_invalid_icc_is_still_reported(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (2, 3)).save(path, icc_profile=b"broken profile")
    audit = InputInspector().inspect(path)
    assert audit.icc_present and audit.icc_description is None


@pytest.mark.parametrize(("mode", "extension"), [("RGBA", "png"), ("P", "png"), ("CMYK", "jpg")])
def test_rgb_capable_modes(tmp_path, mode, extension):
    path = tmp_path / f"source.{extension}"
    Image.new(mode, (3, 2)).save(path)
    assert InputInspector().inspect(path).mode == mode


@pytest.mark.parametrize("mode", ["L", "LA", "I;16", "1"])
def test_reject_grayscale(tmp_path, mode):
    path = tmp_path / "gray.png"
    Image.new(mode, (2, 2)).save(path)
    with pytest.raises(ValueError, match="RGB-capable"):
        InputInspector().inspect(path)


def test_reject_other_format_and_corruption(tmp_path):
    path = tmp_path / "source.bmp"
    Image.new("RGB", (2, 2)).save(path)
    with pytest.raises(ValueError, match="JPEG or PNG"):
        InputInspector().inspect(path)
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot decode"):
        InputInspector().inspect(path)


@pytest.mark.parametrize(
    ("dimensions", "expected"), [((2047, 10), "OTHER"), ((10, 2048), "2K"), ((4095, 10), "2K"), ((4096, 1736), "4K+")]
)
def test_resolution_labels_without_large_arrays(dimensions, expected):
    assert InputInspector.classify_resolution(*dimensions) == expected


def test_invalid_dimensions():
    with pytest.raises(ValueError, match="positive"):
        InputInspector.classify_resolution(0, 2)


def test_reject_16bit_rgb_png(tmp_path):
    import struct
    import zlib

    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    path = tmp_path / "rgb16.png"
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 16, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\xff\x00\x00\x80\x00"))
        + chunk(b"IEND", b"")
    )
    with pytest.raises(ValueError, match="8-bit RGB PNG"):
        InputInspector().inspect(path)
