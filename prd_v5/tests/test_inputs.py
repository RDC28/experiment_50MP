"""Tiny source-file fixtures; no large images or inference."""

import hashlib
import json

import pytest
from PIL import Image, ImageCms

from prd_v5.inputs import InputInspector


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_source_identity_headers_and_explicit_assumption(tmp_path, image_format):
    path = tmp_path / f"source.{image_format.lower()}"
    Image.new("RGB", (12, 8), (40, 80, 120)).save(path, format=image_format)
    before = path.read_bytes()
    record = InputInspector().inspect(path)
    assert (record.width, record.height, record.encoded_bits_per_channel) == (12, 8, 8)
    assert record.image_format == image_format
    assert record.sha256 == hashlib.sha256(before).hexdigest()
    assert record.size_bytes == len(before)
    assert record.icc_sha256 is None
    assert record.colour_policy == "requires_colour_assumption"
    assert InputInspector().inspect(path, assume_unprofiled_srgb=True).colour_policy == "assumed_srgb"
    assert path.read_bytes() == before
    assert json.loads(json.dumps(record.to_dict(), allow_nan=False))["sha256"] == record.sha256
    assert "Header inspected only" in record.validation


@pytest.mark.parametrize("valid", [True, False])
def test_icc_is_hashed_never_silently_assumed_srgb(tmp_path, valid):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes() if valid else b"broken ICC"
    path = tmp_path / "profile.png"
    Image.new("RGB", (4, 4)).save(path, icc_profile=profile)
    result = InputInspector().inspect(path, assume_unprofiled_srgb=True)
    assert result.icc_sha256 == hashlib.sha256(profile).hexdigest()
    assert result.colour_policy == ("requires_icc_transform" if valid else "blocked_invalid_icc")
    assert bool(result.icc_description) == valid


@pytest.mark.parametrize("mode", ["RGBA", "L", "P", "I;16"])
def test_unsupported_pixel_interpretations_are_rejected(tmp_path, mode):
    path = tmp_path / "source.png"
    Image.new(mode, (4, 4)).save(path)
    with pytest.raises(ValueError, match="RGB without alpha"):
        InputInspector().inspect(path)


def test_no_implicit_pixel_decode_or_private_exif_copy(tmp_path, monkeypatch):
    path = tmp_path / "private.jpg"
    exif = Image.Exif()
    exif[42033] = "private body serial"
    Image.new("RGB", (4, 4)).save(path, exif=exif)

    def forbidden_load(*args, **kwargs):
        pytest.fail("Header inspection must not allocate decoded pixels")

    monkeypatch.setattr(Image.Image, "load", forbidden_load)
    record = InputInspector().inspect(path)
    assert record.exif_present
    assert "private body serial" not in json.dumps(record.to_dict())


def test_invalid_file_and_policy(tmp_path):
    path = tmp_path / "bad.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Cannot inspect"):
        InputInspector().inspect(path)
    with pytest.raises(ValueError, match="boolean"):
        InputInspector().inspect(path, assume_unprofiled_srgb="yes")
