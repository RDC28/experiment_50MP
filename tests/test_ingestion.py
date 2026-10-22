import os
import stat

import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image

from helpers import build_icc_profile, d50_colorants
from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import exporters
from synthetic_raw_lab.ingestion import IngestError, ingest, sha256_file


def gradient_rgb(height=48, width=64) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    image = np.stack([x * 255 // (width - 1), y * 255 // (height - 1), (x + y) * 255 // (width + height - 2)], -1)
    image[:8, :8] = (255, 0, 0)  # pure red block to catch channel swaps
    return image.astype(np.uint8)


def test_png_8bit_untagged(tmp_path, project):
    source = tmp_path / "scene.png"
    Image.fromarray(gradient_rgb()).save(source)
    before = (sha256_file(source), source.stat().st_mtime_ns)

    image = ingest(source, project / "inputs", "human")

    assert (sha256_file(source), source.stat().st_mtime_ns) == before  # original untouched
    assert image.stored_path.parent == project / "inputs" / "human"
    assert not os.access(image.stored_path, os.W_OK) or os.geteuid() == 0
    assert not image.stored_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    assert image.sha256 == before[0]
    v = image.validation
    assert v["file"]["format"] == "PNG" and v["file"]["stored_read_only"]
    assert (v["dimensions"]["width"], v["dimensions"]["height"]) == (64, 48)
    assert v["dimensions"]["aspect_ratio_reduced"] == "4:3"
    assert v["samples"]["bit_depth"] == 8 and v["samples"]["color_channels"] == 3
    assert v["alpha"]["present"] is False
    assert v["color"]["resolved"]["origin"] == "untagged_default"
    assert "not scene-referred" in v["color"]["working_signal"]
    assert len(v["histograms"]["R"]) == 256 and sum(v["histograms"]["luma"]) == 64 * 48
    assert any("assumed sRGB" in w for w in image.warnings)
    # No channel swap: the red block decodes as red in the working space.
    assert image.codes[0, 0].tolist() == [255, 0, 0]
    assert image.linear[0, 0, 0] == pytest.approx(1.0) and image.linear[0, 0, 2] == 0
    assert image.linear.dtype == np.float32 and image.linear.shape == (48, 64, 3)


def test_png_16bit_keeps_full_precision(tmp_path, project):
    rgb = (np.random.default_rng(0).random((20, 30, 3)) * 65535).astype(np.uint16)
    source = tmp_path / "deep.png"
    cv2.imwrite(str(source), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    image = ingest(source, project / "inputs", "night")
    assert image.codes.dtype == np.uint16
    assert np.array_equal(image.codes, rgb)
    assert image.validation["samples"]["effective_bit_depth"] == 16


def test_8bit_data_in_16bit_container_is_flagged(tmp_path, project):
    rgb = gradient_rgb().astype(np.uint16) * 257
    source = tmp_path / "padded.png"
    cv2.imwrite(str(source), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    image = ingest(source, project / "inputs", "night")
    assert image.validation["samples"]["effective_bit_depth"] == 8
    assert any("16-bit container holds 8-bit data" in w for w in image.warnings)


def test_jpeg_exif_quality_and_orientation(tmp_path, project):
    source = tmp_path / "photo.jpg"
    exif = Image.Exif()
    exif[305] = "GeneratorX"
    exif[274] = 6
    Image.fromarray(gradient_rgb(96, 128)).save(source, quality=90, exif=exif)
    image = ingest(source, project / "inputs", "human")
    v = image.validation
    assert v["metadata"]["exif"]["Software"] == "GeneratorX"
    assert v["compression"]["jpeg_quality_estimate"] == pytest.approx(90, abs=3)
    assert v["compression"]["blockiness_ratio"] > 0
    assert any("orientation 6" in w for w in image.warnings)


def test_tiff_16bit_with_icc(tmp_path, project):
    rgb = (np.random.default_rng(1).random((24, 32, 3)) * 65535).astype(np.uint16)
    source = tmp_path / "master.tif"
    tifffile.imwrite(source, rgb, photometric="rgb", iccprofile=cm.srgb_icc_profile())
    image = ingest(source, project / "inputs", "night")
    assert np.array_equal(image.codes, rgb)
    color = image.validation["color"]
    assert color["icc_profile_present"]
    assert color["resolved"]["name"] == "srgb" and color["resolved"]["origin"] == "icc_profile"
    assert not any("assumed sRGB" in w for w in image.warnings)


def test_display_p3_png_is_converted(tmp_path, project):
    params = (2.4, 1 / 1.055, 0.055 / 1.055, 1 / 12.92, 0.04045)
    icc = build_icc_profile(d50_colorants(cm.DISPLAY_P3_PRIMARIES), ("para", 3, params), description="Display P3")
    source = tmp_path / "p3.png"
    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[...] = (255, 0, 0)
    Image.fromarray(rgb).save(source, icc_profile=icc)
    image = ingest(source, project / "inputs", "night")
    assert image.color_space.name == "icc:Display P3"
    assert image.linear[0, 0, 0] > 1.0  # P3 red lies outside the sRGB gamut


def test_rgba_alpha_is_reported_and_discarded(tmp_path, project):
    rgba = np.dstack([gradient_rgb(), np.full((48, 64), 255, dtype=np.uint8)])
    rgba[10:20, 10:20, 3] = 0
    source = tmp_path / "cutout.png"
    Image.fromarray(rgba).save(source)
    image = ingest(source, project / "inputs", "human")
    assert image.codes.shape[2] == 3
    assert image.validation["alpha"] == {"present": True, "opaque": False, "min": 0.0}
    assert any("alpha channel is not opaque" in w for w in image.warnings)


def test_greyscale_is_replicated(tmp_path, project):
    source = tmp_path / "grey.png"
    Image.fromarray(gradient_rgb()[..., 1]).save(source)
    image = ingest(source, project / "inputs", "night")
    assert image.codes.shape[2] == 1
    assert image.linear.shape[2] == 3
    assert np.array_equal(image.linear[..., 0], image.linear[..., 2])
    assert any("greyscale" in w for w in image.warnings)


def test_clipping_is_reported(tmp_path, project):
    rgb = gradient_rgb()
    rgb[:, :16] = 255
    source = tmp_path / "bright.png"
    Image.fromarray(rgb).save(source)
    image = ingest(source, project / "inputs", "night")
    assert image.validation["clipping"]["channels"]["G"]["at_max_fraction"] >= 0.25
    assert any("highlight clipping in G" in w for w in image.warnings)


def test_exr_input_uses_header_chromaticities(tmp_path, project):
    linear = np.random.default_rng(2).random((12, 10, 3)).astype(np.float32) * 2
    source = tmp_path / "plate.exr"
    exporters.write_exr(source, linear, {}, precision="float")
    image = ingest(source, project / "inputs", "night")
    assert image.color_space.name == "linear_srgb"
    assert np.allclose(image.linear, linear)


def test_duplicates_are_detected(tmp_path, project):
    rgb = gradient_rgb()
    first = tmp_path / "a.png"
    Image.fromarray(rgb).save(first)
    ingest(first, project / "inputs", "human")
    assert ingest(first, project / "inputs", "human").validation["duplicates"] == []  # re-ingest is not a duplicate

    renamed = tmp_path / "b.png"
    renamed.write_bytes(first.read_bytes())
    assert ingest(renamed, project / "inputs", "night").validation["duplicates"][0]["match"] == "identical file"

    as_tiff = tmp_path / "c.tif"
    tifffile.imwrite(as_tiff, rgb, photometric="rgb")
    duplicate = ingest(as_tiff, project / "inputs", "night")
    assert {d["match"] for d in duplicate.validation["duplicates"]} == {"identical pixels"}
    assert any("duplicate" in w for w in duplicate.warnings)


@pytest.mark.parametrize("case", ["truncated", "garbage", "empty", "unsupported", "cmyk", "missing"])
def test_unusable_sources_are_rejected(tmp_path, project, case):
    source = tmp_path / "bad.png"
    if case == "truncated":
        Image.fromarray(gradient_rgb(200, 200)).save(source)
        source.write_bytes(source.read_bytes()[:-200])
    elif case == "garbage":
        source.write_bytes(os.urandom(512))
    elif case == "empty":
        source.write_bytes(b"")
    elif case == "unsupported":
        source = tmp_path / "bad.bmp"
        Image.fromarray(gradient_rgb()).save(source)
    elif case == "cmyk":
        source = tmp_path / "bad.jpg"
        Image.fromarray(gradient_rgb()).convert("CMYK").save(source)
    with pytest.raises(IngestError):
        ingest(source, project / "inputs", "human")


def test_stored_copy_conflict_is_an_error(tmp_path, project):
    source = tmp_path / "scene.png"
    Image.fromarray(gradient_rgb()).save(source)
    image = ingest(source, project / "inputs", "human")
    image.stored_path.chmod(0o644)
    image.stored_path.write_bytes(b"tampered")
    with pytest.raises(IngestError, match="different content"):
        ingest(source, project / "inputs", "human")
