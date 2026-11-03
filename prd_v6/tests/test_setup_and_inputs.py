"""Setup (step 2), preflight (step 3) and production source handling, without network or GPU."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from prd_v6 import install, preflight, production


def test_install_detects_a_version_change_of_a_loaded_package():
    assert install.changed({"numpy": "2.0.0", "rawpy": "0.27.1"}, {"numpy": "2.1.0", "rawpy": "0.27.1"}) == {
        "numpy": ["2.0.0", "2.1.0"]
    }
    assert install.changed({"numpy": "2.0.0"}, {"numpy": "2.0.0"}) == {}


def test_loaded_distributions_include_numpy():
    assert "numpy" in install.loaded_distributions()


@pytest.mark.parametrize(
    ("gpu", "ram", "disk", "problems", "warnings"),
    [
        ("NVIDIA A100-SXM4-40GB", 89, 200, 0, 0),
        ("NVIDIA H100 80GB HBM3", 89, 200, 0, 0),
        ("NVIDIA L4", 53, 200, 0, 1),
        ("Tesla T4", 13, 100, 2, 0),
        (None, 13, 100, 2, 0),
        ("NVIDIA A100-SXM4-40GB", 40, 50, 2, 0),
    ],
)
def test_runtime_tier_rules(gpu, ram, disk, problems, warnings):
    found = preflight.classify(gpu, ram, disk)
    assert (len(found[0]), len(found[1])) == (problems, warnings)


def test_download_verifies_size_and_digest(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"exp50" * 1000)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    url = source.as_uri()
    record = preflight.download(url, tmp_path / "in" / "a.bin", size=5000, digest=digest, log=lambda m: None)
    assert record["sha256"] == digest and not record["cached"]
    assert preflight.download(url, tmp_path / "in" / "a.bin", size=5000, digest=digest, log=lambda m: None)["cached"]
    with pytest.raises(preflight.PreflightError, match="SHA-256"):
        preflight.download(url, tmp_path / "in" / "b.bin", size=5000, digest="0" * 64, log=lambda m: None)
    assert not (tmp_path / "in" / "b.bin").exists()


@pytest.mark.parametrize(
    ("size", "portrait", "crop_hw"), [((400, 200), False, (200, 300)), ((300, 600), True, (450, 300))]
)
def test_production_source_is_centre_cropped_to_3_2(tmp_path, size, portrait, crop_hw):
    from PIL import Image

    path = tmp_path / "source.png"
    Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8)).save(path)
    rgb, info = production.load_source(path)
    assert info["portrait"] is portrait
    assert rgb.shape[:2] == crop_hw


def test_production_source_honours_exif_orientation(tmp_path):
    from PIL import Image

    image = Image.fromarray(np.zeros((300, 450, 3), dtype=np.uint8))
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90 degrees on display
    path = tmp_path / "rotated.jpg"
    image.save(path, exif=exif)
    rgb, info = production.load_source(path)
    assert info["portrait"] and rgb.shape[:2] == (450, 300)


@pytest.mark.parametrize(
    "url",
    [
        "https://drive.google.com/file/d/1AbC_d-9/view?usp=drive_link",
        "https://drive.google.com/open?id=1AbC_d-9",
        "https://drive.google.com/uc?export=download&id=1AbC_d-9",
        "https://docs.google.com/uc?id=1AbC_d-9",
    ],
)
def test_drive_share_links_resolve_to_the_file_id(url):
    assert preflight.drive_file_id(url) == "1AbC_d-9"


def test_non_drive_urls_are_left_alone(tmp_path):
    assert preflight.drive_file_id("https://example.com/file/d/abc/view") is None
    assert preflight.own_image_dest(tmp_path, "4k_source", "https://example.com/a/pic.jpg?x=1").name == "4k_source.jpg"
    assert (
        preflight.own_image_dest(tmp_path, "4k_source", "https://drive.google.com/file/d/abc/view").name
        == "4k_source.img"
    )


def test_drive_large_file_confirmation_form_is_followed():
    page = (
        '<form id="download-form" action="https://drive.usercontent.google.com/download" method="get">'
        '<input type="hidden" name="id" value="abc"><input type="hidden" name="confirm" value="t">'
        '<input type="hidden" name="uuid" value="u&amp;1"></form>'
    )
    target = preflight._confirm_url(page, "https://drive.google.com/uc")
    assert target == "https://drive.usercontent.google.com/download?id=abc&confirm=t&uuid=u%261"
    assert preflight._confirm_url("<html>sign in</html>", "https://drive.google.com/uc") is None


def test_undecodable_source_fails_early_naming_the_setting(tmp_path):
    page = tmp_path / "4k_source.img"
    page.write_bytes(b"<!DOCTYPE html><html>Google Drive</html>")
    with pytest.raises(preflight.PreflightError, match="SOURCE_4K_URL.*not a decodable image"):
        preflight.check_image(page, "SOURCE_4K_URL")
    assert not page.exists()
    from PIL import Image

    Image.new("RGB", (6, 4)).save(tmp_path / "ok.png")
    preflight.check_image(tmp_path / "ok.png", "SOURCE_2K_URL")
