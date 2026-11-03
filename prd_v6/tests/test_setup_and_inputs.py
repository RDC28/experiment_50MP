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
