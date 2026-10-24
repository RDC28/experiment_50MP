"""Experiment E10: licence checks, verified downloads, degradation models and a tiny end-to-end run.

The Commons API and the download are stubbed; tests never touch the network.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from helpers import evaluation_config, fake_sr
from synthetic_raw_lab import baselines, reference_test, test_images
from synthetic_raw_lab.evaluation_config import ReferenceParams

ROOT = Path(__file__).resolve().parents[1]


def photo(tmp_path: Path, width=360, height=240) -> bytes:
    test_images.write_test_image(tmp_path / "photo.png", width, height)
    path = tmp_path / "photo.jpg"
    cv2.imwrite(str(path), cv2.imread(str(tmp_path / "photo.png")), [cv2.IMWRITE_JPEG_QUALITY, 97])
    return path.read_bytes()


def commons(data: bytes, licence="CC BY-SA 4.0", sha1=None, calls=None):
    digest = sha1 or hashlib.sha1(data).hexdigest()

    def opener(url: str) -> bytes:
        if calls is not None:
            calls.append(url)
        if url.startswith(reference_test.COMMONS_API):
            info = {
                "url": "https://upload.wikimedia.org/x/Test.jpg?utm_source=commons",
                "descriptionurl": "https://commons.wikimedia.org/wiki/File:Test.jpg",
                "width": 360,
                "height": 240,
                "size": len(data),
                "sha1": digest,
                "mime": "image/jpeg",
                "extmetadata": {
                    "LicenseShortName": {"value": licence},
                    "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0"},
                    "Artist": {"value": '<a href="https://example.org">Jane &amp; Co</a>'},
                    "AttributionRequired": {"value": "true"},
                },
            }
            return json.dumps({"query": {"pages": {"7": {"title": "File:Test.jpg", "imageinfo": [info]}}}}).encode()
        return data

    return opener


def reference(data: bytes, scene="general") -> reference_test.Reference:
    return reference_test.Reference("test", "File:Test.jpg", hashlib.sha1(data).hexdigest(), scene, "test photo")


def test_pinned_references_are_free_camera_photographs():
    for ref in reference_test.REFERENCES.values():
        assert ref.title.startswith("File:") and len(ref.sha1) == 40 and ref.scene in ("human", "night")


def test_licence_record_reads_commons_metadata(tmp_path):
    data = photo(tmp_path)
    record = reference_test.licence_record(reference(data), commons(data))
    assert record["licence"] == "CC BY-SA 4.0" and record["author"] == "Jane & Co"
    assert record["file_url"] == "https://upload.wikimedia.org/x/Test.jpg" and record["attribution_required"]


@pytest.mark.parametrize("licence", ["CC BY-NC 4.0", "All rights reserved", "CC BY-ND 4.0"])
def test_non_free_licences_are_refused(tmp_path, licence):
    data = photo(tmp_path)
    with pytest.raises(ValueError, match="not one of the free licences"):
        reference_test.licence_record(reference(data), commons(data, licence=licence))


def test_a_changed_file_on_commons_is_refused(tmp_path):
    data = photo(tmp_path)
    with pytest.raises(ValueError, match="changed on Commons"):
        reference_test.licence_record(reference(data), commons(data, sha1="0" * 40))


def test_fetch_verifies_and_caches_the_download(tmp_path):
    data = photo(tmp_path)
    calls: list[str] = []
    path, record = reference_test.fetch(reference(data), tmp_path / "cache", commons(data, calls=calls))
    assert path.read_bytes() == data and record["cached_file"] == str(path)
    reference_test.fetch(reference(data), tmp_path / "cache", commons(data, calls=calls))
    assert sum(not c.startswith(reference_test.COMMONS_API) for c in calls) == 1  # second fetch reuses the file
    tampered = commons(data)

    def corrupt(url):
        return tampered(url) if url.startswith(reference_test.COMMONS_API) else data[:-10]

    with pytest.raises(ValueError, match="has SHA-1"):
        reference_test.fetch(reference(data), tmp_path / "other", corrupt)


@pytest.mark.parametrize("model", ["area", "camera", "ai_render"])
def test_degradations_make_8_bit_sources_of_the_requested_size(model):
    rng = np.random.default_rng(0)
    truth = rng.random((120, 180, 3)).astype(np.float32)
    codes, record = reference_test.degrade(truth, (90, 60), model, ReferenceParams())
    assert codes.shape == (60, 90, 3) and codes.dtype == np.uint8
    assert record["model"] == model and record["steps"]


def test_area_degradation_is_the_area_average_rounded_to_8_bits():
    truth = np.random.default_rng(1).random((64, 96, 3)).astype(np.float32)
    codes, _ = reference_test.degrade(truth, (48, 32), "area", ReferenceParams())
    expected = np.round(reference_test.cm.linear_to_srgb(np.clip(baselines.downsample_area(truth, 48, 32), 0, 1)) * 255)
    assert np.array_equal(codes, expected.astype(np.uint8))


def test_centre_crop_keeps_the_aspect_and_the_middle():
    image = np.zeros((400, 600, 3), np.float32)
    crop, box = reference_test.centre_crop(image, 0.06)
    assert crop.shape[:2] == (200, 300) and box == [150, 100, 450, 300]
    assert reference_test.centre_crop(image, 0)[1] == [0, 0, 600, 400]


def test_run_reference_test_end_to_end(tmp_path):
    data = photo(tmp_path)
    config = evaluation_config(
        tmp_path,
        ROOT,
        reference={
            "references": [
                {"name": "test", "title": "File:Test.jpg", "sha1": hashlib.sha1(data).hexdigest(), "scene": "general"}
            ],
            "crop_megapixels": 0,
            "scales": [2.0],
            "degradations": ["area", "ai_render"],
            "configurations": ["bicubic", "lanczos", "sr_conservative", "full", "full_no_sensor"],
        },
    )

    def factory(scene, record, figures_dir, stage):
        return fake_sr

    result = reference_test.run_reference_test(
        config, tmp_path, opener=commons(data), sr_runner_factory=factory, log=lambda _: None
    )
    manifest = result.manifest
    assert manifest["status"] == "complete" and manifest["schema"] == reference_test.SCHEMA
    assert manifest["references"][0]["licence"] == "CC BY-SA 4.0"
    assert [c["key"] for c in manifest["cases"]] == ["test_area_x2", "test_ai_render_x2"]
    case = manifest["cases"][0]
    assert case["source_size"] == [180, 120] and case["output_size"] == [360, 240]
    assert Path(case["degraded_source"]).exists() and Path(case["figures"]["review_sheet"]).exists()
    rows = reference_test.summary_rows(manifest)
    assert len(rows) == 10
    lanczos = next(r for r in rows if r["case"] == "test_area_x2" and r["configuration"] == "lanczos")
    assert lanczos["psnr_gain_db"] == 0 and lanczos["hf_nmse"] is not None
    report = result.report_path.read_text()
    assert "CC BY-SA 4.0" in report and "Jane & Co" in report and "## Summary" in report
    assert "—" not in report


def test_reference_test_without_sr_skips_the_learned_rows(tmp_path):
    data = photo(tmp_path, 240, 160)
    config = evaluation_config(
        tmp_path,
        ROOT,
        universal_overrides={"super_resolution": {"enabled": False}},
        reference={
            "references": [
                {"name": "test", "title": "File:Test.jpg", "sha1": hashlib.sha1(data).hexdigest(), "scene": "night"}
            ],
            "crop_megapixels": 0,
            "scales": [2.0],
            "degradations": ["camera"],
            "configurations": ["lanczos", "sr_conservative"],
        },
    )
    result = reference_test.run_reference_test(config, tmp_path, opener=commons(data), log=lambda _: None)
    records = result.manifest["cases"][0]["configurations"]
    assert "disabled" in records[1]["skipped"] and "reference" in records[0]
