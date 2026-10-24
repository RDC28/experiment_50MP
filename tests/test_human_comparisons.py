"""Review artifacts from the real synthetic reconstruction, including native sampling."""

import json
from dataclasses import astuple, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import test_images
from synthetic_raw_lab.human_comparisons import write_human_comparisons
from synthetic_raw_lab.human_config import HumanConfig
from synthetic_raw_lab.human_reconstruction import reconstruct_human_crowd


def test_synthetic_comparison_package(tmp_path):
    encoded, faces = test_images.make_crowd_test_image()
    source = cm.srgb_to_linear(encoded.astype(np.float32) / 255).astype(np.float32)
    config = HumanConfig(
        face_detector="haar", manual_face_boxes=tuple(astuple(face) for face in faces), primary_subject_index=0
    )
    result = reconstruct_human_crowd(source, 480, 320, config, log=lambda _: None)
    package = write_human_comparisons(source, result, tmp_path, tier_config=config.face_tiers)
    json.dumps(package)
    assert set(package["figures"]) == {
        "face_comparison_grid",
        "original_face_crops",
        "lanczos_face_crops",
        "conservative_face_crops",
        "balanced_face_crops",
        "final_face_crops",
        "crowd_overview",
        "small_background_faces",
        "mask_overlay",
    }
    assert len(package["face_crops"]) == len(faces) * 5
    assert "stand-in" in package["notice"].lower()
    for entry in package["figures"].values():
        with Image.open(entry["path"]) as image:
            image.load()
            assert image.width <= 1632 and image.mode == "RGB"
        assert entry["classification"] == "Packaging" and entry["description"]
    for entry in package["face_crops"]:
        face = faces[entry["face_index"]]
        image = (
            source
            if entry["method"] == "original"
            else (result.final if entry["method"] == "final" else result.candidates[entry["method"]])
        )
        sx, sy = image.shape[1] / source.shape[1], image.shape[0] / source.shape[0]
        expected_box = [
            round(face.x * sx),
            round(face.y * sy),
            round((face.x + face.width) * sx),
            round((face.y + face.height) * sy),
        ]
        assert entry["box"] == expected_box
        x0, y0, x1, y1 = expected_box
        expected = np.round(np.clip(cm.linear_to_srgb(image[y0:y1, x0:x1]), 0, 1) * 255).astype(np.uint8)
        with Image.open(entry["path"]) as saved:
            np.testing.assert_array_equal(np.asarray(saved), expected)
    assert len(list(tmp_path.glob("*.png"))) == 24


def test_empty_face_selection_is_packaged(tmp_path):
    source = np.full((16, 24, 3), 0.2, np.float32)
    result = reconstruct_human_crowd(source, 24, 16, HumanConfig(face_detector="haar"), log=lambda _: None)
    package = write_human_comparisons(source, result, tmp_path)
    assert package["face_crops"] == []
    assert len(package["figures"]) == 9
    assert all(Path(entry["path"]).is_file() for entry in package["figures"].values())


def test_mismatched_candidate_rejected_before_writing(tmp_path):
    source = np.full((16, 24, 3), 0.2, np.float32)
    result = reconstruct_human_crowd(source, 24, 16, HumanConfig(face_detector="haar"), log=lambda _: None)
    result = replace(result, candidates={**result.candidates, "balanced": source[:8]})
    with pytest.raises(ValueError, match="share final dimensions"):
        write_human_comparisons(source, result, tmp_path / "figures")
    assert not (tmp_path / "figures").exists()


def test_grid_rows_columns_and_matched_apparent_size():
    from synthetic_raw_lab.human_comparisons import _sheet

    # Different native resolutions of the same 2:1 subject fit identically.
    tiles = []
    for index in range(10):
        tile = np.zeros((10 * (index + 1), 20 * (index + 1), 3), np.uint8)
        tile[:] = (index + 1, 80, 160)
        tiles.append((f"face {index // 5}; method {index % 5}", tile))
    sheet = _sheet(tiles, "Comparison", columns=5)
    assert sheet.shape[1] == 5 * 320 + 4 * 8
    assert sheet.shape[0] < 450  # Two compact landscape rows, including labels.
    for index in range(10):
        ys, xs = np.where(np.all(sheet == (index + 1, 80, 160), axis=2))
        assert xs.max() - xs.min() + 1 == 320
        assert ys.max() - ys.min() + 1 == 160
        assert xs.min() == (index % 5) * 328
        if index >= 5:
            assert ys.min() > 200
