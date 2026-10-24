import numpy as np
import pytest

from synthetic_raw_lab import comparison
from synthetic_raw_lab.subjects import Face, SubjectAnalysis
from synthetic_raw_lab.test_images import make_test_image


@pytest.fixture(scope="module")
def encoded():
    return make_test_image(640, 360).astype(np.float32) / 255


def no_subject(width, height):
    return SubjectAnalysis(width, height, (), "none", "test")


def with_face(width, height, face):
    return SubjectAnalysis(width, height, (face,), "test", "test")


def assert_disjoint_and_in_bounds(crops, width, height):
    for i, a in enumerate(crops):
        assert 0 <= a.x and a.x + a.width <= width and 0 <= a.y and a.y + a.height <= height
        for b in crops[i + 1 :]:
            assert comparison._disjoint(a.x, a.y, a.width, b)


def test_auto_crops_start_with_the_subject_then_spread_out_details(encoded):
    crops = comparison.select_crops(encoded, 96, 5)
    assert crops[0].name in {"subject", "center"} and crops[0].label
    assert [c.name for c in crops[1:]] == ["detail_1", "detail_2", "detail_3", "detail_4"]
    assert all(c.label.startswith("detail: ") for c in crops[1:])
    assert_disjoint_and_in_bounds(crops, 640, 360)
    assert crops == comparison.select_crops(encoded, 96, 5)


def test_without_any_subject_the_centre_is_the_first_crop(encoded):
    crops = comparison.select_crops(encoded, 96, 3, subjects=no_subject(640, 360))
    assert crops[0] == comparison.Crop("center", (640 - 96) // 2, (360 - 96) // 2, 96, 96, "image centre")
    assert [c.name for c in crops] == ["center", "detail_1", "detail_2"]


def test_detail_crops_keep_apart_when_the_image_allows(encoded):
    """Detail crops leave half a crop of space between them, so they show different parts of the scene."""
    crops = comparison.select_crops(encoded, 64, 4, subjects=no_subject(640, 360))
    for i, a in enumerate(crops):
        for b in crops[i + 1 :]:
            assert comparison._disjoint(a.x, a.y, a.width, b, gap=32)


def test_a_face_smaller_than_a_crop_is_shown_whole(encoded):
    crops = comparison.select_crops(encoded, 96, 4, subjects=with_face(640, 360, Face(400, 100, 60, 70)))
    face = crops[0]
    assert (face.name, face.label) == ("face", "primary face")
    assert face.x <= 400 and face.x + 96 >= 460 and face.y <= 100 and face.y + 96 >= 170
    assert [c.name for c in crops[1:]] == ["detail_1", "detail_2", "detail_3"]
    assert_disjoint_and_in_bounds(crops, 640, 360)


def test_a_large_face_gets_eyes_and_mouth_crops_and_keeps_a_detail_crop(encoded):
    """Like the first Colab portrait: the face is much larger than a crop and texture dominates elsewhere."""
    face = Face(250, 20, 150, 300)
    crops = comparison.select_crops(encoded, 96, 4, subjects=with_face(640, 360, face))
    assert [c.name for c in crops] == ["face_eyes", "face_nose_mouth", "detail_1", "detail_2"]
    eyes, mouth = crops[0], crops[1]
    assert eyes.label == "primary face: eyes" and mouth.label == "primary face: nose and mouth"
    assert eyes.x + 48 == 325 and abs(eyes.y + 48 - (20 + 0.4 * 300)) <= 1
    assert (mouth.x, mouth.y) == (eyes.x, eyes.y + 96)
    assert_disjoint_and_in_bounds(crops, 640, 360)
    two = comparison.select_crops(encoded, 96, 2, subjects=with_face(640, 360, face))
    assert [c.name for c in two] == ["face_eyes", "detail_1"]
    one = comparison.select_crops(encoded, 96, 1, subjects=with_face(640, 360, face))
    assert [c.name for c in one] == ["face_eyes"]


def test_face_crops_stay_inside_the_image(encoded):
    crops = comparison.select_crops(encoded, 96, 3, subjects=with_face(640, 360, Face(600, 300, 40, 60)))
    assert crops[0].name == "face"
    assert_disjoint_and_in_bounds(crops, 640, 360)


def test_the_centre_of_interest_frames_the_subject_crop(encoded):
    subject = SubjectAnalysis(640, 360, (), "none", "test", (100, 80), "test")
    crops = comparison.select_crops(encoded, 96, 2, subjects=subject)
    assert crops[0] == comparison.Crop("subject", 52, 32, 96, 96, "centre of interest (saliency)")


def test_without_subject_first_the_centre_leads_plain_non_overlapping_details(encoded, monkeypatch):
    monkeypatch.setattr(comparison, "analyse", lambda _: pytest.fail("subject analysis must not run"))
    crops = comparison.select_crops(encoded, 96, 5, subject_first=False)
    assert [c.name for c in crops] == ["center", "detail_1", "detail_2", "detail_3", "detail_4"]
    assert crops[0].x == (640 - 96) // 2 and crops[0].y == (360 - 96) // 2
    assert_disjoint_and_in_bounds(crops, 640, 360)
    spaced = comparison.select_crops(encoded, 96, 5, subjects=no_subject(640, 360))
    assert crops[1:] != spaced[1:]


def test_an_unlabelled_crop_records_no_label():
    assert comparison.Crop("c", 1, 2, 3, 4).to_dict() == {"name": "c", "x": 1, "y": 2, "width": 3, "height": 4}


def test_crop_size_is_limited_by_the_image(encoded):
    crops = comparison.select_crops(encoded[:50, :80], 256, 1)
    assert (crops[0].width, crops[0].height) == (50, 50)


def test_manual_crops_override(encoded):
    crops = comparison.select_crops(encoded, 96, 4, manual=[(10, 20, 30, 40)])
    assert [c.to_dict() for c in crops] == [
        {"name": "manual_1", "label": "manual crop from the configuration", "x": 10, "y": 20, "width": 30, "height": 40}
    ]
    with pytest.raises(ValueError):
        comparison.select_crops(encoded, 96, 4, manual=[(630, 0, 30, 30)])


def test_output_box_scales_per_axis():
    crop = comparison.Crop("c", 100, 50, 20, 10)
    assert crop.output_box((400, 200), (1000, 600)) == (250, 150, 300, 180)


def test_crop_panel_layout():
    tile = np.zeros((40, 60, 3), dtype=np.uint8)
    panel = comparison.crop_panel(np.zeros((16, 24, 3), np.uint8), {"lanczos": tile, "bicubic": tile + 10})
    assert panel.shape == (40 + 30, 4 * 60 + 3 * 8, 3)
    assert panel.dtype == np.uint8
