import numpy as np
import pytest

from synthetic_raw_lab import comparison
from synthetic_raw_lab.test_images import make_test_image


@pytest.fixture(scope="module")
def encoded():
    return make_test_image(640, 360).astype(np.float32) / 255


def test_auto_crops_are_centre_first_disjoint_and_in_bounds(encoded):
    crops = comparison.select_crops(encoded, 96, 5)
    assert [c.name for c in crops] == ["center", "detail_1", "detail_2", "detail_3", "detail_4"]
    assert crops[0].x == (640 - 96) // 2 and crops[0].y == (360 - 96) // 2
    for i, a in enumerate(crops):
        assert 0 <= a.x and a.x + a.width <= 640 and 0 <= a.y and a.y + a.height <= 360
        for b in crops[i + 1 :]:
            assert comparison._disjoint(a.x, a.y, a.width, b)
    assert crops == comparison.select_crops(encoded, 96, 5)


def test_crop_size_is_limited_by_the_image(encoded):
    crops = comparison.select_crops(encoded[:50, :80], 256, 1)
    assert (crops[0].width, crops[0].height) == (50, 50)


def test_manual_crops_override(encoded):
    crops = comparison.select_crops(encoded, 96, 4, manual=[(10, 20, 30, 40)])
    assert [c.to_dict() for c in crops] == [{"name": "manual_1", "x": 10, "y": 20, "width": 30, "height": 40}]
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
