"""Tiny disk-backed sensor-fit placement checks."""

import numpy as np
import pytest

from synthraw.geometry import GeometryPlan
from synthraw.sensor_canvas import place_sensor_canvas


def plan(mode, *, portrait=False):
    box = (2.0, 0.0, 5.0, 8.0) if portrait else (0.0, 1.5, 8.0, 5.5)
    if mode == "crop":
        box = (0.0, 0.0, 8.0, 8.0)
    return GeometryPlan(
        8,
        8,
        1,
        (0, 0, 8, 8),
        box,
        (box,) if mode == "outpaint" else (),
        (box,) if mode == "margins" else (),
        (0, 0),
        (8, 8),
        (0, 0, 8, 8),
        {},
    )


@pytest.mark.parametrize("mode", ["crop", "outpaint", "margins"])
@pytest.mark.parametrize("portrait", [False, True])
def test_canvas_preserves_content_and_labels_only_invented_bands(tmp_path, mode, portrait):
    geometry = plan(mode, portrait=portrait)
    left, top, right, bottom = map(round, geometry.content_box)
    content = np.linspace(0.1, 0.9, (bottom - top) * (right - left) * 3, dtype=np.float32).reshape(
        bottom - top, right - left, 3
    )
    result = place_sensor_canvas(content, geometry, tmp_path)
    canvas = np.load(result.path)
    supported = np.load(result.source_supported_mask)
    generated = np.load(result.generated_detail_mask)
    np.testing.assert_array_equal(canvas[top:bottom, left:right], content)
    expected_support = np.zeros((8, 8), dtype=np.float32)
    expected_support[top:bottom, left:right] = 1
    np.testing.assert_array_equal(supported, expected_support)
    np.testing.assert_array_equal(generated, 1 - supported if mode == "outpaint" else np.zeros((8, 8)))
    if mode == "outpaint":
        expected = np.pad(content, ((top, 8 - bottom), (left, 8 - right), (0, 0)), mode="edge")
        np.testing.assert_array_equal(canvas, expected)
    elif mode == "margins":
        assert np.all(canvas[supported == 0] == 0)
    assert result.provenance.to_dict()["provenance"] == "F"
    assert result.provenance.confidence is None


def test_canvas_rejects_bad_shape_before_writing(tmp_path):
    with pytest.raises(ValueError, match="match"):
        place_sensor_canvas(np.zeros((2, 2, 3)), plan("crop"), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_canvas_failure_cleans_products(tmp_path):
    content = np.zeros((8, 8, 3))
    content[-1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        place_sensor_canvas(content, plan("crop"), tmp_path)
    assert not list(tmp_path.rglob("*.npy"))


def test_canvas_refuses_to_overwrite_existing_product(tmp_path):
    result = place_sensor_canvas(np.zeros((8, 8, 3)), plan("crop"), tmp_path)
    with pytest.raises(FileExistsError):
        place_sensor_canvas(np.ones((8, 8, 3)), plan("crop"), tmp_path)
    assert np.all(np.load(result.path) == 0)
