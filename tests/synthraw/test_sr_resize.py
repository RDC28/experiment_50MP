"""Tiny tests for the PRD post-SR exact resize."""

import numpy as np
import pytest

from synthraw.super_resolution import resize_exact


@pytest.mark.parametrize("size", [(19, 11), (3, 2), (1, 1), (8, 6)])
def test_constant_is_preserved_at_exact_dimensions(tmp_path, size):
    source = np.full((6, 8, 3), 0.37, dtype=np.float32)
    path = resize_exact(source, tmp_path / "exact.npy", width=size[0], height=size[1])
    result = np.load(path, mmap_mode="r")
    assert result.shape == (size[1], size[0], 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, 0.37, atol=1e-7)


def test_identity_preserves_pixels_from_memmap(tmp_path):
    source_path = tmp_path / "source.npy"
    source = np.random.default_rng(41).random((9, 13, 3)).astype(np.float32)
    np.save(source_path, source)
    path = resize_exact(np.load(source_path, mmap_mode="r"), tmp_path / "exact.npy", width=13, height=9)
    np.testing.assert_allclose(np.load(path), source, atol=1e-7)
    np.testing.assert_array_equal(np.load(source_path), source)


def test_downsample_antialiases_checkerboard(tmp_path):
    pattern = (np.indices((32, 32)).sum(axis=0) % 2).astype(np.float32)
    source = np.repeat(pattern[..., None], 3, axis=2)
    path = resize_exact(source, tmp_path / "exact.npy", width=8, height=8)
    np.testing.assert_allclose(np.load(path)[2:-2, 2:-2], 0.5, atol=1e-3)


def test_resize_preserves_ramp_alignment_and_rgb_channels(tmp_path):
    ramp = np.linspace(0, 1, 16, dtype=np.float32)
    source = np.broadcast_to(np.stack((ramp, ramp * 0.5, ramp * 0), axis=1), (8, 16, 3))
    path = resize_exact(source, tmp_path / "exact.npy", width=32, height=16)
    result = np.load(path)
    expected = ((np.arange(32) + 0.5) / 2 - 0.5) / 15
    np.testing.assert_allclose(result[8, 4:-4, 0], expected[4:-4], atol=0.002)
    np.testing.assert_allclose(result[:, 4:-4, 1], result[:, 4:-4, 0] * 0.5, atol=1e-7)
    assert np.all(result[..., 2] == 0)


@pytest.mark.parametrize("width,height", [(0, 2), (2, -1), (True, 2), (2.5, 2)])
def test_invalid_dimensions_leave_no_output(tmp_path, width, height):
    with pytest.raises(ValueError):
        resize_exact(np.zeros((2, 2, 3)), tmp_path / "exact.npy", width=width, height=height)
    assert not list(tmp_path.iterdir())


def test_nonfinite_source_cleans_partial_output(tmp_path):
    source = np.zeros((4, 4, 3))
    source[-1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        resize_exact(source, tmp_path / "exact.npy", width=7, height=7)
    assert not list(tmp_path.iterdir())


def test_existing_source_is_never_overwritten(tmp_path):
    path = tmp_path / "source.npy"
    np.save(path, np.zeros((2, 2, 3), dtype=np.float32))
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        resize_exact(np.load(path, mmap_mode="r"), path, width=3, height=3)
    assert path.read_bytes() == original


def test_integer_crop_keeps_exact_source_pixels(tmp_path):
    source = np.random.default_rng(42).random((12, 18, 3)).astype(np.float32)
    path = resize_exact(source, tmp_path / "crop.npy", width=9, height=7, source_box=(3, 2, 12, 9))
    np.testing.assert_allclose(np.load(path), source[2:9, 3:12], atol=1e-7)


def test_fractional_crop_preserves_pixel_centre_alignment(tmp_path):
    y, x = np.indices((32, 48), dtype=np.float32)
    source = np.stack((x / 47, y / 31, (x + y) / 78), axis=-1)
    box = (8.25, 6.75, 38.75, 25.25)
    path = resize_exact(source, tmp_path / "crop.npy", width=24, height=16, source_box=box)
    result = np.load(path)
    expected_x = (box[0] + (np.arange(24) + 0.5) * (box[2] - box[0]) / 24 - 0.5) / 47
    expected_y = (box[1] + (np.arange(16) + 0.5) * (box[3] - box[1]) / 16 - 0.5) / 31
    np.testing.assert_allclose(result[..., 0], np.broadcast_to(expected_x, (16, 24)), atol=0.001)
    np.testing.assert_allclose(result[..., 1], np.broadcast_to(expected_y[:, None], (16, 24)), atol=0.001)


def test_geometry_crop_maps_from_sr_grid_to_sensor_content(tmp_path):
    from synthraw.config import ExperimentConfig
    from synthraw.geometry import TargetGeometryPlanner

    config = ExperimentConfig(FAST_DEV_RUN=True, SENSOR_FIT_MODE="CROP_HORIZONTAL")
    plan = TargetGeometryPlanner().plan(48, 20, config)
    # Model output lives on an x4 grid. Crop edges must be scaled, never rounded.
    sr = np.full((80, 192, 3), 0.4, dtype=np.float32)
    box = tuple(edge * 4 for edge in plan.crop_box)
    left, top, right, bottom = map(round, plan.content_box)
    path = resize_exact(sr, tmp_path / "crop.npy", width=right - left, height=bottom - top, source_box=box)
    result = np.load(path, mmap_mode="r")
    assert result.shape == (plan.height, plan.width, 3)
    np.testing.assert_allclose(result, 0.4, atol=1e-7)


@pytest.mark.parametrize(
    "box",
    [
        (-0.1, 0, 3, 3),
        (0, 0, 5, 3),
        (0, 0, 3, 5),
        (2, 0, 1, 3),
        (0, 2, 3, 2),
        (0, 0, float("nan"), 3),
        (0, 0, 3, float("inf")),
        (0, 0, 3),
    ],
)
def test_invalid_crop_is_rejected_before_output_creation(tmp_path, box):
    with pytest.raises(ValueError, match="source box"):
        resize_exact(np.zeros((4, 4, 3)), tmp_path / "crop.npy", width=3, height=3, source_box=box)
    assert not list(tmp_path.iterdir())
