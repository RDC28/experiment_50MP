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
