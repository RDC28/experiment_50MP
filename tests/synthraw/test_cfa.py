"""Tiny Bayer-phase and bounded-buffer contracts for PRD_v3."""

import numpy as np
import pytest

from synthraw.cfa import CFAMosaicer


@pytest.mark.parametrize("phase", ["RGGB", "BGGR", "GRBG", "GBRG"])
@pytest.mark.parametrize("strip_rows", [1, 2, 3, 128])
def test_phase_sampling_and_chunk_invariance(phase, strip_rows):
    rgb = np.arange(7 * 5 * 3, dtype=np.float64).reshape(7, 5, 3) - 50
    result = CFAMosaicer(phase).mosaic(rgb, strip_rows=strip_rows)
    expected = np.array([[rgb[y, x, "RGB".index(phase[2 * (y % 2) + x % 2])] for x in range(5)] for y in range(7)])
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result, CFAMosaicer(phase).mosaic(rgb))
    assert result.dtype == rgb.dtype


@pytest.mark.parametrize("phase", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_split_planes_preserve_both_greens_as_views(phase):
    mosaic = np.arange(35, dtype=np.uint16).reshape(7, 5)
    planes = CFAMosaicer(phase).split_planes(mosaic)
    assert set(planes) == {"R", "G1", "G2", "B"}
    green = 0
    for index, colour in enumerate(phase):
        if colour == "G":
            green += 1
        key = f"G{green}" if colour == "G" else colour
        np.testing.assert_array_equal(planes[key], mosaic[index // 2 :: 2, index % 2 :: 2])
        assert np.shares_memory(planes[key], mosaic)


def test_clean_and_noisy_memmap_outputs(tmp_path):
    clean = np.ones((9, 7, 3), dtype=np.float32)
    noisy = clean + np.random.default_rng(4).normal(0, 3, clean.shape).astype(np.float32)
    sampler = CFAMosaicer("GBRG")
    for name, source in (("09_clean_cfa.npy", clean), ("10_noisy_cfa.npy", noisy)):
        out = np.lib.format.open_memmap(tmp_path / name, mode="w+", dtype=np.float32, shape=(9, 7))
        assert sampler.mosaic(source, out=out, strip_rows=3) is out
        out.flush()
        np.testing.assert_array_equal(np.load(tmp_path / name), sampler.mosaic(source))
    assert np.min(sampler.mosaic(noisy)) < 0


@pytest.mark.parametrize("phase", ["rggb", "RGB", None, []])
def test_invalid_phase(phase):
    with pytest.raises(ValueError, match="phase"):
        CFAMosaicer(phase)


@pytest.mark.parametrize("source", [np.ones((2, 3)), np.ones((2, 3, 4)), np.ones((0, 2, 3)), np.ones((2, 3, 3), int)])
def test_invalid_rgb(source):
    with pytest.raises(ValueError, match="RGB"):
        CFAMosaicer().mosaic(source)


@pytest.mark.parametrize("strip_rows", [0, -1, True, 1.5])
def test_invalid_strip(strip_rows):
    with pytest.raises(ValueError, match="strip_rows"):
        CFAMosaicer().mosaic(np.ones((2, 3, 3)), strip_rows=strip_rows)


def test_output_validation_and_atomic_nonfinite_rejection():
    source = np.ones((5, 7, 3))
    for out in (source[:, :, 0], np.ones((5, 7), np.float32), np.ones((2, 7))):
        with pytest.raises(ValueError, match="out"):
            CFAMosaicer().mosaic(source, out=out)
    out = np.full((5, 7), -1.0)
    source[-1, -1, -1] = np.nan  # Even unsampled channels must be finite.
    with pytest.raises(ValueError, match="finite"):
        CFAMosaicer().mosaic(source, strip_rows=1, out=out)
    np.testing.assert_array_equal(out, -1)


@pytest.mark.parametrize("source", [np.ones((2, 3, 3)), np.ones((0, 3))])
def test_invalid_plane_input(source):
    with pytest.raises(ValueError, match="two-dimensional"):
        CFAMosaicer().split_planes(source)
