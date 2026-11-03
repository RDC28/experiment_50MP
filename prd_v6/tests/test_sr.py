from __future__ import annotations

import numpy as np
import pytest

from prd_v6 import sr


def nearest_x4(tile: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(tile, 4, axis=0), 4, axis=1)


@pytest.mark.parametrize(("length", "tile", "overlap"), [(100, 32, 8), (32, 32, 8), (20, 32, 8), (129, 64, 16)])
def test_tiles_cover_the_image_and_end_at_the_edge(length, tile, overlap):
    starts = sr.tile_starts(length, tile, overlap)
    assert starts[0] == 0
    assert starts[-1] + min(tile, length) == length
    assert all(b - a <= tile - overlap for a, b in zip(starts, starts[1:], strict=False))


def test_feathered_tiles_and_ensemble_reconstruct_an_equivariant_model_exactly():
    image = np.random.default_rng(0).uniform(0, 1, (70, 90, 3)).astype(np.float32)
    result = sr.upscale(image, nearest_x4, scale=4, tile=32, overlap=8, ensemble=True)
    assert np.allclose(result.ensemble, nearest_x4(image), atol=1e-6)
    assert np.allclose(result.identity, nearest_x4(image), atol=1e-6)
    assert result.passes == 8 * result.tiles


def test_ensemble_averages_dihedral_members():
    """A model that is not flip-equivariant gives an ensemble different from its identity member."""

    def biased(tile):
        out = nearest_x4(tile)
        out[:, : out.shape[1] // 2] *= 0.9
        return out

    image = np.full((40, 40, 3), 0.5, dtype=np.float32)
    result = sr.upscale(image, biased, scale=4, tile=64, overlap=8, ensemble=True)
    assert not np.allclose(result.ensemble, result.identity)
    assert np.allclose(result.ensemble.mean(), 0.5 * 0.95, atol=1e-3)


def test_wrong_model_output_shape_is_an_error():
    with pytest.raises(ValueError):
        sr.upscale(np.zeros((16, 16, 3), np.float32), lambda t: t, scale=4, tile=16, overlap=4)
