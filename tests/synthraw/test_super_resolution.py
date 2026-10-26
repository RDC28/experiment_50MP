"""Tiny callable-backed tests for the PRD tiled SR execution core."""

import numpy as np
import pytest

from synthraw.environment import TileParameters
from synthraw.super_resolution import SuperResolutionEngine


@pytest.mark.parametrize("shape", [(11, 17, 3), (1, 1, 3), (7, 4, 3)])
@pytest.mark.parametrize("overlap", [0, 2])
def test_tiling_matches_pointwise_model(tmp_path, shape, overlap):
    source = np.random.default_rng(2).random(shape).astype(np.float32)
    seen = []

    def model(patch):
        seen.append(patch.shape)
        return np.repeat(np.repeat(patch, 2, axis=0), 2, axis=1)

    result = SuperResolutionEngine().reconstruct_tiles(
        source, tmp_path / "sr.npy", model, scale=2, tiles=TileParameters(5, overlap, True, False)
    )
    expected = np.repeat(np.repeat(source, 2, axis=0), 2, axis=1)
    np.testing.assert_allclose(np.load(result.path, mmap_mode="r"), expected, atol=2e-7)
    assert result.tile_count == len(seen)
    assert max(max(s[:2]) for s in seen) <= 5 + 2 * overlap
    assert result.benchmark_seconds >= 0
    assert result.estimated_seconds == result.benchmark_seconds * len(seen)
    assert list(tmp_path.iterdir()) == [result.path]


@pytest.mark.parametrize("failure", ["shape", "nan", "raise", "source"])
def test_failed_tiles_remove_partial_files(tmp_path, failure):
    source = np.zeros((6, 6, 3), dtype=np.float32)
    if failure == "source":
        source[-1] = np.nan

    def model(patch):
        if failure == "raise":
            raise MemoryError("fake OOM")
        return patch[:-1] if failure == "shape" else patch + (np.nan if failure == "nan" else 0)

    with pytest.raises((ValueError, MemoryError)):
        SuperResolutionEngine().reconstruct_tiles(
            source, tmp_path / "sr.npy", model, scale=1, tiles=TileParameters(4, 1, True, False)
        )
    assert not list(tmp_path.iterdir())


def test_existing_output_is_preserved(tmp_path):
    path = tmp_path / "sr.npy"
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        SuperResolutionEngine().reconstruct_tiles(
            np.zeros((2, 2, 3)), path, lambda x: x, scale=1, tiles=TileParameters(4, 1, True, False)
        )
    assert path.read_bytes() == b"original"
