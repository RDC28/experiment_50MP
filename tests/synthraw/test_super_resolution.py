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


@pytest.mark.parametrize("oom", [MemoryError("allocation failed"), RuntimeError("CUDA out of memory")])
def test_oom_retries_once_with_smaller_tiles(tmp_path, oom):
    source = np.random.default_rng(3).random((9, 11, 3)).astype(np.float32)
    seen = []

    def model(patch):
        seen.append(patch.shape)
        if len(seen) == 1:
            raise oom
        return patch

    result = SuperResolutionEngine().reconstruct_tiles(
        source, tmp_path / "sr.npy", model, scale=1, tiles=TileParameters(8, 5, True, False)
    )
    assert result.effective_tile_size == 4
    assert len(result.retry_reasons) == 1
    assert str(oom) in result.retry_reasons[0]
    assert "8 -> 4" in result.retry_reasons[0]
    assert max(seen[0][:2]) == 18
    assert max(max(shape[:2]) for shape in seen[1:]) <= 10
    np.testing.assert_allclose(np.load(result.path), source, atol=2e-7)
    assert list(tmp_path.iterdir()) == [result.path]


@pytest.mark.parametrize("tile_size", [1, 8])
def test_repeated_oom_stops_after_two_attempts(tmp_path, tile_size):
    calls = []

    def model(patch):
        calls.append(patch.shape)
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        SuperResolutionEngine().reconstruct_tiles(
            np.zeros((9, 9, 3)),
            tmp_path / "sr.npy",
            model,
            scale=1,
            tiles=TileParameters(tile_size, 0, True, False),
        )
    assert len(calls) == 2
    assert not list(tmp_path.iterdir())


def test_non_oom_runtime_failure_is_not_retried(tmp_path):
    calls = []

    def model(patch):
        calls.append(patch.shape)
        raise RuntimeError("invalid model weights")

    with pytest.raises(RuntimeError, match="invalid model weights"):
        SuperResolutionEngine().reconstruct_tiles(
            np.zeros((3, 3, 3)),
            tmp_path / "sr.npy",
            model,
            scale=1,
            tiles=TileParameters(4, 1, True, False),
        )
    assert len(calls) == 1
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("strip_rows", [1, 5, 64])
def test_seam_diagnostic_detects_injected_boundary_error(tmp_path, strip_rows):
    from synthraw.super_resolution import diagnose_tile_seams

    reference = np.zeros((17, 19, 3), dtype=np.float32)
    reconstructed = np.full_like(reference, 0.01)
    reconstructed[6::6] = 0.1
    reconstructed[:, 6::6] = 0.1
    result = diagnose_tile_seams(
        reconstructed,
        reference,
        tmp_path / "seams.npy",
        tile_size=4,
        overlap=1,
        scale=2,
        strip_rows=strip_rows,
    )
    assert result.ratio == pytest.approx(10)
    assert not result.meets_target
    np.testing.assert_allclose(np.load(result.residual_map_path, mmap_mode="r"), reconstructed[..., 0])


@pytest.mark.parametrize("error", [0, 0.02])
def test_uniform_residual_has_no_seam_penalty(tmp_path, error):
    from synthraw.super_resolution import diagnose_tile_seams

    reference = np.zeros((12, 15, 3), dtype=np.float32)
    result = diagnose_tile_seams(
        reference + error,
        reference,
        tmp_path / "seams.npy",
        tile_size=5,
        overlap=1,
        scale=1,
    )
    assert result.ratio == pytest.approx(1 if error else 0)
    assert result.meets_target


def test_seam_diagnostic_cleans_nonfinite_input_and_preserves_existing_output(tmp_path):
    from synthraw.super_resolution import diagnose_tile_seams

    source = np.zeros((12, 15, 3), dtype=np.float32)
    source[-1] = np.nan
    path = tmp_path / "seams.npy"
    with pytest.raises(ValueError, match="finite"):
        diagnose_tile_seams(source, source, path, tile_size=5, overlap=1, scale=1, strip_rows=2)
    assert not path.exists()
    path.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        diagnose_tile_seams(source, source, path, tile_size=5, overlap=1, scale=1)
    assert path.read_bytes() == b"original"


def test_seam_diagnostic_requires_boundary_and_interior(tmp_path):
    from synthraw.super_resolution import diagnose_tile_seams

    source = np.zeros((3, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="both tile boundaries"):
        diagnose_tile_seams(source, source, tmp_path / "seams.npy", tile_size=5, overlap=1, scale=1)
    assert not list(tmp_path.iterdir())
