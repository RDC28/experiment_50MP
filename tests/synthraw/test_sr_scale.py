"""Arithmetic-only coverage of PRD_v3's 2K/4K SR scale selection."""

import numpy as np
import pytest

from synthraw.environment import TileParameters
from synthraw.super_resolution import SuperResolutionEngine


@pytest.mark.parametrize(
    "source,target,intermediate",
    [
        ((2048, 868), (10872, 4608), (8192, 3472)),
        ((4096, 1736), (10872, 4608), (16384, 6944)),
        ((868, 2048), (4608, 10872), (3472, 8192)),
        ((1736, 4096), (4608, 10872), (6944, 16384)),
        ((3840, 2160), (8640, 5760), (15360, 8640)),
        ((4096, 2048), (12287, 6143), (16384, 8192)),
        ((2049, 1024), (512, 256), (8196, 4096)),
    ],
)
def test_scale_is_always_the_pinned_x4(source, target, intermediate):
    result = SuperResolutionEngine.plan_scale(
        source_width=source[0], source_height=source[1], content_width=target[0], content_height=target[1]
    )
    assert result.scale == 4
    assert result.intermediate_size == intermediate
    assert result.content_size == target


@pytest.mark.parametrize("source", [(2048, 868), (4096, 1736), (3840, 2160)])
def test_2k_and_4k_sources_keep_the_learned_x4_backend(tmp_path, monkeypatch, source):
    """Regression: 4K sources planned x2, which the x4-only checkpoints reject, so they fell back to Lanczos."""

    def load_x4_only(name, *, scale, cache_dir, accept_licence):
        # Mirrors load_learned_backend: the pinned SwinIR checkpoint is x4 and other scales fail.
        if scale != 4:
            raise ValueError(f"Checkpoint scale x4 does not match requested x{scale}")
        return lambda patch: np.repeat(np.repeat(patch, 4, axis=0), 4, axis=1)

    monkeypatch.setattr("synthraw.learned_sr.load_learned_backend", load_x4_only)
    plan = SuperResolutionEngine.plan_scale(
        source_width=source[0], source_height=source[1], content_width=10872, content_height=4608
    )
    result = SuperResolutionEngine().reconstruct(
        np.full((6, 8, 3), 0.5, dtype=np.float32),
        tmp_path / "sr.npy",
        backend="SWINIR",
        scale=plan.scale,
        tiles=TileParameters(4, 1, True, False),
        cache_dir=tmp_path / "cache",
    )
    assert result.backend == "SWINIR" and result.fallback_reasons == ()
    assert np.load(result.path).shape == (24, 32, 3)


@pytest.mark.parametrize("field", ["source_width", "source_height", "content_width", "content_height"])
@pytest.mark.parametrize("invalid", [0, -1, True, 2.5, None])
def test_invalid_dimensions(field, invalid):
    dimensions = dict(source_width=4096, source_height=1736, content_width=10872, content_height=4608)
    dimensions[field] = invalid
    with pytest.raises(ValueError, match="positive integers"):
        SuperResolutionEngine.plan_scale(**dimensions)
