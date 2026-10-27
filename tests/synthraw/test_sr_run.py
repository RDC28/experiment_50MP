"""Tiny end-to-end PRD reconstruction orchestration checks, without torch."""

from dataclasses import replace

import numpy as np
import pytest

from synthraw.config import ExperimentConfig
from synthraw.environment import EnvironmentReport, TileParameters
from synthraw.geometry import TargetGeometryPlanner
from synthraw.super_resolution import SuperResolutionEngine, resize_exact


def environment(tile_size=16):
    return EnvironmentReport(None, None, None, None, 1024**3, 1024**3, 1, TileParameters(tile_size, 4, True, False))


@pytest.mark.parametrize("mode", ["OUTPAINT_VERTICAL", "CROP_HORIZONTAL", "FIT_WITH_SENSOR_MARGINS"])
def test_run_lanczos_all_sensor_fit_modes(tmp_path, mode):
    config = ExperimentConfig(SR_BACKEND="LANCZOS", SENSOR_FIT_MODE=mode, FAST_DEV_RUN=True)
    source = np.zeros((20, 48, 3), dtype=np.float32)
    source[..., 0] = np.linspace(0.1, 0.9, 48)
    source[..., 1] = np.linspace(0.2, 0.8, 20)[:, None]
    source[..., 2] = 0.4
    geometry = TargetGeometryPlanner().plan(48, 20, config)
    result = SuperResolutionEngine().run(source, geometry, config, environment(), tmp_path)
    canvas = np.load(result.canvas.path, mmap_mode="r")
    support = np.load(result.canvas.source_supported_mask)
    generated = np.load(result.canvas.generated_detail_mask)
    assert canvas.shape == (geometry.height, geometry.width, 3)
    assert result.tiled.backend == "LANCZOS"
    assert result.tiled.fallback_reasons == ()
    assert result.tiled.benchmark_tile_count == 3
    assert result.tiled.estimated_seconds == result.tiled.benchmark_seconds * result.tiled.tile_count
    assert result.seams is not None
    assert np.load(result.seams.residual_map_path).shape == (80, 192)
    assert result.seam_unavailable_reason is None
    assert result.source_fidelity.mean_absolute_error < 0.01
    left, top, right, bottom = map(round, geometry.content_box)
    assert result.scale_plan.content_size == (right - left, bottom - top)
    expected = resize_exact(
        np.load(result.tiled.path, mmap_mode="r"),
        tmp_path / "expected.npy",
        width=right - left,
        height=bottom - top,
        source_box=tuple(value * result.scale_plan.scale for value in geometry.crop_box),
    )
    np.testing.assert_array_equal(canvas[top:bottom, left:right], np.load(expected))
    if mode == "OUTPAINT_VERTICAL":
        np.testing.assert_array_equal(generated, 1 - support)
        np.testing.assert_array_equal(canvas[0], canvas[top])
        assert result.canvas.provenance.to_dict()["provenance"] == "F"
    elif mode == "FIT_WITH_SENSOR_MARGINS":
        assert np.all(canvas[support == 0] == 0)
        assert np.all(generated == 0)
    else:
        assert np.all(support == 1)
        assert np.all(generated == 0)
        assert geometry.crop_box[0] > 0


def test_run_records_fallback_and_single_tile_diagnostic(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("mock GPU unavailable")

    monkeypatch.setattr("synthraw.learned_sr.load_learned_backend", unavailable)
    config = ExperimentConfig(SR_BACKEND="HAT", FAST_DEV_RUN=True)
    geometry = TargetGeometryPlanner().plan(6, 4, config)
    # Keep this single-tile fallback fixture much smaller than the dev canvas.
    geometry = replace(geometry, width=12, height=8, content_box=(0, 0, 12, 8), outpaint_bands=())
    result = SuperResolutionEngine().run(
        np.full((4, 6, 3), 0.5, dtype=np.float32), geometry, config, environment(16), tmp_path
    )
    assert result.tiled.backend == "LANCZOS"
    assert len(result.tiled.fallback_reasons) == 2
    assert all("mock GPU unavailable" in reason for reason in result.tiled.fallback_reasons)
    assert result.seams is None
    assert "Single tile" in result.seam_unavailable_reason
    assert result.tiled.benchmark_tile_count == 1
    np.testing.assert_allclose(np.load(result.canvas.path), 0.5)


def test_generative_detail_strength_is_recorded_as_ignored(tmp_path):
    config = ExperimentConfig(SR_BACKEND="LANCZOS", GENERATIVE_DETAIL_STRENGTH=0.4, FAST_DEV_RUN=True)
    geometry = TargetGeometryPlanner().plan(6, 4, config)
    geometry = replace(geometry, width=12, height=8, content_box=(0, 0, 12, 8), outpaint_bands=())
    source = np.full((4, 6, 3), 0.5, dtype=np.float32)
    result = SuperResolutionEngine().run(source, geometry, config, environment(16), tmp_path)
    assert result.tiled.backend == "LANCZOS"
    assert result.tiled.fallback_reasons == (
        "GENERATIVE_DETAIL_STRENGTH=0.4 ignored: no generative detail backend is implemented; "
        "fidelity output only, no unseen texture added",
    )
    np.testing.assert_allclose(np.load(result.canvas.path), 0.5)
