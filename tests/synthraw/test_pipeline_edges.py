"""Carry-over fix A: no dark border in camera-native RGB or the developed TIFF."""

from datetime import datetime

import numpy as np
import pytest
import tifffile
from PIL import Image

from synthraw.config import ExperimentConfig
from synthraw.pipeline import run_pipeline


def _border_ratios(image: np.ndarray, width: int = 2) -> np.ndarray:
    """Mean of each edge band over the adjacent inner band: top, bottom, left, right."""
    luma = np.asarray(image, dtype=float).mean(axis=2)
    bands = (
        (luma[:width], luma[width : 2 * width]),
        (luma[-width:], luma[-2 * width : -width]),
        (luma[:, :width], luma[:, width : 2 * width]),
        (luma[:, -width:], luma[:, -2 * width : -width]),
    )
    return np.array([edge.mean() / inner.mean() for edge, inner in bands])


@pytest.mark.parametrize("pattern", ["uniform", "colour_gradient"])
def test_smooth_input_has_no_border_discontinuity(tmp_path, pattern):
    rows, cols = np.mgrid[0:40, 0:96]
    if pattern == "uniform":
        pixels = np.full((40, 96, 3), (180, 120, 90))
    else:
        # Single channels reach 0 and 255 at the borders while every pixel stays well exposed.
        pixels = np.stack([255 * cols / 95, 255 * rows / 39, 255 * (1 - cols / 95)], axis=-1)
    source = tmp_path / "smooth.png"
    Image.fromarray(np.round(pixels).astype(np.uint8)).save(source)
    config = ExperimentConfig(OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL", FAST_DEV_RUN=True, SR_BACKEND="LANCZOS")
    result = run_pipeline(
        source, config, tmp_path / "runs", timestamp=datetime(2026, 1, 2, 3, 4, 5), fast_dev_max_side=192
    )
    intermediates = result.root / "intermediates"
    scene = np.load(intermediates / "03_scene_linear.npy")
    camera = np.load(intermediates / "05_camera_native_rgb.npy")
    sr = np.load(intermediates / "02_sr_rgb.npy")
    developed = tifffile.imread(result.tiff_path)
    # Border-to-inner ratios may follow the gradient but must not show a stage-induced step.
    np.testing.assert_allclose(_border_ratios(camera), _border_ratios(scene), atol=0.04)
    np.testing.assert_allclose(_border_ratios(developed), _border_ratios(sr), atol=0.04)
    assert np.all(_border_ratios(developed) > 0.9)
