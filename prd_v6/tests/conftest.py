from __future__ import annotations

import numpy as np
import pytest

from prd_v6 import colour, constants, dng, pipeline

TINY_HW = (180, 270)  # divisible by the 2K (4.5) and 4K (2.25) binning ratios


def tiny_scene(hw=TINY_HW) -> np.ndarray:
    """Smooth linear-sRGB scene with edges and moderate colour, below sensor clip."""
    y, x = np.mgrid[0 : hw[0], 0 : hw[1]].astype(np.float32)
    y /= hw[0]
    x /= hw[1]
    scene = np.stack([0.05 + 0.5 * x, 0.08 + 0.45 * y, 0.1 + 0.3 * (1 - x) * y], axis=-1)
    scene[(x - 0.3) ** 2 + (y - 0.5) ** 2 < 0.03] *= np.array([1.4, 0.7, 0.6], dtype=np.float32)
    scene[:, int(hw[1] * 0.7) : int(hw[1] * 0.75)] *= 0.3
    return np.clip(scene, 0.0, 0.7)


def write_tiny_raw(path, *, hw=TINY_HW, seed=1, neutral=None) -> np.ndarray:
    """A 14-bit RGGB DNG standing in for the A1 II ARW (black 512, clip 15360)."""
    neutral = colour.prior_neutral(pipeline.camera_matrix()) if neutral is None else neutral
    camera = colour.linear_srgb_to_camera(tiny_scene(hw), neutral, pipeline.camera_matrix(), constants.SIGNAL_14)
    layout = np.tile(np.array([[0, 1], [1, 2]], dtype=np.int8), (hw[0] // 2, hw[1] // 2))
    signal = pipeline.mosaic(camera, layout)
    rng = np.random.default_rng(seed)
    electrons = rng.poisson(np.clip(signal, 0, None) * constants.GAIN_E_PER_DN14).astype(np.float32)
    raw = electrons / constants.GAIN_E_PER_DN14 + rng.normal(0, 1.9, signal.shape) + constants.BLACK_14
    samples = np.clip(np.rint(raw), 0, constants.CLIP_14).astype(np.uint16)
    dng.write_dng(
        path,
        samples,
        cfa_pattern="RGGB",
        color_matrix=pipeline.camera_matrix(),
        as_shot_neutral=neutral,
        black=constants.BLACK_14,
        white=constants.CLIP_14,
    )
    return samples


@pytest.fixture(scope="session")
def tiny_raw(tmp_path_factory):
    path = tmp_path_factory.mktemp("reference") / "tiny_reference.dng"
    write_tiny_raw(path)
    return path
