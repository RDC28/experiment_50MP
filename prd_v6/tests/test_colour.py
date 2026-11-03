from __future__ import annotations

import numpy as np
import pytest

from prd_v6 import colour, constants, pipeline


def test_srgb_transfer_round_trips():
    values = np.linspace(0, 1, 1001, dtype=np.float32)
    assert np.allclose(colour.srgb_encode(colour.srgb_decode(values)), values, atol=1e-6)


def test_camera_transform_inverts_exactly():
    rng = np.random.default_rng(0)
    linear = rng.uniform(0, 1, (8, 8, 3)).astype(np.float32)
    neutral = colour.prior_neutral(pipeline.camera_matrix())
    camera = colour.linear_srgb_to_camera(linear, neutral, pipeline.camera_matrix(), constants.SIGNAL_16)
    back = colour.camera_to_linear_srgb(camera, neutral, pipeline.camera_matrix(), constants.SIGNAL_16)
    assert np.allclose(back, linear, atol=1e-5)


def test_render_white_maps_to_clip_in_the_largest_channel():
    """LibRaw's convention: render white is the neutral scaled so its largest channel reaches clip."""
    neutral = colour.prior_neutral(pipeline.camera_matrix())
    white = pipeline.to_camera(np.ones((1, 1, 3), dtype=np.float32), neutral)[0, 0]
    assert np.allclose(white, neutral / neutral.max() * constants.SIGNAL_16, rtol=1e-4)


def test_prior_sits_between_tungsten_and_daylight():
    cam = pipeline.camera_matrix()
    prior = colour.prior_neutral(cam)
    daylight = colour.daylight_neutral(cam)
    assert prior[1] == pytest.approx(1.0)
    assert prior == pytest.approx([0.503, 1.0, 0.550], abs=0.01)
    assert daylight == pytest.approx([0.3744, 1.0, 0.7401], abs=0.01)


def test_neutral_from_libraw_multipliers():
    assert colour.neutral_from_multipliers([2.0, 1.0, 1.8, 1.0]) == pytest.approx([0.5, 1.0, 1 / 1.8])


def test_uint16_encoding_uses_the_full_range():
    encoded = np.array([0.0, 0.5, 1.0, 1.2], dtype=np.float32)
    assert colour.to_uint16(encoded).tolist() == [0, 32768, 65535, 65535]
