"""The reference loader returns the camera's active area, not LibRaw's larger visible array."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from prd_v6 import reference

ACTIVE = (16, 24)
MARGIN = 4  # even, like the A1 II's 12 photosite crop origin


class FakeRaw:
    """rawpy as the A1 II reports it: full array visible (margins 0), active area in crop_*."""

    def __init__(self, top=MARGIN, left=MARGIN, height=ACTIVE[0], width=ACTIVE[1]):
        full = (ACTIVE[0] + 2 * MARGIN, ACTIVE[1] + 2 * MARGIN)
        self.raw_image_visible = np.full(full, 65535, dtype=np.uint16)  # sentinel margin
        self.raw_image_visible[MARGIN : MARGIN + ACTIVE[0], MARGIN : MARGIN + ACTIVE[1]] = 1000
        self.raw_colors_visible = np.tile(np.array([[0, 1], [3, 2]]), (full[0] // 2, full[1] // 2))
        self.color_desc = b"RGBG"
        self.black_level_per_channel = [512] * 4
        self.white_level = 16383
        self.camera_white_level_per_channel = [15360] * 4
        self.camera_whitebalance = [2.0, 1.0, 1.5, 1.0]
        self.daylight_whitebalance = [2.0, 1.0, 1.5, 1.0]
        self.rgb_xyz_matrix = np.eye(4, 3)
        self.sizes = SimpleNamespace(
            raw_height=full[0], raw_width=full[1], iheight=full[0], iwidth=full[1],
            crop_top_margin=top, crop_left_margin=left, crop_height=height, crop_width=width,
        )  # fmt: skip

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_load_returns_the_active_area_with_its_pattern(monkeypatch):
    rawpy = pytest.importorskip("rawpy")
    monkeypatch.setattr(rawpy, "imread", lambda path: FakeRaw())
    ref = reference.load("fake.arw")
    assert ref.shape == ACTIVE
    assert ref.layout.shape == ACTIVE
    assert ref.pattern == "RGGB"
    assert ref.libraw["full_visible_size"] == [ACTIVE[0] + 2 * MARGIN, ACTIVE[1] + 2 * MARGIN]
    assert not ref.clipped().any()  # the 65535 margin never enters the reference
    assert (ref.raw_visible == 1000).all()


def test_no_reported_crop_keeps_the_visible_array(monkeypatch):
    rawpy = pytest.importorskip("rawpy")
    monkeypatch.setattr(rawpy, "imread", lambda path: FakeRaw(0, 0, 0, 0))
    assert reference.load("fake.arw").shape == (ACTIVE[0] + 2 * MARGIN, ACTIVE[1] + 2 * MARGIN)


def test_odd_crop_origin_is_unsuitable():
    with pytest.raises(reference.UnsuitableReference):
        reference.active_window(FakeRaw(top=3))


def test_crop_beyond_the_array_is_unsuitable():
    with pytest.raises(reference.UnsuitableReference):
        reference.active_window(FakeRaw(width=ACTIVE[1] + 2 * MARGIN))
