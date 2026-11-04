"""Realism statistics of a Bayer mosaic, compared against a reference raw (report only)."""

from types import SimpleNamespace

import numpy as np
import pytest

from synthraw.realism import active_area, compare, measure_mosaic, measure_raw, photon_transfer_fit

RGGB = np.array([[0, 1], [3, 2]])
BLACK, WHITE = 512.0, 16383.0
GAIN_DN2_PER_DN, READ_DN = 0.25, 3.0


def _mosaic(*, row_banding=0.0, hot=(), texture=(), seed=0):
    """Horizontal ramp with shot plus read noise, optional row banding, hot photosites and scene texture.

    ``texture`` is a (low, high) signal range in DN whose every photosite gets 300 DN of extra
    scatter, so no block in those signal bins is flat, as in the textured mid-tones of a real scene.
    """
    rng = np.random.default_rng(seed)
    rows, cols = 512, 1024
    signal = np.broadcast_to(np.geomspace(2, 0.6 * (WHITE - BLACK), cols), (rows, cols))
    noisy = signal + rng.normal(size=(rows, cols)) * np.sqrt(GAIN_DN2_PER_DN * signal + READ_DN**2)
    if texture:
        textured = (signal >= texture[0]) & (signal <= texture[1])
        noisy = noisy + 300 * rng.normal(size=(rows, cols)) * textured
    noisy += row_banding * rng.normal(size=(rows, 1))
    mosaic = np.clip(np.round(noisy + BLACK), 0, WHITE)
    for row, column in hot:
        mosaic[row, column] = WHITE
    return mosaic.astype(np.uint16)


def test_photon_transfer_recovers_the_simulated_noise_model():
    planes = measure_mosaic(_mosaic(), black=BLACK, white=WHITE, pattern=RGGB)["planes"]
    assert set(planes) == {"R", "G1", "G2", "B"}
    fit = planes["G1"]["photon_transfer_fit"]
    assert fit["dn2_per_dn"] == pytest.approx(GAIN_DN2_PER_DN, rel=0.25)
    assert fit["full_well_e"] == pytest.approx((WHITE - BLACK) / GAIN_DN2_PER_DN, rel=0.25)
    assert planes["G1"]["defects"]["hot_count"] == 0
    assert planes["G1"]["fraction_clipped"] == 0


def test_photon_transfer_ignores_signal_bins_without_a_flat_block():
    planes = measure_mosaic(_mosaic(texture=(400, 1600)), black=BLACK, white=WHITE, pattern=RGGB)["planes"]
    for name in ("R", "G1", "G2", "B"):
        fit = planes[name]["photon_transfer_fit"]
        assert fit["dn2_per_dn"] == pytest.approx(GAIN_DN2_PER_DN, rel=0.25)
        assert fit["read_noise_e"] == pytest.approx(READ_DN / GAIN_DN2_PER_DN, rel=0.5)


def test_row_banding_and_hot_photosites_are_detected():
    clean = measure_mosaic(_mosaic(), black=BLACK, white=WHITE, pattern=RGGB)["planes"]["R"]
    banded = measure_mosaic(
        _mosaic(row_banding=4.0, hot=[(100, 200), (300, 600)]), black=BLACK, white=WHITE, pattern=RGGB
    )["planes"]["R"]
    assert banded["darkest_blocks"]["row_banding_rms_dn"] > 2 > clean["darkest_blocks"]["row_banding_rms_dn"]
    psd = banded["banding_psd"]
    assert psd["row_axis_power_over_offaxis"] > 3 * clean["banding_psd"]["row_axis_power_over_offaxis"]
    assert banded["defects"]["hot_count"] == 2


def test_comparison_ratios_are_one_for_identical_raws_and_reject_non_bayer():
    measured = measure_mosaic(_mosaic(), black=BLACK, white=WHITE, pattern=RGGB)
    reference = {**measured, "as_shot_neutral": [0.5, 1, 0.8]}
    rows = compare(reference, reference)
    assert rows and all(row["ratio"] == pytest.approx(1) for row in rows if row["ratio"] is not None)
    with pytest.raises(ValueError, match="RGGB-family"):
        measure_mosaic(_mosaic(), black=BLACK, white=WHITE, pattern=np.array([[0, 1], [1, 0]]))


def test_photon_transfer_fit_matches_the_measured_plane():
    mosaic = _mosaic()
    fit = photon_transfer_fit(mosaic[0::2, 1::2], black=BLACK, white=WHITE)
    assert fit == measure_mosaic(mosaic, black=BLACK, white=WHITE, pattern=RGGB)["planes"]["G1"]["photon_transfer_fit"]
    assert photon_transfer_fit(np.full((8, 8), BLACK), black=BLACK, white=WHITE) is None


class _FakeArw:
    """rawpy as the A1 II reports it: the whole array visible, the active area as crop_*; hot margin."""

    def __init__(self, margin=12, *, pattern=RGGB, black=(512, 512, 512, 512)):
        active = _mosaic()
        rows, cols = active.shape[0] + 2 * margin, active.shape[1] + 2 * margin
        self.raw_image_visible = np.full((rows, cols), int(WHITE), dtype=np.uint16)
        self.raw_image_visible[margin : margin + active.shape[0], margin : margin + active.shape[1]] = active
        self.raw_pattern = pattern
        self.color_desc = b"RGBG"
        self.black_level_per_channel = list(black)
        self.white_level = int(WHITE)
        self.camera_white_level_per_channel = None
        self.camera_whitebalance = [2.0, 1.0, 1.5, 1.0]
        self.rgb_xyz_matrix = np.eye(4, 3)
        self.sizes = SimpleNamespace(
            iheight=rows, iwidth=cols, crop_top_margin=margin, crop_left_margin=margin,
            crop_height=active.shape[0], crop_width=active.shape[1],
        )  # fmt: skip

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _measure_fake(monkeypatch, tmp_path, fake):
    rawpy = pytest.importorskip("rawpy")
    monkeypatch.setattr(rawpy, "imread", lambda path: fake)
    path = tmp_path / "fake.arw"
    path.write_bytes(b"not a TIFF")
    return measure_raw(path)


def test_measure_raw_measures_a_camera_raw_on_its_active_area(monkeypatch, tmp_path):
    """Regression: the noise gate measured the A1 II's whole 8672 x 5784 array, border included."""
    measured = _measure_fake(monkeypatch, tmp_path, _FakeArw())
    assert measured["measured_box"] == [12, 12, 12 + 512, 12 + 1024]
    assert measured["shape"] == [512, 1024]
    assert all(plane["fraction_clipped"] == 0 for plane in measured["planes"].values())  # no hot margin


def test_measure_raw_ignores_the_black_of_a_colour_the_mosaic_lacks(monkeypatch, tmp_path):
    """LibRaw reports [black, black, black, 0] when the fourth colour index is unused."""
    fake = _FakeArw(pattern=np.array([[0, 1], [1, 2]]), black=(512, 512, 512, 0))
    assert _measure_fake(monkeypatch, tmp_path, fake)["black"] == 512


def test_active_area_without_a_crop_is_the_visible_array_and_rejects_an_odd_origin():
    fake = _FakeArw()
    fake.sizes.crop_height = fake.sizes.crop_width = 0
    assert active_area(fake) == (slice(0, None), slice(0, None))
    with pytest.raises(ValueError):
        active_area(_FakeArw(margin=3))
