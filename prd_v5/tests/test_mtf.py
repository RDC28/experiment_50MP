"""Known sampled edges verify response without a model or full-sized raster."""

import json

import numpy as np
import pytest
from scipy.special import erf

from prd_v5.mtf import edge_mtf

EVIDENCE = {"edge": "synthetic isolated straight edge", "linearization": "analytic linear intensity"}


@pytest.mark.parametrize("sigma", [1.5, 3, 6])
def test_gaussian_edge_has_expected_bandwidth(sigma):
    spacing = 0.125
    x = np.arange(-64, 64, spacing)
    esf = 0.5 * (1 + erf(x / (np.sqrt(2) * sigma)))
    result = edge_mtf(esf, sample_spacing_px=spacing, evidence=EVIDENCE)
    for level, name in [(0.5, "mtf50"), (0.1, "mtf10")]:
        expected = np.sqrt(-2 * np.log(level)) / (2 * np.pi * sigma)
        assert result[name]["frequency"] == pytest.approx(expected, rel=0.002)
    assert max(result["frequencies"]) <= 0.5
    assert result["response"][0] == 1
    json.dumps(result, allow_nan=False)


def test_sharp_edge_has_no_invented_crossing():
    result = edge_mtf(np.r_[np.zeros(32), np.ones(32)], evidence=EVIDENCE)
    np.testing.assert_allclose(result["response"], 1)
    assert result["mtf50"]["frequency"] is None
    assert "Nyquist" in result["mtf10"]["reason"]


def test_polarity_gain_offset_and_input_immutability():
    profile = erf(np.linspace(-8, 8, 128))
    original = profile.copy()
    first = edge_mtf(profile, evidence=EVIDENCE)
    second = edge_mtf(10 - 3 * profile, evidence=EVIDENCE)
    np.testing.assert_allclose(first["response"], second["response"], atol=1e-14)
    np.testing.assert_array_equal(profile, original)


@pytest.mark.parametrize("profile,evidence", [(None, None), (np.zeros(32), None), (np.ones(32), EVIDENCE)])
def test_missing_measurement_stays_not_measured(profile, evidence):
    result = edge_mtf(profile, evidence=evidence)
    assert result["status"] == "not measured"
    assert "mtf50" not in result


@pytest.mark.parametrize(
    "profile",
    [
        np.zeros((16, 2)),
        np.zeros(15),
        np.zeros(65537),
        [np.nan] * 32,
        [np.inf] * 32,
        np.ones(32, complex),
        np.ones(32, bool),
    ],
)
def test_invalid_profiles(profile):
    with pytest.raises(ValueError, match="profile"):
        edge_mtf(profile, evidence=EVIDENCE)


@pytest.mark.parametrize("spacing", [0, -1, np.nan, np.inf, True, 1j, "1", None])
def test_invalid_spacing(spacing):
    with pytest.raises(ValueError, match="spacing"):
        edge_mtf(np.arange(32), sample_spacing_px=spacing, evidence=EVIDENCE)


def test_evidence_required():
    with pytest.raises(ValueError, match="evidence"):
        edge_mtf(np.arange(32), evidence={"edge": "unknown"})


@pytest.mark.parametrize("axis", ["x", "y"])
def test_tiff_edge_extraction_decodes_before_averaging(tmp_path, axis):
    import tifffile

    from prd_v5.mtf import tiff_edge_mtf

    profile = 0.5 * (1 + erf(np.linspace(-8, 8, 128)))
    linear = np.broadcast_to(profile[None, :, None], (4, 128, 3)).copy()
    encoded = np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)
    pixels = np.rint(encoded * 65535).astype(np.uint16)
    if axis == "y":
        pixels = pixels.transpose(1, 0, 2)
    path = tmp_path / "edge.tif"
    tifffile.imwrite(path, pixels, photometric="rgb")
    region = (0, 0, *pixels.shape[:2])
    result = tiff_edge_mtf(path, region=region, normal_axis=axis, edge_evidence="fixture straight edge")
    decoded = pixels.astype(float) / 65535
    decoded = np.where(decoded <= 0.04045, decoded / 12.92, ((decoded + 0.055) / 1.055) ** 2.4)
    expected = edge_mtf(decoded[..., 0].mean(axis=0 if axis == "x" else 1), evidence=EVIDENCE)
    np.testing.assert_allclose(result["response"], expected["response"], atol=1e-13)
    assert result["region"] == list(region)
    assert result["mtf50"] == pytest.approx(expected["mtf50"])
    json.dumps(result, allow_nan=False)


def test_tiff_edge_missing_selection_does_not_open_file():
    from prd_v5.mtf import tiff_edge_mtf

    assert tiff_edge_mtf("absent")["status"] == "not measured"


@pytest.mark.parametrize("region", [(0, 0, 2, 15), (0, 0, 5, 128), (0, 0, 4, 129)])
def test_tiff_edge_rejects_short_or_out_of_bounds_regions(tmp_path, region):
    import tifffile

    from prd_v5.mtf import tiff_edge_mtf

    path = tmp_path / "edge.tif"
    tifffile.imwrite(path, np.zeros((4, 128, 3), np.uint16), photometric="rgb")
    with pytest.raises(ValueError):
        tiff_edge_mtf(path, region=region, edge_evidence="fixture")


@pytest.mark.parametrize("dtype,shape", [(np.uint8, (4, 128, 3)), (np.uint16, (1025, 1024, 3))])
def test_tiff_edge_rejects_encoding_and_oversized_roi(tmp_path, dtype, shape):
    import tifffile

    from prd_v5.mtf import tiff_edge_mtf

    path = tmp_path / "invalid.tif"
    tifffile.imwrite(path, np.zeros(shape, dtype), photometric="rgb")
    with pytest.raises(ValueError):
        tiff_edge_mtf(path, region=(0, 0, *shape[:2]), edge_evidence="fixture")


def test_tiff_edge_flat_crop_stays_not_measured(tmp_path):
    import tifffile

    from prd_v5.mtf import tiff_edge_mtf

    path = tmp_path / "flat.tif"
    tifffile.imwrite(path, np.full((8, 128, 3), 32768, np.uint16), photometric="rgb")
    result = tiff_edge_mtf(path, region=(2, 8, 6, 120), edge_evidence="fixture")
    assert result["status"] == "not measured"
    assert result["region"] == [2, 8, 6, 120]
