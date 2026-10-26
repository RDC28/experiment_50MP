"""Tiny pre-CFA radiance fixtures for PRD_v3 optical blur."""

import math

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from synthraw.optics import LensSimulator, OpticalBlurParameters


@pytest.mark.parametrize("rows", [1, 3, 9, 32])
def test_strip_halos_match_whole_frame(rows):
    source = np.random.default_rng(2).normal(size=(23, 19, 3)).astype(np.float32)
    parameters = OpticalBlurParameters(lens_blur=True, diffraction=True, defocus=True)
    actual = LensSimulator().blur_radiance(source, parameters, strip_rows=rows)
    expected = gaussian_filter(source, (parameters.sigma_px(), parameters.sigma_px(), 0), mode="reflect")
    np.testing.assert_array_equal(actual, expected)


def test_default_is_independent_identity_with_signed_hdr_headroom():
    source = np.array([[[-0.1, 0.5, 12.0]]])
    actual = LensSimulator().blur_radiance(source)
    np.testing.assert_array_equal(actual, source)
    assert not np.shares_memory(source, actual)


def test_channel_isolation_flux_and_softening():
    source = np.zeros((31, 31, 3))
    source[15, 15, 0] = 20
    result = LensSimulator().blur_radiance(source, OpticalBlurParameters(lens_blur=True, lens_sigma_px=1))
    assert 0 < result[15, 15, 0] < 20
    assert result.sum() == pytest.approx(20)
    assert np.count_nonzero(result[:, :, 1:]) == 0


def test_uniform_radiance_survives_boundaries():
    source = np.full((11, 9, 3), 3.2)
    result = LensSimulator().blur_radiance(source, OpticalBlurParameters(defocus=True), strip_rows=2)
    np.testing.assert_allclose(result, source)


def test_physical_psf_variances_and_fabricated_records():
    p = OpticalBlurParameters(diffraction=True, defocus=True, defocus_radius_px=4)
    airy = 1.028 * 0.550 * 2.8 / 4.16 / math.sqrt(8 * math.log(2))
    assert p.sigma_px() == pytest.approx(math.hypot(airy, 2))
    assert OpticalBlurParameters(diffraction=True, f_number=8).sigma_px() > airy
    records = LensSimulator().blur_records(p)
    assert all(r.provenance == "F" for r in records.values())
    assert records["effective_sigma_px"].value == p.sigma_px()


def test_memmap_output_and_input_preservation(tmp_path):
    source = np.ones((9, 7, 3), dtype=np.float32)
    out = np.memmap(tmp_path / "optics.bin", mode="w+", shape=source.shape, dtype=source.dtype)
    actual = LensSimulator().blur_radiance(source, OpticalBlurParameters(defocus=True), out=out, strip_rows=2)
    assert actual is out
    np.testing.assert_array_equal(out, source)
    np.testing.assert_array_equal(source, 1)


@pytest.mark.parametrize(
    "kwargs", [{"pixel_pitch_um": 0}, {"f_number": -1}, {"lens_sigma_px": float("nan")}, {"defocus": 1}]
)
def test_bad_psf_parameters(kwargs):
    with pytest.raises(ValueError):
        OpticalBlurParameters(**kwargs).sigma_px()


@pytest.mark.parametrize("source", [np.ones((5, 5)), np.ones((5, 5, 4)), np.ones((5, 5, 3), dtype=int)])
def test_rejects_non_rgb_radiance(source):
    with pytest.raises(ValueError):
        LensSimulator().blur_radiance(source)


def test_validation_precedes_output_writes():
    source = np.ones((7, 5, 3))
    source[-1, 0, 0] = np.inf
    out = np.full_like(source, -10)
    with pytest.raises(ValueError, match="finite"):
        LensSimulator().blur_radiance(source, out=out, strip_rows=2)
    np.testing.assert_array_equal(out, -10)


def test_rejects_alias_and_bad_strip_size():
    source = np.ones((7, 5, 3))
    with pytest.raises(ValueError, match="overlap"):
        LensSimulator().blur_radiance(source, out=source)
    with pytest.raises(ValueError, match="strip_rows"):
        LensSimulator().blur_radiance(source, strip_rows=0)


@pytest.mark.parametrize("rows", [1, 4, 30])
def test_radial_distortion_matches_analytic_linear_ramp(rows):
    from synthraw.optics import OpticalDistortionParameters

    y, x = np.mgrid[:11, :13]
    source = np.stack([x, y, 2 * x - y], axis=-1).astype(float)
    p = OpticalDistortionParameters(enabled=True, k1=0.1, k2=0.02)
    actual = LensSimulator().distort_radiance(source, p, strip_rows=rows)
    r2 = ((x - 6) / 6) ** 2 + ((y - 5) / 5) ** 2
    scale = 1 + 0.1 * r2 + 0.02 * r2**2
    source_x, source_y = 6 + (x - 6) * scale, 5 + (y - 5) * scale
    expected = np.stack([source_x, source_y, 2 * source_x - source_y], axis=-1)
    np.testing.assert_allclose(actual[2:-2, 2:-2], expected[2:-2, 2:-2], atol=1e-12)
    np.testing.assert_array_equal(actual[5, 6], source[5, 6])


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 9, 3), (7, 1, 3), (17, 19, 3)])
def test_distortion_strip_equivalence_and_hdr_identity(shape, tmp_path):
    from synthraw.optics import OpticalDistortionParameters

    source = np.random.default_rng(8).normal(size=shape).astype(np.float32) * 10
    simulator = LensSimulator()
    np.testing.assert_array_equal(simulator.distort_radiance(source), source)
    p = OpticalDistortionParameters(enabled=True, k1=-0.05)
    expected = simulator.distort_radiance(source, p, strip_rows=100)
    out = np.memmap(tmp_path / "distortion.bin", mode="w+", shape=shape, dtype=source.dtype)
    assert simulator.distort_radiance(source, p, strip_rows=2, out=out) is out
    np.testing.assert_array_equal(out, expected)


def test_distortion_uniform_channels_and_records():
    from synthraw.optics import OpticalDistortionParameters

    source = np.broadcast_to([-2.0, 0.0, 8.0], (13, 15, 3)).copy()
    p = OpticalDistortionParameters(enabled=True, k1=0.2)
    np.testing.assert_allclose(LensSimulator().distort_radiance(source, p), source)
    records = LensSimulator().distortion_records(p)
    assert records["enabled"].value is True
    assert records["k1"].value == 0.2
    assert all(record.provenance == "F" for record in records.values())


@pytest.mark.parametrize("kwargs", [{"enabled": 1}, {"k1": float("nan")}, {"k2": True}, {"k1": -1}])
def test_distortion_rejects_invalid_parameters_before_writing(kwargs):
    from synthraw.optics import OpticalDistortionParameters

    source = np.ones((5, 5, 3))
    out = np.full_like(source, -10)
    with pytest.raises(ValueError):
        LensSimulator().distort_radiance(source, OpticalDistortionParameters(**kwargs), out=out)
    np.testing.assert_array_equal(out, -10)


def test_vignetting_analytic_attenuation_symmetry_and_colour():
    from synthraw.optics import OpticalVignettingParameters

    source = np.broadcast_to([-2.0, 4.0, 16.0], (9, 11, 3)).copy()
    p = OpticalVignettingParameters(enabled=True, corner_loss_ev=2, power=2)
    actual = LensSimulator().vignette_radiance(source, p, strip_rows=2)
    y, x = np.mgrid[:9, :11]
    radius2 = (((x - 5) / 5) ** 2 + ((y - 4) / 4) ** 2) / 2
    np.testing.assert_allclose(actual, source * np.exp2(-2 * radius2**2)[:, :, None])
    np.testing.assert_array_equal(actual[4, 5], source[4, 5])
    np.testing.assert_array_equal(actual[0, 0], source[0, 0] / 4)
    np.testing.assert_array_equal(actual, actual[::-1, ::-1])
    assert np.all(np.diff(actual[4, 5:, 2]) <= 0)


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 9, 3), (7, 1, 3), (12, 14, 3)])
def test_vignetting_default_and_strip_memmap_equivalence(shape, tmp_path):
    from synthraw.optics import OpticalVignettingParameters

    source = np.random.default_rng(4).normal(size=shape).astype(np.float32)
    simulator = LensSimulator()
    identity = simulator.vignette_radiance(source)
    np.testing.assert_array_equal(identity, source)
    assert not np.shares_memory(identity, source)
    p = OpticalVignettingParameters(enabled=True, corner_loss_ev=1.5, power=0.5)
    expected = simulator.vignette_radiance(source, p, strip_rows=100)
    out = np.memmap(tmp_path / "vignetting.bin", mode="w+", shape=shape, dtype=source.dtype)
    assert simulator.vignette_radiance(source, p, strip_rows=1, out=out) is out
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(simulator.vignette_radiance(source, OpticalVignettingParameters(True, 0)), source)


def test_vignetting_records_are_fabricated():
    from synthraw.optics import OpticalVignettingParameters

    records = LensSimulator().vignetting_records(OpticalVignettingParameters(enabled=True))
    assert records["enabled"].value is True
    assert records["corner_loss_ev"].value == 1
    assert all(record.provenance == "F" for record in records.values())


@pytest.mark.parametrize(
    "kwargs", [{"enabled": 1}, {"corner_loss_ev": -1}, {"corner_loss_ev": float("inf")}, {"power": 0}, {"power": True}]
)
def test_vignetting_rejects_invalid_parameters_before_writing(kwargs):
    from synthraw.optics import OpticalVignettingParameters

    source = np.ones((5, 5, 3))
    out = np.full_like(source, -10)
    with pytest.raises(ValueError):
        LensSimulator().vignette_radiance(source, OpticalVignettingParameters(**kwargs), out=out)
    np.testing.assert_array_equal(out, -10)


@pytest.mark.parametrize("rows", [1, 4, 30])
def test_chromatic_channel_displacement_matches_ramps(rows):
    from synthraw.optics import OpticalChromaticParameters

    y, x = np.mgrid[:11, :13]
    source = np.stack([x + 2 * y, x - y, 2 * x + y], axis=-1).astype(float)
    p = OpticalChromaticParameters(enabled=True, red_scale=1.1, blue_scale=0.9)
    actual = LensSimulator().chromatic_radiance(source, p, strip_rows=rows)
    expected = source.copy()
    for channel, scale in ((0, 1.1), (2, 0.9)):
        xx, yy = 6 + (x - 6) * scale, 5 + (y - 5) * scale
        expected[:, :, channel] = xx + 2 * yy if channel == 0 else 2 * xx + yy
    np.testing.assert_allclose(actual[2:-2, 2:-2], expected[2:-2, 2:-2], atol=1e-12)
    np.testing.assert_array_equal(actual[:, :, 1], source[:, :, 1])
    np.testing.assert_array_equal(actual[5, 6], source[5, 6])


@pytest.mark.parametrize("shape", [(1, 1, 3), (1, 9, 3), (7, 1, 3), (12, 14, 3)])
def test_chromatic_identity_and_strip_memmap_equivalence(shape, tmp_path):
    from synthraw.optics import OpticalChromaticParameters

    source = np.random.default_rng(9).normal(size=shape).astype(np.float32) * 10
    simulator = LensSimulator()
    identity = simulator.chromatic_radiance(source)
    np.testing.assert_array_equal(identity, source)
    assert not np.shares_memory(identity, source)
    np.testing.assert_array_equal(simulator.chromatic_radiance(source, OpticalChromaticParameters(True, 1, 1)), source)
    p = OpticalChromaticParameters(True, 1.1, 0.9)
    expected = simulator.chromatic_radiance(source, p, strip_rows=100)
    out = np.memmap(tmp_path / "chromatic.bin", mode="w+", shape=shape, dtype=source.dtype)
    assert simulator.chromatic_radiance(source, p, strip_rows=1, out=out) is out
    np.testing.assert_array_equal(out, expected)


def test_chromatic_uniform_channels_and_fabricated_records():
    from synthraw.optics import OpticalChromaticParameters

    source = np.broadcast_to([-2.0, 0.0, 8.0], (13, 15, 3)).copy()
    p = OpticalChromaticParameters(enabled=True)
    np.testing.assert_allclose(LensSimulator().chromatic_radiance(source, p), source)
    records = LensSimulator().chromatic_records(p)
    assert records["enabled"].value is True
    assert records["red_scale"].value == 1.001
    assert all(record.provenance == "F" for record in records.values())


@pytest.mark.parametrize(
    "kwargs", [{"enabled": 1}, {"red_scale": 0}, {"blue_scale": -1}, {"red_scale": float("nan")}, {"blue_scale": True}]
)
def test_chromatic_rejects_invalid_parameters_before_writing(kwargs):
    from synthraw.optics import OpticalChromaticParameters

    source = np.ones((5, 5, 3))
    out = np.full_like(source, -10)
    with pytest.raises(ValueError):
        LensSimulator().chromatic_radiance(source, OpticalChromaticParameters(**kwargs), out=out)
    np.testing.assert_array_equal(out, -10)
