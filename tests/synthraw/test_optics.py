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
