import numpy as np
import pytest

from synthetic_raw_lab import baselines
from synthetic_raw_lab.ingestion import pixel_sha256


def step_image(height=20, width=20) -> np.ndarray:
    image = np.full((height, width, 3), 0.2, dtype=np.float32)
    image[:, width // 2 :] = 0.8
    return image


@pytest.mark.parametrize("method", ["lanczos", "bicubic"])
def test_output_size_and_type(method):
    out = baselines.resample(step_image(), 49, 37, method)
    assert out.shape == (37, 49, 3) and out.dtype == np.float32


@pytest.mark.parametrize("method", ["lanczos", "bicubic"])
def test_constant_image_stays_constant(method):
    flat = np.full((16, 24, 3), 0.37, dtype=np.float32)
    assert np.allclose(baselines.resample(flat, 59, 39, method), 0.37, atol=1e-6)


@pytest.mark.parametrize("method", ["lanczos", "bicubic"])
def test_resampling_is_deterministic(method):
    source = np.random.default_rng(0).random((30, 40, 3), dtype=np.float32)
    first = baselines.resample(source, 98, 74, method)
    second = baselines.resample(source, 98, 74, method)
    assert pixel_sha256(first) == pixel_sha256(second)


def test_kernels_differ_and_keep_overshoot():
    lanczos = baselines.resample(step_image(), 80, 80, "lanczos")
    bicubic = baselines.resample(step_image(), 80, 80, "bicubic")
    assert not np.array_equal(lanczos, bicubic)
    # Both kernels ring at a step; float output keeps values beyond the input range.
    assert lanczos.max() > 0.8 and lanczos.min() < 0.2
    assert bicubic.max() > 0.8 and bicubic.min() < 0.2


def test_area_downsample_inverts_integer_upscale_of_flat_regions():
    image = step_image(8, 8)
    up = np.repeat(np.repeat(image, 3, axis=0), 3, axis=1)
    assert np.allclose(baselines.downsample_area(up, 8, 8), image, atol=1e-6)


def test_rejects_unknown_method_and_bad_input():
    with pytest.raises(ValueError):
        baselines.resample(step_image(), 10, 10, "nearest")
    with pytest.raises(ValueError):
        baselines.resample(step_image().astype(np.float64), 10, 10, "lanczos")
