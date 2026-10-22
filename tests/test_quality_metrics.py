import numpy as np
import pytest
from skimage.metrics import structural_similarity

from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import quality_metrics as qm


def textured(height=96, width=128, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = 0.5 + 0.3 * np.sin(x / 5.0)[..., None] * np.array([1.0, 0.8, 0.6])
    base[:, width // 2 :] *= 0.5
    return np.clip(base + rng.normal(0, 0.02, size=(height, width, 3)), 0, 1).astype(np.float32)


def test_identical_images_are_perfectly_consistent():
    linear = cm.srgb_to_linear(textured())
    metrics, delta_e = qm.source_consistency(linear, linear.copy())
    assert metrics["psnr_db"] == float("inf")
    assert metrics["ssim"] == pytest.approx(1.0)
    assert metrics["ms_ssim"]["value"] == pytest.approx(1.0)
    assert metrics["edge_displacement"]["mean_px"] == 0
    assert metrics["color_difference_ciede2000"]["max"] == 0
    assert metrics["luminance_distribution"]["wasserstein_1"] == 0
    assert metrics["lpips"] is None and "weights" in metrics["lpips_note"]
    assert delta_e.shape == linear.shape[:2]


def test_psnr_matches_definition():
    a = textured()
    b = np.clip(a + np.random.default_rng(1).normal(0, 0.01, a.shape).astype(np.float32), 0, 1)
    mse = np.mean((a.astype(np.float64) - b) ** 2)
    assert qm.psnr(a, b) == pytest.approx(10 * np.log10(1 / mse))


def test_ssim_matches_scikit_image():
    a = textured()
    b = np.clip(a * 0.9 + 0.05, 0, 1)
    reference = structural_similarity(
        a, b, channel_axis=2, data_range=1.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False
    )
    assert qm.ssim(a, b) == pytest.approx(reference, abs=1e-4)


def test_edge_displacement_measures_shift():
    image = np.zeros((64, 64), dtype=np.float32)
    image[:, 20:] = 1.0
    shifted = np.roll(image, 3, axis=1)
    shifted[:, :3] = 0
    result = qm.edge_displacement(image, shifted)
    assert result["mean_px"] == pytest.approx(3, abs=0.5)
    assert qm.edge_displacement(image, np.zeros_like(image))["mean_px"] is None


def test_brightness_change_shows_in_colour_and_lightness():
    linear = cm.srgb_to_linear(textured())
    metrics, _ = qm.source_consistency(linear, linear * 1.2)
    assert metrics["luminance_distribution"]["mean_shift"] > 1
    assert metrics["color_difference_ciede2000"]["mean"] > 1


def test_ms_ssim_degrades_with_blur_and_handles_small_images():
    from scipy.ndimage import gaussian_filter

    y = qm.luma(textured(256, 256))
    blurred = gaussian_filter(y, 2)
    full = qm.ms_ssim(y, blurred)
    assert full["levels"] == 5 and 0 < full["value"] < 1
    small = qm.ms_ssim(y[:40, :40], y[:40, :40])
    assert small["levels"] == 2 and small["value"] == pytest.approx(1.0)
    assert qm.ms_ssim(y[:8, :8], y[:8, :8]) == {"value": None, "levels": 0}


def test_image_statistics_match_numpy():
    linear = np.random.default_rng(3).normal(0.5, 0.4, size=(70, 50, 3)).astype(np.float32)
    stats = qm.image_statistics(linear, chunk_rows=16)
    assert stats["mean"]["G"] == pytest.approx(float(linear[..., 1].mean()), abs=1e-5)
    assert stats["std"]["B"] == pytest.approx(float(linear[..., 2].std()), abs=1e-5)
    assert stats["min"]["R"] == pytest.approx(float(linear[..., 0].min()), abs=1e-6)
    assert stats["below_zero_fraction"] == pytest.approx(np.mean(linear < 0))
    assert stats["above_one_fraction"] == pytest.approx(np.mean(linear > 1))


def test_high_frequency_energy_drops_with_blur():
    from scipy.ndimage import gaussian_filter

    image = textured()
    assert qm.high_frequency_energy(gaussian_filter(image, (2, 2, 0))) < qm.high_frequency_energy(image)


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        qm.source_consistency(np.zeros((4, 4, 3), np.float32), np.zeros((4, 5, 3), np.float32))
