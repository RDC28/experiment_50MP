import numpy as np
import pytest

from synthetic_raw_lab import color_management as cm
from synthetic_raw_lab import quality_metrics, test_images
from synthetic_raw_lab import sensor_simulation as ss

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
LETTERS = {"RGGB": "RGGB", "BGGR": "BGGR", "GRBG": "GRBG", "GBRG": "GBRG"}


def plain_model(**overrides) -> ss.SensorModel:
    """Identity colour, unit white balance and full exposure, so the signal equals the input channel."""
    values = {
        "color_matrix": IDENTITY,
        "white_balance_multipliers": (1.0, 1.0, 1.0),
        "exposure_scale": 1.0,
        "adc_bit_depth": 16,
    }
    return ss.SensorModel(**{**values, **overrides})


@pytest.mark.parametrize("pattern", list(ss.CFA_PATTERNS))
def test_cfa_sampling_takes_the_right_channel_at_every_phase(pattern):
    image = np.empty((6, 8, 3), dtype=np.float32)
    image[..., 0], image[..., 1], image[..., 2] = 0.1, 0.4, 0.7
    signal = ss.sample_cfa(image, plain_model(cfa_pattern=pattern))
    for y in range(6):
        for x in range(8):
            channel = "RGB".index(LETTERS[pattern][2 * (y % 2) + x % 2])
            assert signal[y, x] == pytest.approx(image[y, x, channel])


@pytest.mark.parametrize("pattern", list(ss.CFA_PATTERNS))
def test_cfa_masks_partition_the_sensor(pattern):
    masks = ss.cfa_masks(4, 6, pattern)
    assert masks.sum(axis=0).tolist() == np.ones((4, 6), dtype=int).tolist()
    assert [int(m.sum()) for m in masks] == [6, 12, 6]
    assert "RGB"[int(np.argmax(masks[:, 0, 0]))] == pattern[0]
    assert "RGB"[int(np.argmax(masks[:, 1, 1]))] == pattern[3]


def test_assumed_camera_transform_and_exposure():
    model = ss.SensorModel(exposure_scale=0.5)
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 1, (4, 4, 3)).astype(np.float32)
    signal = ss.sample_cfa(image, model)
    expected = image @ model.camera_from_working.T * 0.5
    sites = ss.cfa_masks(4, 4, model.cfa_pattern)
    for channel in range(3):
        np.testing.assert_allclose(signal[sites[channel]], expected[..., channel][sites[channel]], rtol=1e-5)
    # A working-space white lands on 1 / white-balance multiplier, scaled by the exposure.
    white = ss.sample_cfa(np.ones((2, 2, 3), dtype=np.float32), model)
    np.testing.assert_allclose(white, [[0.25, 0.5], [0.5, 0.3125]], rtol=1e-5)


def test_dng_color_matrix_is_consistent_with_as_shot_neutral():
    model = ss.SensorModel()
    neutral = model.xyz_to_camera() @ cm.xy_to_xyz(cm.D65_XY)
    np.testing.assert_allclose(neutral, model.as_shot_neutral(), atol=1e-9)
    assert model.as_shot_neutral().max() == pytest.approx(1.0)


@pytest.mark.parametrize("bits", ss.ADC_BIT_DEPTHS)
def test_quantization_levels_per_bit_depth(bits):
    model = plain_model(adc_bit_depth=bits)
    span = model.white - model.black
    signal = np.linspace(0, 1, 4 * (span + 1), dtype=np.float64).astype(np.float32).reshape(4, -1)
    dn, stats = ss.quantize(signal, model)
    assert dn.dtype == np.uint16
    assert model.white == 2**bits - 1 and model.black == 2 ** (bits - 5)
    assert np.unique(dn).size == span + 1
    assert dn.min() == model.black and dn.max() == model.white
    assert stats["clipped_at_white_fraction"] > 0 and stats["below_black_fraction"] == 0


def test_quantization_clips_out_of_range_signal():
    model = plain_model(adc_bit_depth=12)
    dn, stats = ss.quantize(np.array([[-0.5, 0.0, 1.0, 1.7]], dtype=np.float32), model)
    assert dn.tolist() == [[model.black, model.black, model.white, model.white]]
    assert stats["negative_signal_fraction"] == 0.25 and stats["saturated_signal_fraction"] == 0.5


@pytest.mark.parametrize(
    ("shot", "read", "signal", "gain"),
    [(True, False, 0.25, 1.0), (False, True, 0.25, 1.0), (True, True, 0.4, 4.0), (False, True, 0.0, 1.0)],
)
def test_noise_statistics_match_the_model(shot, read, signal, gain):
    model = ss.SensorModel(shot_noise_enabled=shot, read_noise_enabled=read, read_noise_std=6.0, analog_gain=gain)
    samples = np.full((512, 512), signal, dtype=np.float32)
    dn, _ = ss.quantize(samples, model, seed=7)
    mean, variance = ss.expected_code_statistics(signal, model)
    values = dn.astype(np.float64)
    assert values.mean() == pytest.approx(mean, abs=4 * np.sqrt(variance / values.size) + 0.01)
    assert values.var() == pytest.approx(variance, rel=0.03)


def test_noise_is_reproducible_and_seeded():
    model = ss.SensorModel(shot_noise_enabled=True, read_noise_enabled=True)
    signal = np.full((64, 64), 0.3, dtype=np.float32)
    first, _ = ss.quantize(signal, model, seed=1)
    again, _ = ss.quantize(signal, model, seed=1)
    other, _ = ss.quantize(signal, model, seed=2)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)
    quiet, _ = ss.quantize(signal, ss.SensorModel(), seed=1)
    assert np.unique(quiet).size == 1


@pytest.mark.parametrize("method", ss.DEMOSAIC_METHODS)
@pytest.mark.parametrize("pattern", list(ss.CFA_PATTERNS))
def test_round_trip_of_a_constant_colour_is_exact_to_quantization(method, pattern):
    model = ss.SensorModel(cfa_pattern=pattern, adc_bit_depth=16)
    image = np.ones((32, 48, 3), dtype=np.float32) * np.array([0.2, 0.5, 0.3], dtype=np.float32)
    developed = ss.develop(ss.simulate(image, model).dn, model, method)
    step = 1 / ((model.white - model.black) * model.exposure_scale)
    assert np.abs(developed - image).max() < 4 * step


@pytest.mark.parametrize("method", ss.DEMOSAIC_METHODS)
def test_demosaic_reproduces_linear_ramps_away_from_the_border(method):
    y, x = np.mgrid[0:40, 0:60].astype(np.float32)
    image = np.stack([0.002 * x + 0.1, 0.003 * y + 0.2, 0.001 * (x + y) + 0.05], axis=-1)
    for pattern in ss.CFA_PATTERNS:
        signal = ss.sample_cfa(image, plain_model(cfa_pattern=pattern))
        rgb = ss.demosaic(signal, pattern, method)
        np.testing.assert_allclose(rgb[3:-3, 3:-3], image[3:-3, 3:-3], atol=2e-6)


def test_round_trip_error_bound_on_a_smooth_synthetic_image():
    y, x = np.mgrid[0:96, 0:128].astype(np.float32)
    image = np.stack(
        [
            0.4 + 0.3 * np.sin(x / 9.0) * np.cos(y / 11.0),
            0.35 + 0.25 * np.cos(x / 13.0 + y / 17.0),
            0.3 + 0.2 * np.sin((x + 2 * y) / 15.0),
        ],
        axis=-1,
    ).astype(np.float32)
    model = ss.SensorModel()
    for method, bound_db in (("bilinear", 38.0), ("malvar", 50.0)):
        developed = ss.develop(ss.simulate(image, model).dn, model, method)
        metrics, _ = quality_metrics.source_consistency(image, developed)
        assert metrics["psnr_db"] > bound_db, method
        assert metrics["color_difference_ciede2000"]["mean"] < 0.6, method


def test_malvar_beats_bilinear_on_detailed_content():
    image = cm.srgb_to_linear(test_images.make_test_image(192, 108).astype(np.float32) / 255)
    model = ss.SensorModel(adc_bit_depth=16)
    dn = ss.simulate(image, model).dn
    scores = {m: quality_metrics.psnr(image, np.clip(ss.develop(dn, model, m), 0, 1)) for m in ss.DEMOSAIC_METHODS}
    assert scores["malvar"] > scores["bilinear"]


def test_demosaic_blocks_do_not_change_the_result(monkeypatch):
    rng = np.random.default_rng(3)
    native = rng.uniform(0, 1, (70, 30)).astype(np.float32)
    whole = ss.demosaic(native, "GRBG", "malvar")
    monkeypatch.setattr(ss, "_DEMOSAIC_BLOCK_ROWS", 8)
    assert np.array_equal(ss.demosaic(native, "GRBG", "malvar"), whole)


def test_mosaic_carries_the_synthetic_label_and_provenance():
    mosaic = ss.simulate(np.full((4, 4, 3), 0.5, dtype=np.float32), ss.SensorModel(shot_noise_enabled=True))
    record = mosaic.provenance()
    assert "SYNTHETIC" in mosaic.label and "Not camera RAW" in mosaic.label
    assert record["synthetic_raw_label"] == mosaic.label
    assert set(record["information"]) == {"preserved", "estimated", "synthesized", "simulated", "export_only"}
    assert "shot noise" in record["information"]["simulated"]
    assert record["classification"].startswith("Simulation")
    assert "hypothetical" in record["sensor"]["sensor_model"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"synthetic_raw_label": "camera raw"},
        {"color_matrix": ((1.0, 0.1, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))},
        {"black_level": 20000},
        {"white_level": 70000},
        {"adc_bit_depth": 13},
        {"cfa_pattern": "RGBG"},
        {"exposure_scale": 0.0},
        {"white_balance_multipliers": (1.0, 1.0)},
    ],
)
def test_invalid_sensor_parameters_are_rejected(overrides):
    with pytest.raises(ValueError):
        ss.SensorModel(**overrides)
