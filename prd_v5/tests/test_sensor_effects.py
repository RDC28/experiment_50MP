"""Small CFA-domain physical checks, without model inference or full frames."""

import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from prd_v5.config import AnalyticRunConfig
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.paired import run_analytic_pair
from prd_v5.sensor import BayerSensor, SensorEffects


def simulate(tmp_path, effects, signal=0.2, camera=None, strip_rows=3):
    camera = camera or CameraHypothesis(shot_noise=False, read_noise=False)
    rgb = np.full((24, 32, 3), signal, dtype=np.float64)
    dest = np.memmap(tmp_path / "adc", mode="w+", dtype=np.uint16, shape=rgb.shape[:2])
    try:
        report = BayerSensor(camera, effects).process(rgb, dest, strip_rows=strip_rows)
        return dest.copy(), report
    finally:
        dest._mmap.close()


def test_zero_preserves_baseline_and_replay(tmp_path):
    effects = SensorEffects()
    a, report = simulate(tmp_path, effects)
    np.testing.assert_array_equal(a, round(0.2 * 15871 + 512))
    assert all(not v["effective"] for v in report["body_effects"]["states"].values())
    config = AnalyticRunConfig(sensor_effects=effects)
    assert AnalyticRunConfig.from_json(config.to_json()) == config


def test_dark_temperature_exposure_arithmetic(tmp_path):
    effects = SensorEffects(dark_current_e_per_s=100, temperature_c=26)
    camera = CameraHypothesis(shutter_s=2, shot_noise=False, read_noise=False)
    a, report = simulate(tmp_path, effects, signal=0, camera=camera)
    np.testing.assert_array_equal(a, round(512 + 400 / camera.conversion_gain_e_per_dn))
    assert report["body_effects"]["dark_expectation_e"] == 400


@pytest.mark.parametrize("parameter", ["prnu_sigma", "dsnu_e", "row_fpn_e", "column_fpn_e"])
def test_body_structure_and_strip_invariance(tmp_path, parameter):
    effects = SensorEffects(**{parameter: 0.03 if parameter == "prnu_sigma" else 100})
    a, _ = simulate(tmp_path, effects, strip_rows=1)
    b, _ = simulate(tmp_path, effects, strip_rows=7)
    np.testing.assert_array_equal(a, b)
    assert np.std(a) > 1
    if parameter == "row_fpn_e":
        assert np.all(a == a[:, :1])
    if parameter == "column_fpn_e":
        assert np.all(a == a[:1])
    c, _ = simulate(tmp_path, effects, camera=CameraHypothesis(seed=9, shot_noise=False, read_noise=False))
    assert not np.array_equal(a, c)


def test_prnu_precedes_poisson(tmp_path):
    camera = CameraHypothesis(shot_noise=True, read_noise=False)
    effects = SensorEffects(prnu_sigma=0.1)
    a, _ = simulate(tmp_path, effects, camera=camera)
    body_seed = camera.stream_seed("sensor_body")
    response = np.maximum(np.random.default_rng(np.random.SeedSequence([body_seed, 0, 0])).normal(1, 0.1, 32), 0)
    expected_e = 0.2 * camera.full_well_e * response
    shot = np.random.default_rng(np.random.SeedSequence([camera.stream_seed("sensor_shot"), 0])).poisson(expected_e)
    np.testing.assert_array_equal(a[0], np.rint(shot / camera.conversion_gain_e_per_dn + 512))


def test_defects_and_dead_precedence(tmp_path):
    camera = CameraHypothesis(shutter_s=2, shot_noise=False, read_noise=False)
    hot = SensorEffects(hot_pixel_fraction=1, hot_pixel_e_per_s=100)
    a, _ = simulate(tmp_path, hot, signal=0, camera=camera)
    np.testing.assert_array_equal(a, round(512 + 200 / camera.conversion_gain_e_per_dn))
    b, _ = simulate(tmp_path, replace(hot, dead_pixel_fraction=1, dark_current_e_per_s=10), camera=camera)
    np.testing.assert_array_equal(b, 512)


@pytest.mark.parametrize(
    "values",
    [
        {"prnu_sigma": -1},
        {"dead_pixel_fraction": 2},
        {"dsnu_e": True},
        {"temperature_c": float("nan")},
        {"temperature_doubling_c": 0},
        {"hot_pixel_e_per_s": 1e20},
    ],
)
def test_reject_invalid_effects(values):
    with pytest.raises(ValueError):
        SensorEffects(**values)


def test_paired_roundtrip_records_body_effect_and_adc(tmp_path):
    import rawpy

    source = tmp_path / "source.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(source)
    camera = CameraHypothesis(shutter_s=1, shot_noise=False, read_noise=False)
    effects = SensorEffects(
        dark_current_e_per_s=100, nonlinearity_compression=0.5, hot_pixel_fraction=1, hot_pixel_e_per_s=0
    )
    config = AnalyticRunConfig(
        camera=camera, sensor_effects=effects, target_megapixels=0.001024, assume_unprofiled_srgb=True
    )
    run_analytic_pair(source, source, tmp_path / "pair", config)
    for label in ("2k", "4k"):
        child = tmp_path / "pair" / label
        with rawpy.imread(str(child / "capture.dng")) as raw:
            np.testing.assert_array_equal(
                raw.raw_image, round(512 + 100 / (1 + 0.5 * 100 / camera.full_well_e) / camera.conversion_gain_e_per_dn)
            )
        manifest = json.loads((child / "manifest.json").read_text())
        assert manifest["stages"]["sensor"]["body_effects"]["dark_expectation_e"] == 100
        occupancy = manifest["stages"]["sensor"]["body_effects"]["defect_occupancy"]
        assert occupancy["hot_selected"] == occupancy["photosites"] == 1024
        assert occupancy["dead_selected"] == occupancy["hot_and_dead"] == 0
        assert manifest["stages"]["roundtrip"]["tiff_roundtrip"].startswith("passed")
        replay = AnalyticRunConfig.from_json((child / "config.json").read_text())
        assert replay.sensor_effects == effects
        assert manifest["registry"]["sensor.dark_current_e_per_s"]["provenance"] == "SYNTHETIC"


@pytest.mark.parametrize("coefficient", [0.25, 1.0])
def test_charge_compression_arithmetic_and_strip_invariance(tmp_path, coefficient):
    effects = SensorEffects(nonlinearity_compression=coefficient)
    a, report = simulate(tmp_path, effects, signal=0.8, strip_rows=1)
    b, _ = simulate(tmp_path, effects, signal=0.8, strip_rows=7)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, round(512 + 15871 * 0.8 / (1 + coefficient * 0.8)))
    assert report["body_effects"]["states"]["nonlinearity_compression"]["effective"]
    assert "nonlinear" in report["body_effects"]["noise_profile_limit"]
    config = AnalyticRunConfig(sensor_effects=effects)
    assert AnalyticRunConfig.from_json(config.to_json()) == config


def test_charge_compression_after_shot_before_read(tmp_path):
    camera = CameraHypothesis(shot_noise=True, read_noise=True)
    a, _ = simulate(tmp_path, SensorEffects(nonlinearity_compression=0.5), camera=camera)
    shot = np.random.default_rng(np.random.SeedSequence([camera.stream_seed("sensor_shot"), 0]))
    q = shot.poisson(np.full(32, 0.2 * camera.full_well_e))
    read = np.random.default_rng(np.random.SeedSequence([camera.stream_seed("sensor_read"), 0]))
    signal = q / (1 + 0.5 * q / camera.full_well_e) + read.normal(0, camera.read_noise_e, 32)
    np.testing.assert_array_equal(a[0], np.rint(signal / camera.conversion_gain_e_per_dn + 512))


@pytest.mark.parametrize("value", [-0.1, 1.01, True, float("inf")])
def test_invalid_charge_compression(value):
    with pytest.raises(ValueError):
        SensorEffects(nonlinearity_compression=value)


@pytest.mark.parametrize("hot_fraction,dead_fraction", [(0, 0), (1, 1), (0.4, 0.3)])
def test_realized_defect_occupancy_matches_seeded_maps(tmp_path, hot_fraction, dead_fraction):
    camera = CameraHypothesis(shot_noise=False, read_noise=False)
    effects = SensorEffects(hot_pixel_fraction=hot_fraction, dead_pixel_fraction=dead_fraction)
    a, report = simulate(tmp_path, effects, camera=camera, strip_rows=1)
    b, other = simulate(tmp_path, effects, camera=camera, strip_rows=7)
    np.testing.assert_array_equal(a, b)
    occupancy = report["body_effects"]["defect_occupancy"]
    assert occupancy == other["body_effects"]["defect_occupancy"]
    seed = camera.stream_seed("sensor_body")
    hot = np.array(
        [np.random.default_rng(np.random.SeedSequence([seed, 5, row])).random(32) < hot_fraction for row in range(24)]
    )
    dead = np.array(
        [np.random.default_rng(np.random.SeedSequence([seed, 6, row])).random(32) < dead_fraction for row in range(24)]
    )
    assert occupancy["photosites"] == 24 * 32
    assert occupancy["hot_selected"] == np.count_nonzero(hot)
    assert occupancy["dead_selected"] == np.count_nonzero(dead)
    assert occupancy["hot_and_dead"] == np.count_nonzero(hot & dead)
    assert occupancy["hot_surviving_dead_override"] == np.count_nonzero(hot & ~dead)
    np.testing.assert_array_equal(a[dead], camera.black_level)


@pytest.mark.parametrize("hot_fraction,dead_fraction", [(0, 0), (1, 1), (0.4, 0.3)])
def test_disk_backed_defect_mask_matches_simulation(tmp_path, hot_fraction, dead_fraction):
    camera = CameraHypothesis(shot_noise=False, read_noise=False)
    effects = SensorEffects(hot_pixel_fraction=hot_fraction, dead_pixel_fraction=dead_fraction)
    baseline, occupancy_report = simulate(tmp_path, effects)
    source = np.full((24, 32, 3), 0.2)
    adc = np.memmap(tmp_path / "masked-adc", mode="w+", dtype=np.uint16, shape=(24, 32))
    mask = np.memmap(tmp_path / "defects", mode="w+", dtype=np.uint8, shape=(24, 32))
    try:
        for strip_rows in (1, 7):
            mask[:] = 255  # Every site must be overwritten, including zero-effect sites.
            report = BayerSensor(camera, effects).process(source, adc, strip_rows=strip_rows, defect_mask=mask)
            np.testing.assert_array_equal(adc, baseline)
            seed = camera.stream_seed("sensor_body")
            for y in range(24):
                hot = np.random.default_rng(np.random.SeedSequence([seed, 5, y])).random(32) < hot_fraction
                dead = np.random.default_rng(np.random.SeedSequence([seed, 6, y])).random(32) < dead_fraction
                np.testing.assert_array_equal(mask[y], hot.astype(np.uint8) | (dead.astype(np.uint8) << 1))
            occupancy = occupancy_report["body_effects"]["defect_occupancy"]
            assert np.count_nonzero(mask & 1) == occupancy["hot_selected"]
            assert np.count_nonzero(mask & 2) == occupancy["dead_selected"]
            assert np.count_nonzero(mask == 3) == occupancy["hot_and_dead"]
            assert report["body_effects"]["defect_mask"]["written"] is True
        persisted = np.memmap(tmp_path / "defects", mode="r", dtype=np.uint8, shape=(24, 32))
        try:
            np.testing.assert_array_equal(persisted, mask)
        finally:
            persisted._mmap.close()
        assert occupancy_report["body_effects"]["defect_mask"]["written"] is False
    finally:
        adc._mmap.close()
        mask._mmap.close()


def test_defect_mask_rejected_before_adc_writes(tmp_path):
    source = np.zeros((2, 4, 3))
    adc = np.memmap(tmp_path / "adc", mode="w+", dtype=np.uint16, shape=(2, 4))
    alias = np.memmap(tmp_path / "adc", mode="r+", dtype=np.uint8, shape=(2, 4))
    readonly = np.memmap(tmp_path / "readonly", mode="w+", dtype=np.uint8, shape=(2, 4))
    readonly.flags.writeable = False
    try:
        adc[:] = 123
        for invalid in (np.zeros((2, 4), dtype=np.uint8), adc, alias, readonly):
            with pytest.raises(ValueError, match="defect_mask"):
                BayerSensor().process(source, adc, defect_mask=invalid)
            np.testing.assert_array_equal(adc, 123)
    finally:
        alias._mmap.close()
        readonly._mmap.close()
        adc._mmap.close()
