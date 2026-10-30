"""Tiny ISO-transition tests for an explicitly supplied electronics profile."""

import json
from dataclasses import replace

import numpy as np
import pytest

from prd_v5.config import AnalyticRunConfig
from prd_v5.electronics import DualConversionProfile
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.tests.test_sensor_effects import simulate

PROFILE = DualConversionProfile("synthetic test, not Sony", 800, 52000, 4, 26000, 1)


def test_iso_boundary_pixels_and_noise_are_coupled(tmp_path):
    clean = []
    for iso, regime, capacity, read in ((799, "low", 52000, 4), (800, "high", 26000, 1), (801, "high", 26000, 1)):
        base = CameraHypothesis(iso=iso)
        camera, records, report = PROFILE.resolve(base, accept_experimental=True)
        assert report["selected_regime"] == regime
        assert report["charge_capacity_e"] == capacity
        assert camera.full_well_e == 52000
        assert camera.read_noise_e == read
        assert camera.saturation_e == pytest.approx(52000 * 100 / iso)
        a = iso / (100 * 52000)
        b = (read * a) ** 2 + 1 / (12 * 15871**2)
        np.testing.assert_allclose(camera.noise_profile, (a, b) * 3, rtol=1e-14)
        assert camera.stream_seed("sensor_read") == base.stream_seed("sensor_read")
        assert {r.name for r in records if r.mode == "USER"} == {"full_well_e", "read_noise_e"}
        json.dumps(report, allow_nan=False)
        # Independently reconstruct actual sampled sensor codes, including read-noise discontinuity.
        codes, _ = simulate(tmp_path, AnalyticRunConfig().sensor_effects, signal=0.01, camera=camera)
        q = np.random.default_rng(np.random.SeedSequence([camera.stream_seed("sensor_shot"), 0])).poisson(
            np.full(32, 0.01 * 52000 * 100 / iso)
        )
        q = q + np.random.default_rng(np.random.SeedSequence([camera.stream_seed("sensor_read"), 0])).normal(
            0, read, 32
        )
        expected = np.clip(np.rint(q * (iso / 100) / (52000 / 15871) + 512), 0, 16383)
        np.testing.assert_array_equal(codes[0], expected)
        quiet = replace(camera, shot_noise=False, read_noise=False)
        clean.append(simulate(tmp_path, AnalyticRunConfig().sensor_effects, signal=0.01, camera=quiet)[0])
    # Switching regimes never changes photon counts or the noiseless raw signal.
    np.testing.assert_array_equal(clean[0], clean[1])
    np.testing.assert_array_equal(clean[1], clean[2])


def test_selected_profile_roundtrip_and_replay(tmp_path):
    import rawpy
    import tifffile
    from PIL import Image

    camera, records, _ = PROFILE.resolve(CameraHypothesis(iso=800), accept_experimental=True)
    config = AnalyticRunConfig(camera=camera, target_megapixels=0.001024, assume_unprofiled_srgb=True)
    assert AnalyticRunConfig.from_json(config.to_json()) == config
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), (32, 32, 32)).save(source)
    output = tmp_path / "bundle"
    config.run(source, output, camera_selections=records)
    with tifffile.TiffFile(next(output.glob("*.dng"))) as handle:
        values = np.asarray(handle.pages[0].tags[51041].value).reshape(-1, 2)
        np.testing.assert_allclose(values.ravel(), camera.noise_profile)
    with rawpy.imread(str(next(output.glob("*.dng")))) as raw:
        assert raw.white_level == 16383
        assert raw.raw_image.dtype == np.uint16
    manifest = json.loads((output / "manifest.json").read_text())
    assert "synthetic test, not Sony" in json.dumps(manifest)
    assert "selected_regime" in (output / "report.html").read_text()


def test_explicit_acceptance_and_zero_effect():
    with pytest.raises(ValueError, match="acceptance"):
        PROFILE.resolve(CameraHypothesis())
    profile = replace(PROFILE, low_read_noise_e=3)
    camera, _, report = profile.resolve(CameraHypothesis(), accept_experimental=True)
    assert camera == CameraHypothesis()
    assert report["zero_effect"] is True


@pytest.mark.parametrize(
    "values",
    [
        {"switch_iso": True},
        {"switch_iso": 100},
        {"switch_iso": 32001},
        {"label": " "},
        {"low_full_well_e": 0},
        {"high_read_noise_e": -1},
        {"high_full_well_e": float("nan")},
        {"high_full_well_e": 6000},
    ],
)
def test_invalid_profiles(values):
    with pytest.raises(ValueError):
        replace(PROFILE, **values)
