"""Independent tiny-array evidence for radial optical attenuation."""

import json

import numpy as np
import pytest

from prd_v5.optics import Vignetting


def test_radial_arithmetic_hdr_symmetry_and_strip_invariance(tmp_path):
    source = np.full((5, 7, 3), (2, 3, 4), dtype=np.float32)
    dest = np.memmap(tmp_path / "optics.bin", mode="w+", shape=source.shape, dtype=np.float32)
    reports = []
    for rows in (1, 3, 9):
        reports.append(Vignetting(0.8).process(source, dest, strip_rows=rows, accept_experimental=True))
        expected = np.empty_like(source)
        for y in range(5):
            for x in range(7):
                r2 = ((x - 3) ** 2 + (y - 2) ** 2) / 18.5
                expected[y, x] = source[y, x].astype(float) * (1 - 0.8 * r2)
        np.testing.assert_array_equal(dest, expected)
        np.testing.assert_array_equal(dest, dest[::-1, ::-1])
        np.testing.assert_array_equal(dest[2, 3], source[2, 3])
    assert reports[0] == reports[1] == reports[2]
    assert reports[0]["changed_pixels"] == 34
    assert dest.max() == 4  # Radiance above one is retained.
    json.dumps(reports[0], allow_nan=False)
    dest.flush()
    persisted = np.memmap(tmp_path / "optics.bin", mode="r", shape=source.shape, dtype=np.float32)
    np.testing.assert_array_equal(persisted, expected)


def test_zero_default_preserves_pixels_without_acceptance(tmp_path):
    source = np.random.default_rng(4).uniform(0, 8, (4, 6, 3)).astype(np.float32)
    dest = np.memmap(tmp_path / "out", mode="w+", shape=source.shape, dtype=np.float32)
    report = Vignetting().process(source, dest)
    np.testing.assert_array_equal(dest, source)
    assert report["status"] == "zero_effect"
    assert report["changed_pixels"] == 0
    assert report["minimum_transmission"] == 1


@pytest.mark.parametrize("strength", [True, -0.1, 1.1, float("nan"), float("inf"), "0"])
def test_invalid_strength(strength):
    with pytest.raises(ValueError):
        Vignetting(strength)


def test_prewrite_gates_and_invalid_signal(tmp_path):
    source = np.ones((2, 3, 3), dtype=np.float32)
    dest = np.memmap(tmp_path / "out", mode="w+", shape=source.shape, dtype=np.float32)
    dest[:] = -5
    for kwargs in ({}, {"accept_experimental": 1}, {"accept_experimental": True, "strip_rows": 0}):
        with pytest.raises(ValueError):
            Vignetting(0.5).process(source, dest, **kwargs)
        assert np.all(dest == -5)
    with pytest.raises(ValueError, match="separate"):
        Vignetting().process(dest, dest)
    with pytest.raises(ValueError, match="memmap"):
        Vignetting().process(source, np.empty_like(source))
    for invalid in (-1, float("nan"), float("inf")):
        source[0, 0, 0] = invalid
        with pytest.raises(ValueError, match="nonnegative"):
            Vignetting().process(source, dest)


def test_single_pixel_is_optical_center(tmp_path):
    source = np.ones((1, 1, 3), dtype=np.float32)
    dest = np.memmap(tmp_path / "out", mode="w+", shape=source.shape, dtype=np.float32)
    report = Vignetting(1).process(source, dest, accept_experimental=True)
    np.testing.assert_array_equal(dest, source)
    assert report["changed_pixels"] == 0


def test_pipeline_attenuation_replay_and_pre_cfa_arithmetic(tmp_path):
    import rawpy
    from PIL import Image

    from prd_v5.config import AnalyticRunConfig
    from prd_v5.hypothesis import CameraHypothesis

    source = tmp_path / "source.png"
    Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8)).save(source)
    base = AnalyticRunConfig(
        camera=CameraHypothesis(shot_noise=False, read_noise=False),
        target_megapixels=0.001024,
        assume_unprofiled_srgb=True,
        strip_rows=3,
    )
    from dataclasses import replace

    configs = [base, replace(base, vignetting=Vignetting(0.8), accept_experimental_optics=True)]
    pixels = []
    for i, config in enumerate(configs):
        restored = AnalyticRunConfig.from_json(config.to_json())
        assert restored == config
        output = tmp_path / str(i)
        report = restored.run(source, output)
        with rawpy.imread(str(output / "capture.dng")) as raw:
            pixels.append(raw.raw_image.copy())
        assert report["stages"]["roundtrip"]["tiff_roundtrip"].startswith("passed")
        assert report["stages"]["optics"]["status"] == ("experimental" if i else "zero_effect")
        assert "optics.vignetting.corner_strength" in report["registry"]
        assert "centered_radial_linear_camera_attenuation" in (output / "report.html").read_text()
        assert AnalyticRunConfig.from_json((output / "config.json").read_text()) == config
    # Independently bound quantization error of attenuation in the stored ADC domain.
    y, x = np.indices((32, 32))
    transmission = 1 - 0.8 * ((x - 15.5) ** 2 + (y - 15.5) ** 2) / 512
    expected = (pixels[0].astype(float) - base.camera.black_level) * transmission + base.camera.black_level
    assert np.max(np.abs(pixels[1] - expected)) <= 1
    assert pixels[1][0, 0] < pixels[0][0, 0]


def test_config_rejects_unaccepted_optics_before_output(tmp_path):
    from prd_v5.config import AnalyticRunConfig

    with pytest.raises(ValueError, match="acceptance"):
        AnalyticRunConfig(vignetting=Vignetting(0.2))
    with pytest.raises(ValueError, match="boolean"):
        AnalyticRunConfig(accept_experimental_optics=1)
    with pytest.raises(ValueError, match="Vignetting"):
        AnalyticRunConfig(vignetting={})
