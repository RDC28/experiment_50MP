"""Tiny independent coordinate and interpolation evidence for synthetic optics."""

import json
import math

import numpy as np
import pytest

from prd_v5.optics import Distortion


def reference(source, model):
    height, width = source.shape[:2]
    output = np.empty_like(source)
    outside = 0
    scale = math.hypot(width, height) / 2
    for row in range(height):
        for col in range(width):
            x, y = (col + 0.5 - width / 2) / scale, (row + 0.5 - height / 2) / scale
            r = x * x + y * y
            u = x * (1 + model.k1 * r + model.k2 * r * r) + 2 * model.p1 * x * y + model.p2 * (r + 2 * x * x)
            v = y * (1 + model.k1 * r + model.k2 * r * r) + model.p1 * (r + 2 * y * y) + 2 * model.p2 * x * y
            u, v = u * scale + width / 2 - 0.5, v * scale + height / 2 - 0.5
            outside += int(u < 0 or u > width - 1 or v < 0 or v > height - 1)
            u, v = min(max(u, 0), width - 1), min(max(v, 0), height - 1)
            left, top = math.floor(u), math.floor(v)
            pixel = np.zeros(3)
            for yy, wy in ((top, 1 - (v - top)), (min(top + 1, height - 1), v - top)):
                for xx, wx in ((left, 1 - (u - left)), (min(left + 1, width - 1), u - left)):
                    pixel += source[yy, xx].astype(float) * wx * wy
            output[row, col] = pixel
    return output, outside


@pytest.mark.parametrize("model", [Distortion(k1=0.2), Distortion(k2=-0.2), Distortion(p1=0.1, p2=-0.15)])
def test_independent_map_interpolation_persistence_and_strips(tmp_path, model):
    source = np.memmap(tmp_path / "source", mode="w+", shape=(5, 7, 3), dtype=np.float32)
    source[:] = np.random.default_rng(12).uniform(0, 8, source.shape)
    expected, outside = reference(source, model)
    dest = np.memmap(tmp_path / "dest", mode="w+", shape=source.shape, dtype=np.float32)
    reports = []
    for rows in (1, 3, 20):
        reports.append(model.process(source, dest, strip_rows=rows, accept_experimental=True))
        np.testing.assert_allclose(dest, expected, rtol=1e-6, atol=1e-6)
    assert reports[0] == reports[1] == reports[2]
    assert reports[0]["outside_source_pixels"] == outside
    assert reports[0]["changed_pixels"] > 0
    assert dest.max() > 1
    np.testing.assert_array_equal(dest[2, 3], source[2, 3])
    persisted = np.memmap(tmp_path / "dest", mode="r", shape=source.shape, dtype=np.float32)
    np.testing.assert_array_equal(dest, persisted)
    json.dumps(reports[0], allow_nan=False)


@pytest.mark.parametrize("shape", [(4, 6, 3), (1, 1, 3), (1, 5, 3), (5, 1, 3)])
def test_zero_identity_and_degenerate_dimensions(tmp_path, shape):
    source = np.random.default_rng(3).uniform(0, 5, shape).astype(np.float32)
    dest = np.memmap(tmp_path / "out", mode="w+", shape=shape, dtype=np.float32)
    report = Distortion().process(source, dest)
    np.testing.assert_array_equal(dest, source)
    assert report["status"] == "zero_effect"
    assert report["changed_pixels"] == report["outside_source_pixels"] == 0
    model = Distortion(k1=0.1, p2=0.1)
    model.process(source, dest, accept_experimental=True)
    np.testing.assert_allclose(dest, reference(source, model)[0], atol=1e-6)


@pytest.mark.parametrize("name", ["k1", "k2", "p1", "p2"])
@pytest.mark.parametrize("value", [True, -0.251, 0.251, float("nan"), float("inf"), "0"])
def test_invalid_coefficients(name, value):
    with pytest.raises(ValueError, match=name):
        Distortion(**{name: value})


def test_prewrite_validation(tmp_path):
    source = np.ones((3, 4, 3), dtype=np.float32)
    dest = np.memmap(tmp_path / "out", mode="w+", shape=source.shape, dtype=np.float32)
    dest[:] = -5
    for kwargs in ({}, {"accept_experimental": 1}, {"accept_experimental": True, "strip_rows": 0}):
        with pytest.raises(ValueError):
            Distortion(k1=0.1).process(source, dest, **kwargs)
        assert np.all(dest == -5)
    with pytest.raises(ValueError, match="separate"):
        Distortion().process(dest, dest)
    with pytest.raises(ValueError, match="memmap"):
        Distortion().process(source, np.empty_like(source))
    for value in (-1, np.nan, np.inf):
        source[-1, -1, -1] = value
        with pytest.raises(ValueError, match="nonnegative"):
            Distortion().process(source, dest, strip_rows=1)
        assert np.all(dest == -5)


def test_pipeline_distortion_replay_roundtrip_and_strip_invariance(tmp_path):
    from dataclasses import replace

    import rawpy
    from PIL import Image

    from prd_v5.config import AnalyticRunConfig
    from prd_v5.hypothesis import CameraHypothesis

    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, 10:20] = 180
    source = tmp_path / "source.png"
    Image.fromarray(image).save(source)
    base = AnalyticRunConfig(
        camera=CameraHypothesis(shot_noise=False, read_noise=False),
        target_megapixels=0.001024,
        assume_unprofiled_srgb=True,
    )
    active = replace(base, distortion=Distortion(k1=0.2, p1=0.1), accept_experimental_optics=True)
    captures = []
    for i, config in enumerate((base, replace(active, strip_rows=1), replace(active, strip_rows=7))):
        restored = AnalyticRunConfig.from_json(config.to_json())
        assert restored == config
        output = tmp_path / str(i)
        report = restored.run(source, output)
        with rawpy.imread(str(output / "capture.dng")) as raw:
            captures.append(raw.raw_image.copy())
            assert raw.raw_image.dtype == np.uint16
        stage = report["stages"]["distortion"]
        assert stage["status"] == ("experimental" if i else "zero_effect")
        assert stage["changed_pixels"] > 0 if i else stage["changed_pixels"] == 0
        assert report["stages"]["roundtrip"]["tiff_roundtrip"].startswith("passed")
        assert report["registry"]["optics.distortion.k1"]["value"] == config.distortion.k1
        assert AnalyticRunConfig.from_json((output / "config.json").read_text()) == config
        if i:
            assert "brown_conrady_backward" in (output / "report.html").read_text()
    assert np.count_nonzero(captures[0] != captures[1]) > 32
    np.testing.assert_array_equal(captures[1], captures[2])


def test_distortion_config_acceptance_and_types():
    from prd_v5.config import AnalyticRunConfig

    with pytest.raises(ValueError, match="acceptance"):
        AnalyticRunConfig(distortion=Distortion(k1=0.1))
    with pytest.raises(ValueError, match="Distortion"):
        AnalyticRunConfig(distortion={})
