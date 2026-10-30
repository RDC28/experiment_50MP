"""Observable optics behavior, strip invariance and independently decoded RAW."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from prd_v5.config import AnalyticRunConfig
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.optical_effects import OpticalEffects


def process(tmp_path, source, parameters, rows):
    output = np.memmap(tmp_path / f"{rows}.bin", mode="w+", shape=source.shape, dtype=np.float32)
    try:
        report = parameters.process(source, output, aperture=5.6, pixel_pitch_um=4.16, strip_rows=rows)
        return output.copy(), report
    finally:
        output._mmap.close()


@pytest.mark.parametrize(
    "parameters",
    [
        OpticalEffects(lateral_ca_px=1),
        OpticalEffects(axial_ca_sigma_px=1),
        OpticalEffects(psf_center_sigma_px=0.4, psf_edge_sigma_px=1.5),
        OpticalEffects(diffraction=True),
        OpticalEffects(defocus_radius_px=2),
        OpticalEffects(motion_length_px=3, motion_angle_deg=30),
        OpticalEffects(lateral_ca_px=0.5, diffraction=True, motion_length_px=2),
    ],
)
def test_impulse_changes_and_is_strip_invariant(tmp_path, parameters):
    source = np.zeros((33, 35, 3), dtype=np.float32)
    source[7, 9] = 3
    a, report = process(tmp_path, source, parameters, 3)
    b, other = process(tmp_path, source, parameters, 40)
    np.testing.assert_allclose(a, b, atol=1e-7)
    assert not np.array_equal(a, source)
    assert np.isfinite(a).all() and np.min(a) >= 0
    assert report == other
    assert report["status"] == "experimental"


def test_uniform_hdr_and_disabled_identity(tmp_path):
    source = np.full((9, 11, 3), 4, dtype=np.float32)
    for parameters in (OpticalEffects(), OpticalEffects(diffraction=True, motion_length_px=4, lateral_ca_px=1)):
        actual, _ = process(tmp_path, source, parameters, 3)
        np.testing.assert_allclose(actual, source, atol=1e-6)


def test_optics_config_replays_and_changes_decodable_raw(tmp_path):
    import rawpy

    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[8:12, 8:12] = 180
    path = tmp_path / "source.png"
    Image.fromarray(pixels).save(path)
    base = AnalyticRunConfig(
        camera=CameraHypothesis(shot_noise=False, read_noise=False),
        assume_unprofiled_srgb=True,
        target_megapixels=0.001024,
    )
    advanced = replace(
        base, optical_effects=OpticalEffects(diffraction=True, motion_length_px=2), accept_experimental_optics=True
    )
    arrays = []
    for index, config in enumerate((base, advanced)):
        config = AnalyticRunConfig.from_json(config.to_json())
        report = config.run(path, tmp_path / str(index))
        assert report["stages"]["optical_effects"]["aperture"] == base.camera.aperture
        with rawpy.imread(str(tmp_path / str(index) / "capture.dng")) as raw:
            arrays.append(raw.raw_image.copy())
    assert not np.array_equal(*arrays)


@pytest.mark.parametrize(
    "values",
    [
        {"motion_length_px": 65},
        {"lateral_ca_px": float("nan")},
        {"diffraction": 1},
        {"wavelength_nm": 100},
        {"defocus_radius_px": -1},
    ],
)
def test_invalid_priors(values):
    with pytest.raises(ValueError):
        OpticalEffects(**values)
