from __future__ import annotations

import numpy as np
import pytest

from prd_v6 import colour, constants, dng, pipeline

rawpy = pytest.importorskip("rawpy")
CAM = pipeline.camera_matrix()


def _layout(hw):
    return np.tile(np.array([[0, 1], [1, 2]], dtype=np.int8), (hw[0] // 2, hw[1] // 2))


def test_cfa_dng_reads_back_in_libraw(tmp_path):
    rng = np.random.default_rng(3)
    samples = rng.integers(constants.BLACK_16, constants.WHITE_16, (48, 64), dtype=np.uint16)
    neutral = colour.prior_neutral(pipeline.camera_matrix())
    path = dng.write_dng(tmp_path / "cfa.dng", samples, cfa_pattern="RGGB", color_matrix=CAM, as_shot_neutral=neutral)
    with rawpy.imread(str(path)) as raw:
        assert np.array_equal(raw.raw_image_visible, samples)
        assert raw.white_level == constants.WHITE_16
        assert list(raw.black_level_per_channel) == [constants.BLACK_16] * 4
    assert dng.libraw_check(path, expected_hw=(48, 64), cfa_pattern="RGGB")["status"] == "passed"
    tags = dng.read_dng_tags(path)
    assert tags["Make"] == "Synthetic"
    forward = np.asarray(tags["ForwardMatrix1"], dtype=np.float64).reshape(9, 2)
    np.testing.assert_allclose(forward[:, 0] / forward[:, 1], colour.forward_matrix(CAM).reshape(9), atol=1e-6)
    assert tuple(tags["BaselineExposure"]) == (0, 1)


def test_linear_raw_dng_reads_back_in_libraw(tmp_path):
    rng = np.random.default_rng(4)
    samples = rng.integers(constants.BLACK_16, constants.WHITE_16, (40, 60, 3), dtype=np.uint16)
    neutral = colour.prior_neutral(pipeline.camera_matrix())
    path = dng.write_dng(tmp_path / "linear.dng", samples, cfa_pattern=None, color_matrix=CAM, as_shot_neutral=neutral)
    with rawpy.imread(str(path)) as raw:
        assert np.array_equal(np.asarray(raw.raw_image)[..., :3], samples)
    assert dng.libraw_check(path, expected_hw=(40, 60), cfa_pattern=None)["status"] == "passed"
    assert np.array_equal(dng.read_samples(path), samples)


@pytest.mark.parametrize(
    ("pattern", "shape", "samples_per_pixel", "photometric"),
    [(None, (8, 12, 3), 3, 34892), ("RGGB", (8, 12), 1, 32803)],
)
def test_dng_has_no_extra_samples_tag(tmp_path, pattern, shape, samples_per_pixel, photometric):
    """dng_validate rejects ExtraSamples in IFD 0; tifffile adds it for 3-sample images unless told not to."""
    samples = np.full(shape, constants.BLACK_16, dtype=np.uint16)
    neutral = colour.prior_neutral(pipeline.camera_matrix())
    path = dng.write_dng(tmp_path / "t.dng", samples, cfa_pattern=pattern, color_matrix=CAM, as_shot_neutral=neutral)
    tags = dng.read_dng_tags(path)
    assert "ExtraSamples" not in tags
    assert tags["SamplesPerPixel"] == samples_per_pixel
    assert int(tags["PhotometricInterpretation"]) == photometric


def test_libraw_develops_the_forward_model_back_to_its_input(tmp_path):
    """The DNG tags and the colour transform agree: LibRaw recovers the linear values we encoded."""
    hw = (64, 96)
    linear = np.empty(hw + (3,), dtype=np.float32)
    linear[:, : hw[1] // 2] = (0.30, 0.20, 0.10)
    linear[:, hw[1] // 2 :] = (0.05, 0.25, 0.40)
    neutral = colour.prior_neutral(pipeline.camera_matrix())
    clean, _ = pipeline.cfa_samples(linear, neutral, _layout(hw), noise_seed=None)
    path = dng.write_dng(tmp_path / "flat.dng", clean, cfa_pattern="RGGB", color_matrix=CAM, as_shot_neutral=neutral)
    developed = pipeline.develop_cfa_dng(path)
    expected = colour.srgb_encode(linear)
    for cols in (slice(8, hw[1] // 2 - 8), slice(hw[1] // 2 + 8, -8)):
        assert np.abs(developed[8:-8, cols] - expected[8:-8, cols]).max() < 2 / 255


def test_tiff16_is_photoshop_readable_16_bit_with_icc(tmp_path):
    encoded = np.random.default_rng(5).uniform(0, 1, (30, 45, 3)).astype(np.float32)
    path = dng.write_tiff16(tmp_path / "out.tif", encoded, colour.srgb_icc_profile())
    check = dng.tiff16_check(path, (30, 45))
    assert check["status"] == "passed", check


def test_writer_rejects_wrong_shapes(tmp_path):
    cam = pipeline.camera_matrix()
    with pytest.raises(TypeError):
        dng.write_dng(
            tmp_path / "a.dng",
            np.zeros((4, 4), np.float32),
            cfa_pattern="RGGB",
            color_matrix=cam,
            as_shot_neutral=[1, 1, 1],
        )
    with pytest.raises(ValueError):
        dng.write_dng(
            tmp_path / "b.dng",
            np.zeros((4, 4, 3), np.uint16),
            cfa_pattern="RGGB",
            color_matrix=cam,
            as_shot_neutral=[1, 1, 1],
        )
