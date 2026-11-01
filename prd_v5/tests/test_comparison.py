"""Independently calculated full-raster metrics and bounded execution contracts."""

import json

import numpy as np
import pytest
import tifffile

from prd_v5.comparison import compare_renders


def write(path, array):
    tifffile.imwrite(path, array, photometric="rgb", metadata=None)
    return path


def test_full_resolution_metrics_and_strip_invariance(tmp_path):
    left = np.zeros((5, 4, 3), dtype=np.uint16)
    right = left.copy()
    left[0, 0] = [65535, 0, 0]
    right[-1, -1] = [0, 65535, 0]
    a = write(tmp_path / "a.tiff", left)
    b = write(tmp_path / "b.tiff", right)
    report = compare_renders(a, b, strip_rows=1)
    assert report == compare_renders(a, b, strip_rows=4)
    assert report["samples"] == 60
    assert report["mae"] == pytest.approx(2 / 60)
    assert report["rmse"] == pytest.approx(np.sqrt(2 / 60))
    assert report["max_absolute_error"] == 1
    assert report["channel_bias_first_minus_second"] == pytest.approx([1 / 20, -1 / 20, 0])
    assert report["psnr_db"] == pytest.approx(10 * np.log10(30))
    assert report["saturated_pixel_fraction"] == [1 / 20, 1 / 20]
    assert report["alignment"] == "none"
    json.dumps(report, allow_nan=False)


def test_identical_renders_have_json_safe_infinite_psnr(tmp_path):
    path = write(tmp_path / "a.tiff", np.zeros((2, 3, 3), dtype=np.uint16))
    report = compare_renders(path, path)
    assert report["identical"]
    assert report["psnr_db"] is None
    assert "infinite" in report["psnr_null_reason"]
    json.dumps(report, allow_nan=False)


def test_unequal_rasters_skip_without_resampling(tmp_path):
    a = write(tmp_path / "a.tiff", np.zeros((2, 3, 3), dtype=np.uint16))
    b = write(tmp_path / "b.tiff", np.zeros((4, 3, 3), dtype=np.uint16))
    report = compare_renders(a, b)
    assert report["status"] == "not measured"
    assert "Unequal" in report["reason"]
    assert "rmse" not in report


@pytest.mark.parametrize("rows", [0, -1, True, 1.5])
def test_invalid_strip_size_before_io(rows):
    with pytest.raises(ValueError, match="strip_rows"):
        compare_renders("missing", "missing", strip_rows=rows)


def test_invalid_dtype_releases_mapping(tmp_path, monkeypatch):
    a = write(tmp_path / "a.tiff", np.zeros((2, 3, 3), dtype=np.uint8))
    original = tifffile.memmap
    opened = []

    def track(*args, **kwargs):
        array = original(*args, **kwargs)
        opened.append(array)
        return array

    monkeypatch.setattr(tifffile, "memmap", track)
    with pytest.raises(ValueError, match="uint16 RGB"):
        compare_renders(a, a)
    assert opened[0]._mmap.closed


def test_native_pixel_region_metrics_exclude_outside_changes(tmp_path):
    left = np.zeros((6, 8, 3), dtype=np.uint16)
    right = left.copy()
    left[0, 0] = 65535  # Outside the requested validation crop.
    left[2, 3, 0] = 65535
    right[3, 4, 1] = 65535
    a = write(tmp_path / "a.tiff", left)
    b = write(tmp_path / "b.tiff", right)
    report = compare_renders(a, b, region=(2, 3, 4, 5), strip_rows=1)
    assert report == compare_renders(a, b, region=(2, 3, 4, 5), strip_rows=7)
    assert report["region"] == [2, 3, 4, 5]
    assert report["shapes"] == [[6, 8, 3], [6, 8, 3]]
    assert report["samples"] == 12
    assert report["mae"] == pytest.approx(2 / 12)
    assert report["rmse"] == pytest.approx(np.sqrt(2 / 12))
    assert report["channel_bias_first_minus_second"] == pytest.approx([0.25, -0.25, 0])
    assert report["saturated_pixel_fraction"] == [0.25, 0.25]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "region", [(0, 0, 0, 1), (-1, 0, 1, 1), (0, 2, 1, 1), (0, 0, True, 1), (0, 0, 1.5, 2), (0, 0, 1), [0, 0, 1, 1]]
)
def test_invalid_region_before_io(region):
    with pytest.raises(ValueError, match="region"):
        compare_renders("missing", "missing", region=region)


def test_region_on_unequal_rasters_and_bounds_cleanup(tmp_path, monkeypatch):
    a = write(tmp_path / "a.tiff", np.zeros((2, 3, 3), dtype=np.uint16))
    b = write(tmp_path / "b.tiff", np.zeros((4, 5, 3), dtype=np.uint16))
    assert compare_renders(a, b, region=(0, 0, 2, 3))["identical"]
    original = tifffile.memmap
    opened = []

    def track(*args, **kwargs):
        array = original(*args, **kwargs)
        opened.append(array)
        return array

    monkeypatch.setattr(tifffile, "memmap", track)
    with pytest.raises(ValueError, match="exceeds"):
        compare_renders(a, b, region=(0, 0, 3, 3))
    assert len(opened) == 2
    assert all(array._mmap.closed for array in opened)


def test_bayer_electron_metrics_preserve_phase_and_crop_origin(tmp_path):
    from prd_v5.comparison import compare_bayer

    a = np.zeros((7, 9), dtype=np.float32)
    b = np.arange(63, dtype=np.float32).reshape(7, 9)
    first, second = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(first, a)
    np.save(second, b)
    region = (1, 3, 6, 8)
    report = compare_bayer(first, second, cfa_pattern="RGGB", region=region, strip_rows=1)
    other = compare_bayer(first, second, cfa_pattern="RGGB", region=region, strip_rows=4)
    assert report == other
    difference = -b[1:6, 3:8].astype(float)
    assert report["samples"] == 25
    assert report["rmse"] == pytest.approx(np.sqrt(np.mean(difference**2)))
    for phase, (key, result) in enumerate(report["phases"].items()):
        expected = -b[1 + (phase // 2 - 1) % 2 : 6 : 2, 3 + (phase % 2 - 3) % 2 : 8 : 2]
        assert key.endswith("RGGB"[phase])
        assert result["bias_first_minus_second"] == pytest.approx(expected.mean())
        assert result["mae"] == pytest.approx(np.abs(expected).mean())
    json.dumps(report, allow_nan=False)


def test_bayer_unequal_shape_and_invalid_samples(tmp_path):
    from prd_v5.comparison import compare_bayer

    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(a, np.zeros((2, 2), np.float32))
    np.save(b, np.zeros((4, 4), np.float32))
    assert compare_bayer(a, b, cfa_pattern="BGGR")["status"] == "not measured"
    np.save(b, np.full((2, 2), np.nan, np.float32))
    with pytest.raises(ValueError, match="nonfinite"):
        compare_bayer(a, b, cfa_pattern="BGGR")
    np.save(b, np.zeros((2, 2), np.float64))
    with pytest.raises(ValueError, match="float32"):
        compare_bayer(a, b, cfa_pattern="BGGR")


def test_frequency_bands_separate_known_native_pixel_sinusoids(tmp_path):
    x = np.arange(128)

    def sinusoid(frequency):
        row = np.rint(32768 + 12000 * np.sin(2 * np.pi * frequency * x)).astype(np.uint16)
        return np.broadcast_to(row[None, :, None], (128, 128, 3)).copy()

    a = write(tmp_path / "low.tiff", sinusoid(1 / 32))
    b = write(tmp_path / "high.tiff", sinusoid(3 / 16))
    report = compare_renders(a, b)["frequency_bands"]
    assert report["status"] == "measured spectral disagreement"
    bands = report["bands"]
    assert np.argmax([band["first_power"] for band in bands]) == 0
    assert np.argmax([band["second_power"] for band in bands]) == 2
    expected_variance = (12000 / 65535) ** 2 / 2
    assert sum(band["first_power"] for band in bands) == pytest.approx(expected_variance, rel=0.002)
    assert sum(band["second_power"] for band in bands) == pytest.approx(expected_variance, rel=0.002)
    assert sum(band["difference_power"] for band in bands) == pytest.approx(2 * expected_variance, rel=0.002)
    assert report["measured_pixels"] == 128**2
    json.dumps(report, allow_nan=False)


def test_frequency_bounded_fft_partial_edges_and_crop(tmp_path, monkeypatch):
    rng = np.random.default_rng(17)
    image = rng.integers(0, 65536, (263, 259, 3), dtype=np.uint16)
    path = write(tmp_path / "random.tiff", image)
    original = np.fft.fft2
    shapes = []

    def bounded(array):
        shapes.append(array.shape)
        assert max(array.shape) <= 256
        return original(array)

    monkeypatch.setattr(np.fft, "fft2", bounded)
    result = compare_renders(path, path)["frequency_bands"]
    assert result["patches"] == 2
    assert result["measured_pixels"] == 263 * 256
    assert result["omitted_edge_pixels"] == 263 * 3
    assert set(shapes) == {(256, 256), (7, 256)}
    assert all(band["difference_power"] == 0 for band in result["bands"])
    crop = compare_renders(path, path, region=(1, 2, 129, 130))["frequency_bands"]
    assert crop["patches"] == 1 and crop["measured_pixels"] == 128**2


def test_frequency_dc_removed_and_tiny_crop_explicitly_unmeasured(tmp_path):
    a = write(tmp_path / "dark.tiff", np.zeros((8, 8, 3), np.uint16))
    b = write(tmp_path / "bright.tiff", np.full((8, 8, 3), 65535, np.uint16))
    result = compare_renders(a, b)
    assert result["mae"] == 1
    assert all(band["difference_power"] == 0 for band in result["frequency_bands"]["bands"])
    tiny = compare_renders(a, b, region=(0, 0, 2, 3))["frequency_bands"]
    assert tiny["status"] == "not measured"
    assert tiny["omitted_edge_pixels"] == 6
    assert all(band["first_power"] is None for band in tiny["bands"])
    json.dumps(tiny, allow_nan=False)
