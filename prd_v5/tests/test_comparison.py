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
