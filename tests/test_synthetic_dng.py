import json
import shutil

import numpy as np
import pytest
import tifffile

from synthetic_raw_lab import exporters
from synthetic_raw_lab import sensor_simulation as ss
from synthetic_raw_lab import synthetic_dng as sd

METADATA = {"synthetic_raw_label": ss.SYNTHETIC_RAW_LABEL, "run_id": "test"}


def gradient(height=48, width=64) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.stack([0.1 + 0.6 * x / width, 0.2 + 0.5 * y / height, 0.6 - 0.4 * x / width], axis=-1)


def write(tmp_path, model=None, image=None, name="mosaic.dng"):
    image = gradient() if image is None else image
    mosaic = ss.simulate(image, model or ss.SensorModel(shot_noise_enabled=True, read_noise_enabled=True))
    result = sd.write_synthetic_dng(tmp_path / name, mosaic, METADATA, sd.preview_from_linear(image))
    return result, mosaic


@pytest.mark.parametrize("pattern", list(ss.CFA_PATTERNS))
def test_dng_passes_structural_and_libraw_validation(tmp_path, pattern):
    result, _ = write(tmp_path, ss.SensorModel(cfa_pattern=pattern, adc_bit_depth=12))
    assert result.valid and result.kind == "synthetic_mosaic_dng"
    for check in ("dng_version", "photometric_cfa", "cfa_pattern", "color_matrix_1", "black_level", "white_level"):
        assert result.checks[check]
    for check in ("libraw_opens", "libraw_cfa_pattern", "libraw_color_matrix", "libraw_mosaic_identical"):
        assert result.checks[check]


def test_dng_layout_and_provenance_metadata(tmp_path):
    result, mosaic = write(tmp_path)
    with tifffile.TiffFile(result.path) as tif:
        ifd0 = tif.pages[0]
        raw = ifd0.pages[0]
        assert not tif.is_bigtiff
        assert ifd0.tags[254].value == 1 and raw.tags[254].value == 0
        assert "SIMULATED" in ifd0.tags[50708].value
        assert json.loads(ifd0.description) == METADATA
        assert int(raw.photometric) == 32803
        assert np.array_equal(raw.asarray(), mosaic.dn)
    assert result.path.suffix == ".dng" and result.path.stat().st_size > mosaic.dn.nbytes


def test_validation_detects_a_mismatching_model(tmp_path):
    result, mosaic = write(tmp_path)
    other = ss.SyntheticMosaic(mosaic.dn, ss.SensorModel(black_level=600, cfa_pattern="GRBG"), mosaic.seed)
    checks = sd.structural_checks(result.path, other, METADATA, result.pixel_sha256)
    assert not checks["black_level"] and not checks["cfa_pattern"]
    assert not sd.libraw_checks(result.path, other)["libraw_cfa_pattern"]


def test_a_failed_check_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sd, "libraw_checks", lambda path, mosaic: {"libraw_opens": False})
    with pytest.raises(exporters.ExportValidationError, match="libraw_opens"):
        write(tmp_path)


def test_dng_metadata_must_be_labelled_synthetic(tmp_path):
    mosaic = ss.simulate(gradient(), ss.SensorModel())
    with pytest.raises(ValueError, match="synthetic"):
        sd.write_synthetic_dng(tmp_path / "x.dng", mosaic, {"run_id": "x"}, sd.preview_from_linear(gradient()))


def test_libraw_develops_the_dng_back_to_the_input_colour(tmp_path):
    image = np.ones((64, 96, 3), dtype=np.float32) * np.array([0.30, 0.45, 0.20], dtype=np.float32)
    result, mosaic = write(tmp_path, ss.SensorModel(), image)
    developed = sd.develop_with_libraw(result.path, mosaic)
    assert developed.shape == image.shape
    np.testing.assert_allclose(developed[8:-8, 8:-8], image[8:-8, 8:-8], atol=2e-3)


@pytest.mark.skipif(shutil.which("exiftool") is None, reason="exiftool not installed")
def test_exiftool_validates_the_dng(tmp_path):
    result, _ = write(tmp_path)
    report = sd.exiftool_validate(result.path)
    assert report["ok"], report
    assert result.checks["exiftool_validate_ok"]


def test_mosaic_tiff_keeps_the_codes_and_label(tmp_path):
    mosaic = ss.simulate(gradient(), ss.SensorModel(adc_bit_depth=10))
    result = sd.write_mosaic_tiff(tmp_path / "mosaic.tif", mosaic, METADATA)
    assert result.valid and result.kind == "synthetic_mosaic_tiff"
    with tifffile.TiffFile(result.path) as tif:
        assert np.array_equal(tif.pages[0].asarray(), mosaic.dn)
        assert ss.SYNTHETIC_RAW_LABEL in tif.pages[0].description
    with pytest.raises(ValueError, match="synthetic"):
        sd.write_mosaic_tiff(tmp_path / "bad.tif", mosaic, {"run_id": "x"})
