"""Real tiny output bundle contract and synthetic ADC/gain validation."""

import json
from dataclasses import replace
from html.parser import HTMLParser
from importlib.resources import files

import numpy as np
import pytest
import rawpy
from PIL import Image

from prd_v5.delivery import digest
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.physics import photon_transfer
from prd_v5.presets import full_quality_config


class _TableRows(HTMLParser):
    """Collect the text cells of each rendered HTML table row."""

    def __init__(self):
        super().__init__()
        self.rows, self.cell = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("th", "td"):
            self.cell = ""

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            self.rows[-1].append(self.cell)
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data


@pytest.mark.parametrize("adc_bits", [12, 14, 16])
def test_full_quality_fixture_outputs_registry_and_decodable_adc(tmp_path, adc_bits):
    source = tmp_path / "source.png"
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[8:24, 8:24] = [90, 128, 160]
    image[0, 0] = 255
    Image.fromarray(image).save(source)
    config = replace(
        full_quality_config(), target_megapixels=0.001024, camera=CameraHypothesis(adc_bits=adc_bits, profile="GENERIC")
    )
    output = tmp_path / "run"
    report = config.run(source, output)
    expected = {
        "synthetic_50mp.dng",
        "synthetic_50mp_16bit.tif",
        "synthetic_50mp_preview.jpg",
        "clean_bayer.npy",
        "noisy_bayer.npy",
        "field_registry.json",
        "camera_hypothesis.json",
        "experiment_config.json",
        "validation.json",
        "validation_report.html",
        "generated_region_mask.png",
        "uncertainty_map.png",
        "source_hashes.json",
        "checkpoint_hashes.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    with rawpy.imread(str(output / "synthetic_50mp.dng")) as raw:
        assert raw.white_level == (1 << adc_bits) - 1
        assert raw.raw_image.shape == (32, 32)
    catalogue = json.loads(files("prd_v5").joinpath("field_catalogue.json").read_text())
    registry = json.loads((output / "field_registry.json").read_text())
    assert {entry["name"] for entry in catalogue} <= registry.keys()
    for record in registry.values():
        assert record["method"] and record["validation"]
    for name in ("GPS", "Camera serial", "Lens serial", "Timestamp", "Battery"):
        assert registry[name]["value"] is None
    assert np.load(output / "clean_bayer.npy").shape == (32, 32)
    assert np.load(output / "noisy_bayer.npy").shape == (32, 32)
    for name, evidence in report["artifacts"].items():
        assert digest(output / name) == evidence["sha256"]
    assert report["stages"]["raster"]["method"].startswith("Lanczos-3")
    assert report["validation"]["acceptance"].startswith("conditional")
    assert np.asarray(Image.open(output / "uncertainty_map.png"))[0, 0] == 255
    validation = json.loads((output / "validation.json").read_text())
    assert validation["HDR validity"]["status"] == "failed"
    hdr = validation["HDR validity"]
    assert hdr["source_endpoint_pixels"] == int(np.count_nonzero(np.load(output / "source_clipped.npy")))
    assert hdr["unsupported_pixels"] == int(np.count_nonzero(~np.load(output / "source_support.npy")))
    assert hdr["source_endpoint_pixels"] > 0
    claims = np.load(output / "hdr_claims.npy")
    expected_forbidden = claims & (np.load(output / "source_clipped.npy") | ~np.load(output / "source_support.npy"))
    assert hdr["claimed_truth_pixels"] == int(np.count_nonzero(claims)) > 0
    assert hdr["forbidden_claim_pixels"] == int(np.count_nonzero(expected_forbidden)) > 0
    assert hdr["numerical_hdr_fidelity"] == "not measured"
    assert validation["Metadata coherence"]["status"] == "passed"
    assert "AsShotNeutral" in validation["Metadata coherence"]["checked"]
    assert "Full-well capacity" in validation["Metadata coherence"]["unmapped_registry_fields"]
    assert registry["Checkpoint hashes"]["value"] is None
    assert registry["Checkpoint hashes"]["provenance"] == "UNSPECIFIED"
    page = (output / "validation_report.html").read_text()
    parser = _TableRows()
    parser.feed(page)
    rendered = {row[0]: row[1:] for row in parser.rows if row}
    for gate, record in validation.items():
        if isinstance(record, dict) and "status" in record:
            assert rendered[gate][0] == record["status"]
    assert rendered["HDR validity"] == ["failed", validation["HDR validity"]["reason"]]
    assert "Lanczos-3" in report["pipeline"] and "bilinear" not in report["pipeline"]
    assert "Lanczos-3" in page


@pytest.mark.parametrize("iso", [50, 400])
def test_portrait_pitch_and_clean_electrons_match_adc(tmp_path, iso):
    from prd_v5.config import AnalyticRunConfig

    source = tmp_path / "source.png"
    Image.fromarray(np.full((40, 30, 3), 90, dtype=np.uint8)).save(source)
    camera = CameraHypothesis(iso=iso, shot_noise=False, read_noise=False)
    output = tmp_path / "run"
    report = AnalyticRunConfig(camera=camera, target_megapixels=0.0012, assume_unprofiled_srgb=True).run(source, output)
    width, height = report["geometry"]["width"], report["geometry"]["height"]
    assert height > width
    registry = json.loads((output / "field_registry.json").read_text())
    assert registry["Pixel pitch"]["value"] == pytest.approx(35900 / height)
    assert registry["Sensor physical size"]["value"] == pytest.approx([35.9 * width / height, 35.9])
    electron_per_dn = camera.conversion_gain_e_per_dn / camera.analogue_gain
    np.testing.assert_allclose(
        np.load(output / "clean_bayer.npy"), np.load(output / "noisy_bayer.npy"), atol=electron_per_dn / 2 + 1e-3
    )


@pytest.mark.parametrize("seed", [0, 42, 100])
def test_photon_transfer_measures_consistent_temporal_model(seed):
    result = photon_transfer(CameraHypothesis(seed=seed))
    assert result["status"] == "passed"
    assert result["slope_relative_error"] < result["relative_tolerance"]
    assert result["dark_relative_error"] < result["relative_tolerance"]
    assert result["scope"] == "baseline simulator self-consistency only"


def test_gain_clipping_is_not_reported_as_noise_validation():
    assert photon_transfer(CameraHypothesis(iso=102400))["status"] == "not measured"


def test_extended_low_iso_well_and_adc_tags_remain_coherent(tmp_path):
    camera = CameraHypothesis(iso=50, shot_noise=False, read_noise=False)
    gain = camera.analogue_gain / camera.conversion_gain_e_per_dn
    assert camera.saturation_e == camera.full_well_e
    assert camera.white_level == round(camera.black_level + camera.full_well_e * gain)
    assert camera.white_level < (1 << camera.adc_bits) - 1
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), "white").save(source)
    config = replace(full_quality_config(), target_megapixels=0.001024, camera=camera)
    output = tmp_path / "run"
    report = config.run(source, output)
    with rawpy.imread(str(output / "capture.dng")) as raw:
        assert raw.white_level == camera.white_level
        assert int(raw.raw_image.max()) <= camera.white_level
    assert report["registry"]["White level"]["value"] == camera.white_level


def _hdr_inputs(tmp_path, claims, *, effective=True):
    clipped = np.zeros((4, 4), dtype=bool)
    clipped[0, 0] = True
    if claims is not None:
        np.save(tmp_path / "hdr_claims.npy", claims)
    report = {"stages": {"analytic": {"hdr": {"effective": effective}}}}
    return report, clipped, np.ones((4, 4), dtype=bool)


def test_hdr_audit_passes_when_expansion_avoids_clipped_pixels(tmp_path):
    from prd_v5.delivery import audit_hdr

    claims = np.zeros((4, 4), dtype=bool)
    claims[2, 2] = True
    report, clipped, support = _hdr_inputs(tmp_path, claims)
    result = audit_hdr(tmp_path, report, clipped, support)
    assert result["status"] == "passed"
    assert result["claimed_truth_pixels"] == 1 and result["forbidden_claim_pixels"] == 0
    assert result["numerical_hdr_fidelity"] == "not measured"


def test_hdr_audit_fails_when_expansion_claims_clipped_pixels(tmp_path):
    from prd_v5.delivery import audit_hdr

    claims = np.zeros((4, 4), dtype=bool)
    claims[0, 0] = claims[2, 2] = True
    report, clipped, support = _hdr_inputs(tmp_path, claims)
    result = audit_hdr(tmp_path, report, clipped, support)
    assert result["status"] == "failed"
    assert result["forbidden_claim_pixels"] == 1


def test_hdr_audit_not_measured_without_hdr_stage(tmp_path):
    from prd_v5.delivery import audit_hdr

    report, clipped, support = _hdr_inputs(tmp_path, None, effective=False)
    result = audit_hdr(tmp_path, report, clipped, support)
    assert result["status"] == "not measured"
    assert "No HDR" in result["reason"]


def test_analytic_claim_mask_marks_clipped_source_only_when_hdr_runs():
    from prd_v5.analytic import AnalyticISP

    block = np.full((2, 2, 3), 0.2, dtype=np.float32)
    block[0, 0] = 1.0
    assert not AnalyticISP().hdr_claim_mask(block).any()
    mask = AnalyticISP(hdr_max_radiance=4).hdr_claim_mask(block)
    assert mask.tolist() == [[True, False], [False, False]]
