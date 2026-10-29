"""PRD_v3 HTML validation report produced by a tiny end-to-end validation run."""

import json
import re
import shutil
from datetime import datetime

import numpy as np
import pytest
from PIL import Image

from synthraw.config import ExperimentConfig
from synthraw.figures import FIGURE_NAMES
from synthraw.pipeline import run_pipeline
from synthraw.provenance import SCIENTIFIC_BOUNDARY, SYNTHETIC_CAPTURE_DISCLOSURE
from synthraw.report import (
    QUESTIONS,
    REPORT_NAME,
    ReportGenerator,
    answer,
    format_measured,
    mask_percentages,
    near_threshold,
    validate_and_report,
)
from synthraw.validation import ValidationCheck, ValidationThresholds
from synthraw.validation_runner import ValidationRun, ValidationRunner


@pytest.fixture(scope="module")
def run_root(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("report_run")
    rows, cols = np.mgrid[0:20, 0:48]
    pixels = np.stack([255 * cols / 47, 255 * rows / 19, 128 + 100 * np.sin(cols / 6)], axis=-1)
    source = tmp_path / "source.png"
    Image.fromarray(np.clip(np.round(pixels), 0, 255).astype(np.uint8)).save(source)
    config = ExperimentConfig(OUTPUT_GEOMETRY="CINEMA_50MP_VIRTUAL", FAST_DEV_RUN=True, SR_BACKEND="LANCZOS")
    return run_pipeline(
        source, config, tmp_path / "runs", timestamp=datetime(2026, 1, 2, 3, 4, 5), fast_dev_max_side=96
    ).root


@pytest.fixture(scope="module")
def reported(run_root):
    return validate_and_report(run_root)


def test_tiny_end_to_end_run_writes_a_self_contained_passed_report(reported):
    run, path = reported
    assert path == run.root / "validation" / REPORT_NAME and path.is_file()
    text = path.read_text()
    assert run.status == "PASSED"
    assert "VALIDATION PASSED" in text and "VALIDATION FAILED" not in text
    # Self-contained: every image is inlined, nothing is fetched from elsewhere.
    sources = re.findall(r'src="([^"]+)"', text)
    assert sources and all(src.startswith("data:image/png;base64,") for src in sources)
    assert len(sources) == len(FIGURE_NAMES) + 1  # nine figures plus the downsampled preview
    assert not re.search(r'(href|src)="(https?:|file:|/)', text) and "<link" not in text and "<script" not in text
    for _key, title, _names, _figures in QUESTIONS:
        assert title in text
    assert SYNTHETIC_CAPTURE_DISCLOSURE in text and SCIENTIFIC_BOUNDARY in text
    for check in run.checks:
        assert check.name in text
    assert "Generated HDR area" in text and "Metadata field provenance" in text
    # The measured round-trip baseline is reported, not just a pass mark.
    ssim = next(c for c in run.checks if c.name == "roundtrip_supported_ssim").measured["ssim"]
    assert f"ssim={ssim:.4g}" in text
    assert "<em>" not in text and "\u2014" not in text


def test_failed_hard_gate_yields_a_failed_report(reported, tmp_path):
    run, _ = reported
    root = shutil.copytree(run.root, tmp_path / "copy")
    radiance = np.load(root / "intermediates" / "04_hdr_radiance.npy")
    radiance[3, 5, 1] = -0.5
    np.save(root / "intermediates" / "04_hdr_radiance.npy", radiance)
    failed, path = validate_and_report(root)
    text = path.read_text()
    assert failed.status == "FAILED"
    assert "VALIDATION FAILED" in text and "VALIDATION PASSED" not in text
    assert "hdr_negative_radiance" in text.split("VALIDATION FAILED", 1)[1].split("</div>", 1)[0]
    assert "<title>Synthetic RAW validation - FAILED" in text
    assert "No - hard gate failed" in text
    assert json.loads((root / "validation" / "validation_results.json").read_text())["status"] == "FAILED"


def test_missed_qa_targets_are_reported_without_failing(reported, tmp_path):
    run, _ = reported
    root = shutil.copytree(run.root, tmp_path / "strict")
    strict = ValidationThresholds(roundtrip_supported_ssim=0.99999, median_delta_e00=1e-6)
    result, path = validate_and_report(root, runner=ValidationRunner(strict))
    text = path.read_text()
    assert result.status == "PASSED"
    assert "QA target(s) missed" in text and "Partially - QA targets missed" in text
    assert "roundtrip_supported_ssim" in text.split("QA target(s) missed", 1)[1].split("</div>", 1)[0]
    assert "0.99999" in text  # the configured threshold is shown


def test_report_without_figures_marks_them_missing(reported, tmp_path):
    run, _ = reported
    root = shutil.copytree(run.root, tmp_path / "bare")
    for name in FIGURE_NAMES:
        (root / "validation" / name).unlink(missing_ok=True)
    run.root, original = root, run.root
    try:
        text = ReportGenerator().generate(run).read_text()
    finally:
        run.root = original
    assert all(f"{name} not written for this run" in text for name in FIGURE_NAMES)


def test_mask_percentages_match_the_confidence_masks(run_root):
    percentages = mask_percentages(run_root, strip_rows=7)
    supported = np.load(run_root / "confidence" / "source_supported_mask.npy")
    assert percentages["source_supported_mask.npy"] == pytest.approx(100 * supported.mean())
    assert set(percentages) >= {"highlight_fabricated_mask.npy", "shadow_fabricated_mask.npy"}


def test_answers_and_formatting():
    hard_fail = ValidationCheck("cfa_phase", "CFA", False, hard=True)
    qa_miss = ValidationCheck("roundtrip_whole_ssim", "Round trip", False, target=">= 0.93")
    ok = ValidationCheck("noise_fit_r2", "Noise self-consistency", True, target="R^2 >= 0.95")
    unmeasured = ValidationCheck("roundtrip_supported_ssim", "Round trip", None)
    assert answer([hard_fail, qa_miss]).startswith("No - hard gate failed: cfa_phase")
    assert answer([qa_miss, ok]).startswith("Partially")
    assert answer([ok]).startswith("Yes")
    assert answer([unmeasured]) == "Not measured in this run"
    assert format_measured({"a": 0.123456, "b": {"c": [1, 2, 3, 4, 5, 6, 7]}, "d": None}) == (
        "a=0.1235; b={c=[1, 2, 3, 4, 5, 6, ...]}; d=None"
    )


def _glance(text):
    return text.split("Read this first", 1)[1].split("</div>", 1)[0]


def test_report_leads_with_unaligned_numbers_alignment_gain_and_skipped_features(reported):
    """Regression: the report showed only aligned round-trip scores and hid skipped features."""
    run, path = reported
    glance = _glance(path.read_text())
    unaligned = next(c for c in run.checks if c.name == "roundtrip_unaligned").measured["unaligned"]
    gain = next(c for c in run.checks if c.name == "colour_delta_e00").measured["alignment_gain"]
    assert f"alignment gain {gain:.4g}" in glance
    assert format_measured(unaligned["median_delta_e00_supported"]) in glance
    # C2PA is requested by default but cannot sign in the test environment, so it is listed as skipped.
    assert "C2PA unavailable: " in glance
    assert "Realism against a real reference raw" in path.read_text()
    assert "self-consistency" in path.read_text()


def test_near_threshold_lists_thin_passes_only():
    thresholds = {"median_delta_e00": 2.5, "roundtrip_supported_ssim": 0.5}
    checks = [
        ValidationCheck("colour_delta_e00", "Colour", True, measured={"median_delta_e00": 2.474}),
        ValidationCheck("roundtrip_supported_ssim", "Round trip", True, measured={"ssim": 0.99}),
    ]
    rows = near_threshold(checks, thresholds)
    assert [row["check"] for row in rows] == ["colour_delta_e00"]
    assert rows[0]["margin"] == pytest.approx(0.0104)
    failed = [ValidationCheck("colour_delta_e00", "Colour", False, measured={"median_delta_e00": 2.6})]
    assert near_threshold(failed, thresholds) == []


def test_glance_lists_untargeted_skips_but_not_report_only_rows(tmp_path):
    checks = [
        ValidationCheck("roundtrip_supported_ssim", "Round trip", None, detail="TIFF not written"),
        ValidationCheck("rawtherapee_neutral_render", "DNG structure", None, target="report only", detail="ok"),
        ValidationCheck("hdr_generated_area", "HDR", None, target="report only: no threshold", detail="area"),
    ]
    text = "".join(ReportGenerator._at_a_glance(ValidationRun(tmp_path, checks), {"fallbacks": ["HDR off"]}, {}))
    skipped = text.split("Skipped, fallen back or no-op features", 1)[1]
    assert "HDR off" in skipped and "roundtrip_supported_ssim: TIFF not written" in skipped
    assert "rawtherapee_neutral_render" not in skipped and "hdr_generated_area" not in skipped


def test_near_threshold_measures_bounded_metrics_against_their_headroom():
    thresholds = {"roundtrip_supported_ssim": 0.97, "noise_fit_r2": 0.95}
    # 0.995 is within 10% of 0.97 but uses 83% of the headroom to 1; 0.972 uses under 7%.
    checks = [
        ValidationCheck("roundtrip_supported_ssim", "Round trip", True, measured={"ssim": 0.972}),
        ValidationCheck("noise_fit_r2", "Noise", True, measured={"r2": 0.995}),
    ]
    rows = near_threshold(checks, thresholds)
    assert [row["check"] for row in rows] == ["roundtrip_supported_ssim"]
    assert rows[0]["margin"] == pytest.approx(0.002 / 0.03)
    perfect = [ValidationCheck("roundtrip_supported_ssim", "Round trip", True, measured={"ssim": 1.0})]
    assert near_threshold(perfect, thresholds) == []


def test_reference_raw_adds_a_realism_section(reported, tmp_path):
    run, _ = reported
    root = shutil.copytree(run.root, tmp_path / "realism")
    # The run's own DNG stands in for a camera raw here: every ratio is then exactly 1.
    reference = tmp_path / "reference.dng"
    shutil.copy(root / "raw" / "synthetic_50mp.dng", reference)
    _, path = validate_and_report(root, reference_raw=reference)
    document = json.loads((root / "validation" / "realism_comparison.json").read_text())
    assert document["label"].startswith("Realism")
    ratios = [row["ratio"] for row in document["rows"] if row["ratio"] is not None]
    assert ratios and all(ratio == pytest.approx(1) for ratio in ratios)
    text = path.read_text()
    section = text.split("Realism against a real reference raw", 1)[1]
    assert "reference.dng" in section and "Dark positive tail" in section
    assert "Realism: reference raw against the synthetic DNG" in section
    assert not list(root.rglob("reference.dng"))  # the reference is read in place, never copied in

    (tmp_path / "broken.arw").write_bytes(b"not a raw")
    result, path = validate_and_report(root, reference_raw=tmp_path / "broken.arw")
    assert result.status == "PASSED"
    assert "Realism comparison unavailable" in path.read_text()
