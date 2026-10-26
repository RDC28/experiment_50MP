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
    validate_and_report,
)
from synthraw.validation import ValidationCheck, ValidationThresholds
from synthraw.validation_runner import ValidationRunner


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
    assert "<em>" not in text and "—" not in text


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
    ok = ValidationCheck("noise_fit_r2", "Noise", True, target="R^2 >= 0.95")
    unmeasured = ValidationCheck("roundtrip_supported_ssim", "Round trip", None)
    assert answer([hard_fail, qa_miss]).startswith("No - hard gate failed: cfa_phase")
    assert answer([qa_miss, ok]).startswith("Partially")
    assert answer([ok]).startswith("Yes")
    assert answer([unmeasured]) == "Not measured in this run"
    assert format_measured({"a": 0.123456, "b": {"c": [1, 2, 3, 4, 5, 6, 7]}, "d": None}) == (
        "a=0.1235; b={c=[1, 2, 3, 4, 5, 6, ...]}; d=None"
    )
