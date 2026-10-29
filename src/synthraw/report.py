"""PRD_v3 self-contained HTML validation report (``validation/validation_report.html``).

``ReportGenerator`` renders a finished ``ValidationRun`` into one HTML file with inline CSS and
base64-embedded figures. It answers the PRD's four validation questions, lists every hard gate and
QA target against its threshold, reports generated-versus-supported percentages and carries the
synthetic-capture disclosure. A failed hard gate yields a FAILED report, never a success report.
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from synthraw.figures import FIGURE_NAMES, ValidationFigureWriter
from synthraw.provenance import PROVENANCE_CLASSES, SCIENTIFIC_BOUNDARY, SYNTHETIC_CAPTURE_DISCLOSURE
from synthraw.validation import ValidationCheck, ValidationThresholds, hard_failures
from synthraw.validation_runner import ValidationRun, ValidationRunner

REPORT_NAME = "validation_report.html"
# PRD "Validation, outputs and production acceptance": the four questions, the rows answering
# each, and the figures shown beside them. Rows not listed fall under internal coherence.
QUESTIONS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "visual",
        "Is the final image visually convincing?",
        (
            "roundtrip_supported_ssim",
            "roundtrip_whole_ssim",
            "colour_delta_e00",
            "roundtrip_unaligned",
            "sr_seam_ratio",
        ),
        ("roundtrip_difference.png",),
    ),
    (
        "coherent",
        "Is the RAW internally physically/statistically coherent?",
        (),
        (
            "cfa_planes.png",
            "mean_variance_curve.png",
            "noise_histograms.png",
            "noise_psd.png",
            "colour_validation.png",
        ),
    ),
    (
        "development",
        "Does it survive ordinary RAW development?",
        (
            "libraw_reopen",
            "demosaic_ahd",
            "demosaic_additional",
            "exposure_range",
            "white_balance_response",
            "rawtherapee_neutral_render",
        ),
        ("demosaic_comparison.png", "exposure_contact_sheet.png", "wb_contact_sheet.png"),
    ),
    (
        "support",
        "How much information was actually supported by the input versus fabricated?",
        ("hdr_generated_area",),
        (),
    ),
)
# Confidence masks reported as percentages of the output frame (PRD HDR validation).
SUPPORT_MASKS = (
    ("source_supported_mask.npy", "Source-supported pixels (HDR, not extrapolated)"),
    ("sr_source_supported_mask.npy", "SR geometric source support"),
    ("highlight_fabricated_mask.npy", "Highlight radiance fabricated"),
    ("shadow_fabricated_mask.npy", "Shadow radiance fabricated"),
    ("outpaint_mask.npy", "Outpaint bands (invented edge extension)"),
    ("generated_detail_mask.npy", "Learned SR detail (model-synthesised)"),
)
# Numeric gates and targets checked for a thin margin:
# (check, measured key, threshold field, higher is better, upper bound of the metric or None).
MARGIN_RULES: tuple[tuple[str, str, str, bool, float | None], ...] = (
    ("roundtrip_supported_ssim", "ssim", "roundtrip_supported_ssim", True, 1.0),
    ("roundtrip_whole_ssim", "ssim", "roundtrip_whole_ssim", True, 1.0),
    ("colour_delta_e00", "median_delta_e00", "median_delta_e00", False, None),
    ("sr_seam_ratio", "ratio", "seam_ratio", False, None),
    ("noise_profile_agreement", "S_relative_error", "noise_profile_rel_tol", False, None),
    ("noise_profile_agreement", "O_relative_error", "noise_profile_rel_tol", False, None),
    ("noise_fit_r2", "r2", "noise_fit_r2", True, 1.0),
    ("white_balance_response", "clipped_increase", "wb_max_clipped_increase", False, None),
)
# A passing value closer than this fraction of its threshold is reported as marginal. For a metric bounded
# above (SSIM, R^2) the fraction is of the headroom between threshold and bound, not of the threshold itself.
NEAR_THRESHOLD_FRACTION = 0.10
_CSS = """
body{font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0 auto;max-width:1100px;
padding:24px;color:#1d1d1f;background:#fff}
h1{font-size:24px;margin:0 0 8px}h2{font-size:18px;margin:28px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}
h3{font-size:15px;margin:18px 0 6px}
.banner{padding:14px 18px;border-radius:6px;font-weight:600;margin:12px 0}
.banner.passed{background:#e7f6ec;border:1px solid #2e7d4f;color:#1b5e36}
.banner.failed{background:#fdecea;border:1px solid #b3261e;color:#8c1d18}
.disclosure{background:#f4f4f6;border-left:4px solid #6b6b75;padding:10px 14px;margin:12px 0}
table{border-collapse:collapse;width:100%;margin:8px 0;table-layout:fixed}table.kv{width:auto;min-width:50%}
th,td{border:1px solid #ddd;padding:5px 8px;text-align:left;vertical-align:top;overflow-wrap:anywhere}
th{background:#f4f4f6}td.num{text-align:right;font-variant-numeric:tabular-nums}
.pass{color:#1b5e36;font-weight:600}.fail{color:#b3261e;font-weight:600}.info{color:#555}
.answer{font-weight:600}figure{margin:10px 0}figure img{max-width:100%;height:auto;border:1px solid #eee}
figcaption{color:#555;font-size:12px}code{font-size:12px}
img.preview{width:min(100%,1024px);image-rendering:pixelated}
.glance{border:1px solid #c9c9d1;border-radius:6px;padding:4px 16px 10px;margin:12px 0}
.glance h2{border:0;margin-top:12px}.warn{color:#8a5300;font-weight:600}
"""


def _number(value: float) -> str:
    if not math.isfinite(value):
        return str(value)
    return f"{value:.4g}"


def format_measured(value: Any, *, max_items: int = 6, top: bool = True) -> str:
    """Compact human-readable rendering of a check's measured values (long arrays truncated)."""
    if isinstance(value, Mapping):
        items = [f"{key}={format_measured(item, max_items=max_items, top=False)}" for key, item in value.items()]
        return "; ".join(items) if top else "{" + ", ".join(items) + "}"
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        items = [format_measured(item, max_items=max_items, top=False) for item in value[:max_items]]
        return "[" + ", ".join(items) + (", ..." if len(value) > max_items else "") + "]"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, float):
        return _number(value)
    return str(value)


def _status(check: ValidationCheck) -> tuple[str, str]:
    if check.passed is None:
        return "info", "report only" if _kind(check) == "report only" else "n/a"
    if check.passed:
        return "pass", "pass"
    return "fail", "FAIL"


def _kind(check: ValidationCheck) -> str:
    if check.hard:
        return "hard gate"
    return "report only" if not check.target or "report" in check.target else "QA target"


def question_of(check: ValidationCheck) -> str:
    """Key of the PRD question a validation row answers."""
    for key, _title, names, _figures in QUESTIONS:
        if check.name in names:
            return key
    return "coherent"


def answer(checks: Iterable[ValidationCheck], *, report_only: bool = False) -> str:
    """One-line verdict for a question from its rows; hard failures dominate missed QA targets."""
    checks = list(checks)
    if hard_failures(checks):
        return "No - hard gate failed: " + ", ".join(check.name for check in hard_failures(checks))
    missed = [check.name for check in checks if check.passed is False]
    if missed:
        return "Partially - QA targets missed (measured baseline reported): " + ", ".join(missed)
    if report_only:
        return "Reported below - measured percentages, no arbitrary pass threshold"
    if not any(check.passed for check in checks):
        return "Not measured in this run"
    return "Yes - every gate and target in this section passed"


def mask_percentages(root: str | Path, *, strip_rows: int = 512) -> dict[str, float]:
    """Percentage of the frame covered by each confidence mask, scanned strip-wise (float masks at 0.5)."""
    confidence = Path(root) / "confidence"
    out = {}
    for name, _label in SUPPORT_MASKS:
        path = confidence / name
        if not path.is_file():
            continue
        mask = np.load(path, mmap_mode="r")
        hits = sum(
            int(np.count_nonzero(np.asarray(mask[i : i + strip_rows]) > 0.5)) for i in range(0, len(mask), strip_rows)
        )
        out[name] = 100.0 * hits / max(1, mask.size)
    return out


def _embed_png(path: Path, *, max_side: int | None = None) -> str:
    if max_side is None:
        data = path.read_bytes()
    else:
        from PIL import Image

        with Image.open(path) as image:
            image.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG", optimize=True)
            data = buffer.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.is_file() else {}


def near_threshold(checks: Iterable[ValidationCheck], thresholds: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Gates and QA targets that passed by less than NEAR_THRESHOLD_FRACTION of their threshold (or headroom)."""
    by_name = {check.name: check for check in checks}
    rows = []
    for name, key, threshold_name, higher, bound in MARGIN_RULES:
        check = by_name.get(name)
        value = None if check is None else check.measured.get(key)
        threshold = thresholds.get(threshold_name)
        if check is None or not check.passed or not isinstance(value, (int, float)) or not threshold:
            continue
        scale = bound - threshold if bound is not None and bound > threshold else threshold
        margin = (value - threshold) / scale if higher else (threshold - value) / scale
        if 0 <= margin < NEAR_THRESHOLD_FRACTION:
            rows.append(
                {
                    "check": name,
                    "metric": key,
                    "value": float(value),
                    "threshold": float(threshold),
                    "margin": float(margin),
                    "hard": check.hard,
                }
            )
    return rows


class ReportGenerator:
    """Write ``validation/validation_report.html`` for a finished ``ValidationRun``."""

    def __init__(self, *, preview_side: int = 1024) -> None:
        self.preview_side = preview_side

    def generate(self, run: ValidationRun, figures: Mapping[str, Path] | None = None) -> Path:
        root = Path(run.root)
        out = root / "validation"
        out.mkdir(parents=True, exist_ok=True)
        if figures is None:
            figures = {name: out / name for name in FIGURE_NAMES if (out / name).is_file()}
        path = out / REPORT_NAME
        path.write_text(self.render(run, figures), encoding="utf-8")
        return path

    def render(self, run: ValidationRun, figures: Mapping[str, Path]) -> str:
        root = Path(run.root)
        esc = html.escape
        failures = hard_failures(run.checks)
        status = run.status
        provenance = _read_json(root / "metadata" / "provenance.json")
        thresholds = run.summary.get("thresholds") or asdict(ValidationThresholds())
        parts = [
            "<!DOCTYPE html>",
            '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>Synthetic RAW validation - {esc(status)} - {esc(root.name)}</title>",
            f"<style>{_CSS}</style></head><body>",
            "<h1>Synthetic RAW validation report</h1>",
            f"<p>Run <code>{esc(root.name)}</code> - generated "
            f"{esc(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'))}</p>",
        ]
        if failures:
            parts.append(
                '<div class="banner failed">VALIDATION FAILED - '
                f"{len(failures)} hard gate(s) failed: {esc(', '.join(c.name for c in failures))}. "
                "This run is not an acceptable synthetic capture; its outputs must not be used as a successful "
                "result.</div>"
            )
        else:
            missed = [c.name for c in run.checks if not c.hard and c.passed is False]
            note = (
                f" {len(missed)} QA target(s) missed and reported with measured values: {esc(', '.join(missed))}."
                if missed
                else " All QA targets met."
            )
            parts.append(f'<div class="banner passed">VALIDATION PASSED - every hard gate passed.{note}</div>')
        parts += [
            '<div class="disclosure"><strong>Synthetic-capture disclosure.</strong> '
            f"{esc(SYNTHETIC_CAPTURE_DISCLOSURE)}<br><strong>Scientific boundary.</strong> "
            f"{esc(SCIENTIFIC_BOUNDARY)}</div>",
        ]
        parts += self._at_a_glance(run, provenance, thresholds)
        parts += self._run_facts(run, provenance)
        parts.append("<h2>The four validation questions</h2>")
        parts.append("<table><tr><th style='width:40%'>Question</th><th>Answer</th></tr>")
        for key, title, _names, _figures in QUESTIONS:
            rows = [check for check in run.checks if question_of(check) == key]
            verdict = answer(rows, report_only=key == "support")
            parts.append(f"<tr><td>{esc(title)}</td><td class='answer'>{esc(verdict)}</td></tr>")
        parts.append("</table>")
        parts += self._hard_gates(run.checks)
        for key, title, _names, figure_names in QUESTIONS:
            parts.append(f"<h2>{esc(title)}</h2>")
            if key == "visual":
                preview = root / "masters" / "preview_8bit.png"
                if preview.is_file():
                    parts.append(
                        self._figure(
                            _embed_png(preview, max_side=self.preview_side),
                            "Preview of the scene-linear master (downsampled)",
                            preview=True,
                        )
                    )
            if key == "support":
                parts += self._support(run, provenance)
            parts += self._metrics([check for check in run.checks if question_of(check) == key])
            for name in figure_names:
                if name in figures and Path(figures[name]).is_file():
                    parts.append(self._figure(_embed_png(Path(figures[name])), name))
                else:
                    parts.append(f"<p class='info'>{esc(name)} not written for this run.</p>")
        parts += self._realism(root, figures)
        parts.append("<h2>Thresholds (configuration)</h2>")
        parts.append(
            "<p class='info'>SSIM and deltaE00 limits are project QA targets, not claims about published model "
            "performance; missing them is reported, never hidden by altering output.</p>"
            "<table class='kv'><tr><th>Threshold</th><th>Value</th></tr>"
        )
        parts += [f"<tr><td>{esc(str(k))}</td><td class='num'>{esc(str(v))}</td></tr>" for k, v in thresholds.items()]
        parts.append("</table></body></html>")
        return "\n".join(parts) + "\n"

    @staticmethod
    def _figure(src: str, caption: str, *, preview: bool = False) -> str:
        caption = html.escape(caption)
        css = ' class="preview"' if preview else ""
        return f'<figure><img{css} src="{src}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'

    @staticmethod
    def _at_a_glance(run: ValidationRun, provenance: Mapping[str, Any], thresholds: Mapping[str, Any]) -> list[str]:
        """What a pass hides: unaligned round-trip numbers, the alignment gain, thin margins, skipped features."""
        esc = html.escape
        checks = {check.name: check for check in run.checks}
        parts = ["<div class='glance'><h2>Read this first</h2>"]
        unaligned_check = checks.get("roundtrip_unaligned")
        colour = checks.get("colour_delta_e00")
        if unaligned_check is not None and colour is not None:
            unaligned = unaligned_check.measured.get("unaligned", {})
            aligned = {
                "supported_ssim": checks["roundtrip_supported_ssim"].measured.get("ssim"),
                "whole_ssim": checks["roundtrip_whole_ssim"].measured.get("ssim"),
                "median_delta_e00_supported": colour.measured.get("median_delta_e00"),
            }
            gain = colour.measured.get("alignment_gain")
            gain_text = "n/a"
            if isinstance(gain, (int, float)) and gain > 0:
                gain_text = f"{gain:.4g} ({math.log2(gain):+.2f} EV)"
            parts.append(
                "<p>The round-trip gates are scored after a global 3x3 colour alignment. The same comparison "
                f"without it, and the grey-axis gain the alignment applied (near 1 when exposure and "
                f"BaselineExposure are consistent): <strong>alignment gain {esc(gain_text)}</strong>.</p>"
                "<table class='kv'><tr><th>Round trip</th><th>Unaligned</th><th>Aligned (gated)</th></tr>"
            )
            for key, label in (
                ("supported_ssim", "SSIM, source-supported pixels"),
                ("whole_ssim", "SSIM, whole frame"),
                ("median_delta_e00_supported", "Median deltaE00, source-supported"),
            ):
                parts.append(
                    f"<tr><td>{esc(label)}</td><td class='num'>{esc(format_measured(unaligned.get(key)))}</td>"
                    f"<td class='num'>{esc(format_measured(aligned[key]))}</td></tr>"
                )
            parts.append("</table>")
        marginal = near_threshold(run.checks, thresholds)
        if marginal:
            parts.append(
                f"<p class='warn'>Passed within {NEAR_THRESHOLD_FRACTION:.0%} of the threshold, or of the "
                "headroom to 1 for SSIM and R^2 (a small change in content or seed could fail them):</p>"
                "<table class='kv'><tr><th>Check</th><th>Measured</th><th>Threshold</th><th>Margin</th></tr>"
            )
            parts += [
                f"<tr><td>{esc(row['check'])} ({esc(row['metric'])}{', hard gate' if row['hard'] else ''})</td>"
                f"<td class='num'>{row['value']:.4g}</td><td class='num'>{row['threshold']:.4g}</td>"
                f"<td class='num'>{100 * row['margin']:.1f}%</td></tr>"
                for row in marginal
            ]
            parts.append("</table>")
        else:
            parts.append(f"<p>No gate or target passed within {NEAR_THRESHOLD_FRACTION:.0%} of its threshold.</p>")
        skipped = [str(reason) for reason in provenance.get("fallbacks", [])]
        skipped += [
            f"{check.name}: {check.detail or 'not measured'}"
            for check in run.checks
            if check.passed is None and "report" not in check.target
        ]
        if skipped:
            parts.append("<p class='warn'>Skipped, fallen back or no-op features in this run:</p><ul>")
            parts += [f"<li>{esc(reason)}</li>" for reason in skipped]
            parts.append("</ul>")
        else:
            parts.append("<p>No feature was skipped, fell back or ran as a no-op.</p>")
        parts.append("</div>")
        return parts

    @staticmethod
    def _realism(root: Path, figures: Mapping[str, Path]) -> list[str]:
        """Side-by-side realism metrics against a real reference raw, when one was supplied."""
        from synthraw.realism import REALISM_FIGURE, REALISM_JSON

        esc = html.escape
        document = _read_json(root / "validation" / REALISM_JSON)
        parts = ["<h2>Realism against a real reference raw</h2>"]
        if not document:
            parts.append(
                "<p class='info'>No reference raw was supplied, so realism was not measured. The noise gates "
                "above check the simulator against its own NoiseProfile (self-consistency), not against a real "
                "sensor. Set REFERENCE_RAW in the notebook validation cell to compare.</p>"
            )
            return parts
        if document.get("error"):
            parts.append(f"<p class='fail'>Realism comparison unavailable: {esc(str(document['error']))}</p>")
            return parts
        reference = document.get("reference", {})
        parts.append(
            f"<p class='info'>{esc(document.get('label', ''))}. Reference "
            f"<code>{esc(str(reference.get('file')))}</code>. {esc(document.get('caveat', ''))}</p>"
            "<table><tr><th style='width:34%'>Metric</th><th style='width:7%'>Plane</th><th>Unit</th>"
            "<th>Reference</th><th>Synthetic</th><th>Synthetic / reference</th></tr>"
        )
        for row in document.get("rows", []):
            parts.append(
                f"<tr><td>{esc(row['metric'])}</td><td>{esc(row['plane'])}</td><td>{esc(row['unit'])}</td>"
                f"<td class='num'>{esc(format_measured(row['reference']))}</td>"
                f"<td class='num'>{esc(format_measured(row['synthetic']))}</td>"
                f"<td class='num'>{esc(format_measured(row['ratio']))}</td></tr>"
            )
        parts.append("</table>")
        figure = figures.get(REALISM_FIGURE) or root / "validation" / REALISM_FIGURE
        if Path(figure).is_file():
            parts.append(
                ReportGenerator._figure(_embed_png(Path(figure)), "Realism: reference raw against the synthetic DNG")
            )
        return parts

    @staticmethod
    def _run_facts(run: ValidationRun, provenance: Mapping[str, Any]) -> list[str]:
        esc = html.escape
        summary = run.summary
        facts = [
            ("DNG", summary.get("dng")),
            ("Mosaic (rows x cols)", " x ".join(str(v) for v in summary.get("mosaic_shape", [])) or None),
            ("CFA phase", summary.get("cfa_phase")),
            (
                "DNG identity",
                format_measured(provenance.get("dng_identity")) if provenance.get("dng_identity") else None,
            ),
            ("Source SHA-256", provenance.get("source", {}).get("sha256")),
            ("Backends", format_measured(provenance.get("backends")) if provenance.get("backends") else None),
            ("Fallbacks", ", ".join(provenance.get("fallbacks", [])) or "none"),
        ]
        rows = [
            f"<tr><th style='width:30%'>{esc(k)}</th><td>{esc(str(v))}</td></tr>" for k, v in facts if v is not None
        ]
        return ["<h2>Run</h2><table>", *rows, "</table>"]

    @staticmethod
    def _hard_gates(checks: list[ValidationCheck]) -> list[str]:
        esc = html.escape
        parts = ["<h2>Hard gates</h2>", "<table><tr><th>Gate</th><th>Domain</th><th>Criterion</th><th>Result</th></tr>"]
        for check in checks:
            if check.hard:
                css, label = _status(check)
                parts.append(
                    f"<tr><td>{esc(check.name)}</td><td>{esc(check.domain)}</td><td>{esc(check.target)}</td>"
                    f"<td class='{css}'>{label}</td></tr>"
                )
        parts.append("</table>")
        return parts

    @staticmethod
    def _metrics(checks: list[ValidationCheck]) -> list[str]:
        esc = html.escape
        if not checks:
            return []
        parts = [
            "<table><tr><th style='width:16%'>Check</th><th style='width:9%'>Type</th><th style='width:16%'>Target</th>"
            "<th>Measured</th><th style='width:8%'>Result</th></tr>"
        ]
        for check in checks:
            css, label = _status(check)
            measured = format_measured(check.measured)
            if check.detail:
                measured = f"{measured} - {check.detail}" if measured else check.detail
            parts.append(
                f"<tr><td>{esc(check.name)}</td><td>{esc(_kind(check))}</td><td>{esc(check.target)}</td>"
                f"<td>{esc(measured)}</td><td class='{css}'>{label}</td></tr>"
            )
        parts.append("</table>")
        return parts

    @staticmethod
    def _support(run: ValidationRun, provenance: Mapping[str, Any]) -> list[str]:
        esc = html.escape
        labels = dict(SUPPORT_MASKS)
        rows = [
            ("Generated HDR area", run.summary.get("generated_hdr_area_percentage")),
            *((labels[name], value) for name, value in mask_percentages(run.root).items()),
        ]
        fraction = run.summary.get("source_supported_fraction")
        if fraction is not None:
            rows.append(("Round-trip mask (SR support and HDR support)", 100.0 * float(fraction)))
        parts = [
            "<h3>Generated versus supported area</h3><table class='kv'><tr><th>Region</th><th>% of frame</th></tr>"
        ]
        parts += [
            f"<tr><td>{esc(label)}</td><td class='num'>{'n/a' if value is None else f'{float(value):.2f}'}</td></tr>"
            for label, value in rows
        ]
        parts.append("</table>")
        counts = provenance.get("field_provenance_counts")
        if counts:
            total = max(1, sum(counts.values()))
            parts.append(
                "<h3>Metadata field provenance</h3><table class='kv'><tr><th>Class</th><th>Fields</th><th>%</th></tr>"
            )
            parts += [
                f"<tr><td>{esc(code)} - {esc(PROVENANCE_CLASSES.get(code, ''))}</td><td class='num'>{count}</td>"
                f"<td class='num'>{100.0 * count / total:.1f}</td></tr>"
                for code, count in counts.items()
            ]
            parts.append("</table>")
        return parts


def validate_and_report(
    root: str | Path,
    *,
    runner: ValidationRunner | None = None,
    figure_writer: ValidationFigureWriter | None = None,
    report: ReportGenerator | None = None,
    reference_run: str | Path | None = None,
    reference_raw: str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[ValidationRun, Path]:
    """Run the validation sequence, write the figures and the HTML report; returns the run and report path.

    ``reference_raw`` is a real camera raw (ARW, DNG, ...) measured side by side with the run's DNG
    for the report's realism section. It is read in place, never copied into the run tree, and a
    file LibRaw cannot measure is reported in that section rather than failing validation.
    """
    run = (runner or ValidationRunner()).run(root, reference_run=reference_run, log=log)
    if reference_raw is not None:
        from synthraw.realism import REALISM_JSON, write_realism

        try:
            write_realism(root, reference_raw)
        except Exception as error:  # noqa: BLE001 - a bad reference must not hide the validation result
            out = Path(root) / "validation"
            out.mkdir(parents=True, exist_ok=True)
            (out / REALISM_JSON).write_text(json.dumps({"error": f"{type(error).__name__}: {error}"}) + "\n")
            if log:
                log(f"Realism comparison unavailable: {error}")
    figures = (figure_writer or ValidationFigureWriter()).write(run)
    return run, (report or ReportGenerator()).generate(run, figures)
