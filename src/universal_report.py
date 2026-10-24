"""The notebook 03 run report (PRD 27): rebuilt from ``manifest.json`` alone, so a failed run still reports.

It leads with the final image and the scene classification (with the cues and reasons behind it), then the
route's reconstruction, the standardised evaluation against Lanczos, the crops and the stage log. The
specialised human and night routes link to their own full reports.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .report_builder import _bytes, _dims, _models_text, _num, _table, _version


def write_universal_report(manifest: dict, path: Path) -> Path:
    path.write_text(render_universal_report(manifest, path.parent))
    return path


def _relative(path: str | None, report_dir: Path) -> str:
    if not path:
        return "-"
    return os.path.relpath(path, report_dir)


def _value(value, digits: int = 4) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return _num(value, digits)
    return "-" if value is None else str(value)


def render_universal_report(manifest: dict, report_dir: Path) -> str:
    """The Markdown report; figures are linked relative to ``report_dir``."""
    source = manifest.get("source", {})
    figures = manifest.get("figures", {})
    route = manifest.get("route", {})
    out = [f"# Phase 4 universal report: {Path(source.get('original_path', 'unknown source')).name}\n"]
    out.append(f"> {manifest.get('notice', '')}\n")
    out.append(f"Status: **{manifest.get('status')}**\n")
    if manifest.get("error"):
        out.append(f"Failure: `{manifest['error']['message']}`. Outputs written before the failure are kept.\n")

    out.append("## 1. Final full-size image\n")
    out.append(final_section(manifest))
    if (manifest.get("final_output") or {}).get("preview"):
        out.append(f"![final image]({manifest['final_output']['preview']})\n")
    out.append("## 2. Run summary\n")
    out.append(summary_section(manifest))
    out.append("## 3. Scene analysis and classification\n")
    out.append(scene_section(manifest))
    if figures.get("scene"):
        out.append(f"![scene analysis]({_relative(figures['scene'], report_dir)})\n")
    out.append("## 4. Reconstruction\n")
    out.append(reconstruction_section(manifest, report_dir))
    for key, caption in (("regions", "combined region masks"), ("fallbacks", "failure-rule fallbacks")):
        if figures.get(key):
            out.append(f"![{caption}]({_relative(figures[key], report_dir)})\n")
    out.append("## 5. Standardised evaluation\n")
    out.append(evaluation_section(manifest))
    panels = route.get("crop_panels") or []
    if panels:
        out.append("## 6. Crop comparisons\n")
        out.append(
            "Identical crops at 100% output pixels: source (nearest-neighbour), each candidate, the final image "
            "and the amplified difference between the processing input and the final.\n"
        )
        for panel in panels:
            link = _relative(panel["path"], report_dir)
            out.append(f"**{panel['crop']}** (output box {panel['box']})\n\n![{panel['crop']}]({link})\n")
    out.append("## 7. Conclusions\n")
    out.append("\n".join(f"- {line}" for line in universal_conclusions(manifest)) + "\n")
    out.append("## 8. Stage log\n")
    out.append(stage_section(manifest))
    out.append("## 9. Scientific limitations\n")
    out.append(
        "- The source is a rendered, display-referred image; nothing here recovers scene radiance, sensor data or "
        "clipped highlights.\n"
        "- Super-resolution synthesises detail from learned priors; it is inferred, not recovered, and may be wrong "
        "even when it looks plausible.\n"
        "- The classification, its thresholds and the improvement gates are provisional heuristics; review the "
        "crops at 100% before trusting any final image.\n"
    )
    return "\n".join(out)


def final_section(manifest: dict) -> str:
    """The final image: what it is, why, and its verified files."""
    final = manifest.get("final_output") or {}
    files = final.get("files") or {}
    if not files:
        return "No final image: the run did not reach the final-output stage.\n"
    return f"**{final['label']}**: {final['reason']}.\n\n" + _table(
        ["File", "Size", "Dimensions", "SHA-256", "Saved as"],
        [
            (
                f"`{Path(f['path']).name}`",
                _bytes(f["size_bytes"]),
                _dims(f["width"], f["height"]),
                f"`{f['file_sha256'][:16]}...`",
                f"{f['how']} of `{Path(f['from']).name}`",
            )
            for f in files.values()
        ],
    )


def summary_section(manifest: dict) -> str:
    source = manifest.get("source", {})
    runtime = manifest.get("runtime", {})
    target = manifest.get("route", {}).get("target", {})
    gpu = runtime.get("gpu")
    return _table(
        ["Item", "Value"],
        [
            ("Run ID", manifest.get("run_id", "-")),
            ("Source", f"`{Path(source.get('original_path', '-')).name}`"),
            ("Source SHA-256", f"`{source.get('sha256', '-')}`"),
            ("Input dimensions", _dims(source.get("width"), source.get("height"))),
            (
                "Output dimensions",
                _dims(target.get("width"), target.get("height"))
                + (f" ({target['megapixels']:.3f} MP)" if target.get("megapixels") else ""),
            ),
            ("Pipeline version", _version(manifest)),
            ("Notebook version", manifest.get("notebook_version") or "- (run outside a notebook)"),
            ("Models", _models_text(manifest)),
            ("GPU", f"{gpu['name']} ({gpu['memory_total_gb']} GB)" if gpu else "none detected (CPU only)"),
            ("Total runtime", f"{manifest.get('total_seconds', 0):.1f} s"),
            ("Peak resident memory", f"{manifest.get('memory', {}).get('pipeline_peak_rss_mb', '-')} MB"),
            ("Storage used", _bytes(manifest.get("storage_used_bytes"))),
        ],
    )


def scene_section(manifest: dict) -> str:
    """The category, the reasons and cues behind it, the scene statistics and the selected preset."""
    return analysis_markdown(manifest.get("scene") or {}, manifest.get("preset"))


def analysis_markdown(scene: dict, preset: dict | None = None) -> str:
    """Markdown of an ``analyse_scene`` result (also shown by the notebook before the run)."""
    classification = scene.get("classification")
    if not classification:
        return "Not reached.\n"
    out = [
        f"Category: **{classification['category']}** ({classification['confidence']} confidence), processed by "
        f"the {classification['pipeline']}.\n"
    ]
    if classification.get("needs_confirmation"):
        out.append(
            "> Classification confidence is low, so the conservative general settings are used (PRD 20.2). "
            "Confirm the category with `analysis.manual_override` (the notebook's CATEGORY field).\n"
        )
    out.append("Why:\n\n" + "\n".join(f"- {reason}" for reason in classification["reasons"]) + "\n")
    cues = [(group, c) for group, items in classification["cues"].items() for c in items]
    out.append(
        _table(
            ["Decision", "Cue", "Value", "Rule", "Passed", "Margin"],
            [
                (g, c["cue"], _value(c["value"], 3), c["rule"], _value(c["passed"]), f"{c['margin']:+.1%}")
                for g, c in cues
            ],
        )
    )
    stats = scene["statistics"]
    faces = stats["faces"]
    out.append(
        _table(
            ["Statistic", "Value"],
            [
                ("Measured on", stats["measured_on"]),
                (
                    "Luma mean / p05 / median / p95",
                    " / ".join(f"{stats['luma'][k]:.3f}" for k in ("mean", "p05", "median", "p95")),
                ),
                ("Dark fraction", f"{stats['dark_fraction']:.2%}"),
                ("Near-black fraction", f"{stats['near_black_fraction']:.2%}"),
                ("Mean saturation", f"{stats['mean_saturation']:.3f}"),
                ("Faces (method, observed, counted)", f"{faces['method']}, {faces['observed']}, {faces['counted']}"),
                (
                    "Largest face",
                    f"{faces['largest_face_px']} px shortest side, "
                    f"{faces['largest_face_area_fraction']:.2%} of the frame",
                ),
                (
                    "Light regions (bright + clipped)",
                    f"{stats['regions']['bright_light'] + stats['regions']['clipped_highlight']:.2%}",
                ),
            ],
        )
    )
    out.append(f"_{classification['note']}_\n")
    if preset:
        out.append(f"**Preset:** {preset['settings']}. Selected because: {preset['why']}.\n")
    return "\n".join(out)


def reconstruction_section(manifest: dict, report_dir: Path | None = None) -> str:
    """The route: the delegated run and its SR gate, or the common pipeline's candidates and processing."""
    route = manifest.get("route", {})
    out = []
    if route.get("delegate"):
        delegate = route["delegate"]
        report = delegate["report"] if report_dir is None else _relative(delegate["report"], report_dir)
        out.append(
            f"The {route['pipeline']} processed this scene: {route.get('label', '-')}. Its full report, with every "
            f"stage, comparison and metric, is [{Path(delegate['report']).parent.name}/report.md]({report}).\n"
        )
        out.append(
            _table(
                ["Attempt", "Status", "SR gate", "Report"],
                [
                    (
                        a["run_id"],
                        a.get("status", "-"),
                        _gate_text(a.get("gate")),
                        a.get("report", "-") if report_dir is None else _relative(a.get("report"), report_dir),
                    )
                    for a in route.get("attempts", [])
                ],
            )
        )
    elif route.get("selection"):
        selection = route["selection"]
        out.append(f"Input to universal processing: **{selection['label']}**, because {selection['why']}.\n")
        rows = []
        for candidate in selection["candidates"].values():
            m = candidate["measurements"]
            verdict = candidate.get("verdict")
            rows.append(
                (
                    candidate["label"],
                    f"{m['detail_energy']:.3e}",
                    "-" if verdict is None else f"{verdict['detail_gain']:+.1%}",
                    f"{m['source_consistency']['ssim']:.5f}",
                    f"{m['source_consistency']['delta_e_mean']:.3f}",
                    f"{m['clipped_fraction']:.4%}",
                    "baseline"
                    if verdict is None
                    else ("kept" if verdict["passed"] else "rejected: " + ", ".join(verdict["failed"])),
                )
            )
        out.append(
            _table(["Candidate", "Detail energy", "Gain vs Lanczos", "SSIM", "dE2000 mean", "Clipped", "Verdict"], rows)
        )
        processing = route.get("processing")
        if processing:
            out.append(
                f"Universal processing: {len(processing['stages'])} Layer 5B steps "
                f"({sum(s['enabled'] for s in processing['stages'])} enabled); the PRD 12.6 failure rule fell back in "
                f"{processing['fallbacks']} blocks ({processing['fallback_fraction']:.3%} of the output).\n"
            )
            out.append(
                _table(
                    ["Step", "Enabled", "Provenance", "Note"],
                    [
                        (s["name"], _value(s["enabled"]), s["provenance"], s.get("note") or "-")
                        for s in processing["stages"]
                    ],
                )
            )
        masks = route.get("masks", {})
        if masks.get("faces"):
            out.append(f"Face masks: {len(masks['faces'])}. {masks['face_policy']}\n")
        if route.get("faces"):
            out.append(_face_rows(route["faces"]))
    else:
        out.append("Not reached.\n")
    return "\n".join(out)


def evaluation_section(manifest: dict) -> str:
    """The standardised evaluation of the final master against Lanczos."""
    evaluation = manifest.get("evaluation")
    if not evaluation:
        return "Not reached.\n"
    f, lanczos, verdict = evaluation["final"], evaluation["lanczos"], evaluation["improvement"]
    fc, lc = f["source_consistency"], lanczos["source_consistency"]
    table = _table(
        ["Measurement", "Lanczos", "Final", "Check", "Passed"],
        [
            ("Detail energy", f"{lanczos['detail_energy']:.3e}", f"{f['detail_energy']:.3e}", "-", "-"),
            ("PSNR to source (dB)", _num(lc["psnr_db"], 2), _num(fc["psnr_db"], 2), "-", "-"),
            ("SSIM to source", _num(lc["ssim"], 5), _num(fc["ssim"], 5), "-", "-"),
            ("dE2000 mean to source", _num(lc["delta_e_mean"], 3), _num(fc["delta_e_mean"], 3), "-", "-"),
            ("Clipped samples", f"{lanczos['clipped_fraction']:.4%}", f"{f['clipped_fraction']:.4%}", "-", "-"),
        ]
        + [
            ("", "", "", f"{c['check']} {_value(c['value'])} ({c['limit']})", _value(c["passed"]))
            for c in verdict["checks"]
        ],
    )
    return (
        f"{evaluation['compared']} ({_dims(*evaluation['size'])}).\n\n"
        + table
        + "\nMeasurable improvement over Lanczos under the provisional gates: "
        + f"**{'yes' if verdict['passed'] else 'no'}**"
        + ("" if verdict["passed"] else f" ({', '.join(verdict['failed'])})")
        + f". {evaluation['note']}\n"
    )


def stage_section(manifest: dict) -> str:
    return _table(
        ["Stage", "Class (PRD 5.2)", "Seconds", "Peak RSS MB", "Notes"],
        [
            (
                s["name"],
                s["classification"],
                f"{s['seconds']:.2f}",
                s["rss_peak_mb"] or "-",
                "; ".join(s["notes"]) or "-",
            )
            for s in manifest.get("stages", [])
        ],
    )


def _gate_text(gate: dict | None) -> str:
    if not gate:
        return "no SR (Lanczos input)"
    seams = gate.get("seams") or {}
    return f"{gate['mode']} SR {'passed' if gate['passed'] else 'FAILED'} (hard seam {seams.get('max_score', '-')})"


def _face_rows(evaluation: dict) -> str:
    rows = evaluation.get("faces") or []
    if not rows:
        return ""
    return "Face diagnostics (consistency with the source, never identity proof):\n\n" + _table(
        ["Face", "Tier", "Primary", "Box (source px)", "SSIM", "dE2000 mean"],
        [
            (
                r["face_index"],
                r["tier"],
                _value(r["primary"]),
                r["box"],
                _num(r["ssim"], 4),
                _num(r["color_difference_ciede2000"]["mean"], 3),
            )
            for r in rows
        ],
    )


def universal_conclusions(manifest: dict) -> list[str]:
    """Factual, automatically checkable statements; visual judgements are left to human review."""
    lines = []
    classification = manifest.get("scene", {}).get("classification")
    route = manifest.get("route", {})
    evaluation = manifest.get("evaluation")
    final = manifest.get("final_output", {})
    if manifest.get("status") != "complete":
        lines.append(f"The run did not complete ({manifest.get('status')}); conclusions cover finished stages only.")
    if classification:
        lines.append(
            f"Classified as {classification['category']} ({classification['confidence']} confidence) and processed "
            f"by the {classification['pipeline']}."
        )
        if classification.get("needs_confirmation"):
            lines.append(
                "Confidence was low: confirm or correct the category with analysis.manual_override and re-run."
            )
    for attempt in route.get("attempts", []):
        gate = attempt.get("gate")
        if gate and not gate["passed"]:
            lines.append(
                f"{gate['mode'].capitalize()} SR failed its seam gate in {attempt['run_id']}, so the scene was "
                "re-processed without SR; no SR result that failed a gate reached the final image."
            )
    selection = route.get("selection")
    if selection:
        lines.append(f"The common pipeline started from {selection['label']}: {selection['why']}.")
    if evaluation:
        verdict = evaluation["improvement"]
        gain = verdict["detail_gain"]
        if verdict["passed"]:
            lines.append(
                f"The final image measurably improves on Lanczos under the provisional gates: {gain:+.1%} detail "
                "energy without exceeding the source-consistency and clipping limits."
            )
        else:
            lines.append(
                f"The final image does not measurably improve on Lanczos under the provisional gates ({gain:+.1%} "
                f"detail energy; failed: {', '.join(verdict['failed'])})."
            )
    files = final.get("files") or {}
    if files:
        size = next(iter(files.values()))
        lines.append(
            f"Final image: {size['width']}x{size['height']} ({size['width'] * size['height'] / 1e6:.1f} MP), "
            f"{', '.join(files)}, verified by SHA-256."
        )
    lines.append(
        "Detail energy also rises with ringing, noise and seams: visual acceptance needs human review of the crops "
        "at 100% and 200%."
    )
    return lines


def review_figures(manifest: dict) -> list[tuple[str, str]]:
    """(caption, path) of every review figure of a run, the delegated specialised run's included."""
    figures = []
    if manifest.get("figures", {}).get("final"):
        figures.append(("final image (review JPEG)", manifest["figures"]["final"]))
    route = manifest.get("route", {})
    for panel in route.get("crop_panels") or []:
        figures.append((f"crop {panel['crop']} (output box {panel['box']})", panel["path"]))
    for key, caption in (("regions", "combined region masks"), ("fallbacks", "failure-rule fallback blocks")):
        if manifest.get("figures", {}).get(key):
            figures.append((caption, manifest["figures"][key]))
    delegate = route.get("delegate")
    if delegate and Path(delegate["manifest"]).exists():
        delegated = json.loads(Path(delegate["manifest"]).read_text())
        for name, record in (delegated.get("comparisons", {}).get("figures") or {}).items():
            if isinstance(record, dict) and record.get("path"):
                figures.append((record.get("description", name), record["path"]))
        for panel in delegated.get("crop_panels") or []:
            figures.append((f"night crop {panel['crop']} (output box {panel['box']})", panel["path"]))
        for key in ("banding", "halo", "regions", "fallbacks"):
            if (delegated.get("figures") or {}).get(key):
                figures.append((f"night {key}", delegated["figures"][key]))
    return [(caption, path) for caption, path in figures if Path(path).exists()]
