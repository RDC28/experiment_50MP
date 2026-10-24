"""Markdown reports of the Phase 6 ablation and reference-test runs (PRD 27), written next to their manifests."""

from __future__ import annotations

import os
from pathlib import Path

from .run_collection import fmt, markdown_table

LIMITATIONS = (
    "- Source consistency (the output reduced back to the source size and compared with it) shows agreement with "
    "the source; it cannot show that detail above the source sampling limit is correct (PRD 23.1).",
    "- Detail energy (variance of the Laplacian of sRGB luma) rises with real detail and equally with ringing, "
    "noise, grain and seams; it measures change, not correctness.",
    "- Grain, bloom and the sensor round trip are simulations; they recover nothing (PRD 13, 14, 16).",
    "- Every threshold is provisional, to be calibrated experimentally (PRD 25.3).",
)


def _rel(path: str | None, folder: Path) -> str | None:
    if not path:
        return None
    try:
        return os.path.relpath(path, folder)
    except ValueError:
        return path


def configuration_table(records: list[dict], reference: bool = False) -> tuple[list[str], list[tuple]]:
    header = ["configuration", "stages", "total s", "peak RSS MB", "detail vs Lanczos", "source PSNR dB", "source dE00"]
    if reference:
        header += ["truth PSNR dB", "truth SSIM", "gain over Lanczos dB", "above-limit error (hf NMSE)", "hf corr."]
    rows = []
    for r in records:
        if "measurements" not in r:
            reason = r.get("skipped") or r.get("error") or "not run"
            rows.append((r["label"], ", ".join(r["stages"]), f"skipped: {reason}", *["-"] * (len(header) - 3)))
            continue
        m, vs = r["measurements"], r.get("vs_lanczos", {})
        gain = vs.get("detail_gain")
        row = [
            r["label"],
            ", ".join(r["stages"]),
            fmt(r["seconds"]["total"], 1),
            fmt(r.get("peak_rss_mb"), 0),
            f"{gain:+.1%}" if gain is not None else "-",
            fmt(m["source_consistency"]["psnr_db"]),
            fmt(m["source_consistency"]["delta_e_mean"], 3),
        ]
        if reference:
            ref = r.get("reference") or {}
            psnr_gain = vs.get("reference_psnr_db")
            row += [
                fmt(ref.get("psnr_db")),
                fmt(ref.get("ssim"), 4),
                f"{psnr_gain:+.2f}" if psnr_gain is not None else "-",
                fmt(ref.get("hf_nmse"), 3),
                fmt(ref.get("hf_correlation"), 3),
            ]
        rows.append(tuple(row))
    return header, rows


def contribution_table(contributions: list[dict]) -> tuple[list[str], list[tuple]]:
    header = ["stage", "compared with", "detail change", "source PSNR change dB", "truth PSNR change dB", "verdict"]
    rows = []
    for c in contributions:
        truth = c.get("reference_psnr_db")
        if truth is not None:
            verdict = (
                "helps accuracy" if truth > 0.05 else "harms accuracy" if truth < -0.05 else "no measurable effect"
            )
        else:
            verdict = "accuracy not measured (no reference)"
        gain = c.get("detail_gain")
        rows.append(
            (
                c["stage"],
                f"`{c['without']}`",
                f"{gain:+.1%}" if gain is not None else "-",
                f"{c['source_psnr_db']:+.2f}",
                f"{truth:+.2f}" if truth is not None else "-",
                verdict,
            )
        )
    return header, rows


def _commit(manifest: dict) -> str:
    return ((manifest.get("package") or {}).get("git") or {}).get("commit", "-")[:12]


def write_ablation_report(manifest: dict, path: Path) -> Path:
    path = Path(path)
    folder = path.parent
    lines = [f"# Phase 6 ablation: {Path(manifest.get('source', {}).get('original_path', 'source')).name}", ""]
    lines += [f"> {manifest['notice']}", "", f"Status: **{manifest['status']}**", ""]
    crop = manifest.get("crop", {})
    scene = manifest.get("scene", {})
    lines += ["## 1. Run summary", ""]
    lines += [
        markdown_table(
            ["item", "value"],
            [
                ("run", f"`{manifest.get('run_id', '-')}`"),
                ("source SHA-256", f"`{manifest.get('source', {}).get('sha256', '-')}`"),
                (
                    "scene",
                    f"{scene.get('kind', '-')} ({'automatic' if scene.get('chosen_by') == 'auto' else 'configured'})",
                ),
                ("crop (source px)", f"{crop.get('box', '-')} centred on the {crop.get('centre_basis', '-')}"),
                ("output size", "x".join(str(v) for v in crop.get("output_size", []) or ["-"])),
                ("scale", " x ".join(f"{v:.4f}" for v in crop.get("scale", [])) or "-"),
                ("GPU", (manifest.get("runtime", {}).get("gpu") or {}).get("name", "none")),
                ("total runtime", f"{manifest.get('total_seconds', 0):.1f} s"),
                ("peak resident memory", f"{manifest.get('memory', {}).get('pipeline_peak_rss_mb', 0):.0f} MB"),
                (
                    "package",
                    f"{manifest['package']['version']}, commit {_commit(manifest)}",
                ),
            ],
        ),
        "",
        crop.get("note", ""),
        "",
        "## 2. Scientific limitations",
        "",
        *LIMITATIONS,
        "",
    ]
    if "error" in manifest:
        lines += ["## Failure", "", f"Stage `{manifest['error']['stage']}`: {manifest['error']['message']}", ""]
    records = manifest.get("configurations", [])
    if records:
        lines += ["## 3. Configurations (PRD 31 matrix, E02, E07, E09)", ""]
        lines += [markdown_table(*configuration_table(records)), ""]
        lines += [
            "Seconds count the shared upscale in every configuration (its standalone runtime). Detail is the change "
            "in detail energy over Lanczos; source PSNR and CIEDE2000 compare the output reduced to the source size "
            "with the source.",
            "",
        ]
    if manifest.get("contributions"):
        lines += ["## 4. Stage contributions (E09: full pipeline minus one stage)", ""]
        lines += [markdown_table(*contribution_table(manifest["contributions"])), ""]
        lines += [
            "Without a ground-truth reference this table can only show what each stage changes, not whether the "
            "change is correct; the reference test (E10) answers that.",
            "",
        ]
    sweep = manifest.get("performance_sweep") or []
    if sweep:
        lines += ["## 5. SR settings sweep (PRD 28.3)", ""]
        header = [
            "setting",
            "seconds",
            "inference s",
            "tile",
            "overlap",
            "precision",
            "peak GPU MB",
            "seam gate",
            "max diff (8-bit)",
            "PSNR vs default dB",
        ]
        rows = [
            (
                s.get("setting", "-"),
                fmt(s.get("seconds"), 1),
                fmt(s.get("inference_seconds"), 1),
                fmt(s.get("tile_size")),
                fmt(s.get("tile_overlap")),
                fmt(s.get("precision")),
                fmt(s.get("peak_device_mb"), 0),
                fmt(s.get("seam_gate_passed")) if s.get("seam_gate_passed") is not None else "-",
                fmt(s.get("max_abs_difference_8bit"), 2),
                fmt(s.get("psnr_vs_default_db")),
            )
            if "skipped" not in s and "error" not in s
            else (s.get("setting", "-"), s.get("skipped") or s.get("error"), *["-"] * 8)
            for s in sweep
        ]
        lines += [markdown_table(header, rows), ""]
    bit_depth = manifest.get("bit_depth")
    if bit_depth:
        lines += ["## 6. Bit depth (E01, PRD 14.5)", ""]
        study = bit_depth["study"]
        lines += [
            markdown_table(
                [
                    "representation",
                    "quantisation RMSE (linear)",
                    "outside range",
                    "shadow levels after +3 stops",
                    "cycle drift RMSE",
                ],
                [
                    (
                        r["name"],
                        f"{r['quantization_rmse_linear']:.2e}",
                        f"{r['outside_range_fraction']:.3%}",
                        fmt(r["edit_shadow_levels"], 1),
                        f"{r['cycle_drift_rmse_linear']:.2e}",
                    )
                    for r in study["representations"]
                ],
            ),
            "",
            f"Measured on the `{bit_depth['image']}` reconstruction of the crop; {study['parameters']['edit']}.",
        ]
        if bit_depth.get("expansion"):
            lines += ["", f"E01 on the source codes: {bit_depth['expansion']['finding']}"]
        lines += [""]
    figures = manifest.get("figures") or {}
    if figures.get("review_sheet"):
        lines += ["## 7. Visual review (PRD 26)", ""]
        lines += [f"![review sheet]({_rel(figures['review_sheet'], folder)})", ""]
        lines += [
            f"Blind sheet (PRD 26.3), configurations in shuffled order: `{_rel(figures['blind_sheet'], folder)}`; the "
            "key is in `figures/blind_key.json`. Score naturalness, structural fidelity, perceived detail, artefact "
            "severity and overall preference before opening the key.",
            "",
        ]
    lines += ["## 8. Conclusions", ""]
    lines += [f"- {line}" for line in ablation_conclusions(manifest)]
    path.write_text("\n".join(lines) + "\n")
    return path


def ablation_conclusions(manifest: dict) -> list[str]:
    records = {r["name"]: r for r in manifest.get("configurations", []) if "measurements" in r}
    out = []
    skipped = [r["name"] for r in manifest.get("configurations", []) if "measurements" not in r]
    if skipped:
        out.append(f"Not run: {', '.join(skipped)} (see the configuration table for why).")
    for name in ("sr_conservative", "sr_balanced", "sr_aggressive"):
        if name in records and "vs_lanczos" in records[name]:
            vs = records[name]["vs_lanczos"]
            gate = records[name].get("seam_gate_passed")
            out.append(
                f"{records[name]['label']}: detail {vs['detail_gain']:+.1%} over Lanczos, source PSNR "
                f"{vs['source_psnr_db']:+.2f} dB"
                + ("" if gate is None else f", seam gate {'pass' if gate else 'FAIL'}")
                + "."
            )
    for c in manifest.get("contributions", []):
        if c.get("detail_gain") is not None:
            out.append(f"{c['stage']}: changes detail energy by {c['detail_gain']:+.1%} in the full pipeline.")
    if not out:
        out.append("No configuration completed.")
    out.append("Visual acceptance (PRD 25.2) needs the review sheet at 100% and the blind sheet scores.")
    return out


def write_reference_report(manifest: dict, path: Path) -> Path:
    from .reference_test import summary_rows

    path = Path(path)
    folder = path.parent
    lines = ["# Phase 6 controlled reference test (E10)", "", f"> {manifest['notice']}", ""]
    lines += [f"Status: **{manifest['status']}**", "", "## 1. References and licences", ""]
    lines += [
        markdown_table(
            ["reference", "photograph", "author", "licence", "size", "crop used"],
            [
                (
                    r["name"],
                    f"[{r['title'].removeprefix('File:')}]({r['page']})",
                    r["author"],
                    f"[{r['licence']}]({r['licence_url']})" if r.get("licence_url") else r["licence"],
                    f"{r['width']} x {r['height']}",
                    str(r.get("crop_box", "-")),
                )
                for r in manifest.get("references", [])
            ],
        ),
        "",
        "Each photograph was fetched at run time from Wikimedia Commons, verified against its pinned SHA-1, and is "
        "used here only to measure reconstruction accuracy. Attribution as above; the degraded copies and "
        "reconstructions in this run folder are adaptations under the same licence.",
        "",
        "## 2. Method",
        "",
        "Each reference (centre crop) is degraded to 1/scale of its size, then reconstructed to its own size by "
        "every configuration and compared with it. PSNR, SSIM and CIEDE2000 are full-reference accuracy. The luma "
        "is also split at the source sampling limit: the low band is what the degraded source could hold; "
        "`hf NMSE` is the error above it relative to that band's energy in the truth (1.0 is what adding no detail "
        "scores, below 1 is real detail recovered, above 1 is added detail that is wrong more than right) and "
        "`hf corr.` its correlation with the truth.",
        "",
    ]
    if "error" in manifest:
        lines += ["## Failure", "", f"Stage `{manifest['error']['stage']}`: {manifest['error']['message']}", ""]
    for index, case in enumerate(manifest.get("cases", []), start=3):
        steps = "; ".join(case["degradation"]["steps"])
        lines += [
            f"## {index}. {case['reference']}, {case['degradation']['model']} degradation, x{case['scale']:g}",
            "",
        ]
        lines += [
            f"Source {case['source_size'][0]} x {case['source_size'][1]} -> {case['output_size'][0]} x "
            f"{case['output_size'][1]}; faces observed: {case['faces']}. Degradation: {steps}.",
            "",
            markdown_table(*configuration_table(case["configurations"], reference=True)),
            "",
        ]
        if case.get("figures", {}).get("review_sheet"):
            lines += [f"![{case['key']}]({_rel(case['figures']['review_sheet'], folder)})", ""]
    rows = summary_rows(manifest)
    if rows:
        lines += ["## Summary", ""]
        lines += [f"- {line}" for line in reference_conclusions(rows)]
    path.write_text("\n".join(lines) + "\n")
    return path


def reference_conclusions(rows: list[dict]) -> list[str]:
    out = []
    cases = sorted({r["case"] for r in rows})
    for case in cases:
        entries = [r for r in rows if r["case"] == case]
        best = max(entries, key=lambda r: r["psnr_db"])
        lanczos = next((r for r in entries if r["configuration"] == "lanczos"), None)
        gain = f"{best['psnr_db'] - lanczos['psnr_db']:+.2f} dB over Lanczos" if lanczos else "no Lanczos row"
        nmse = f", above-limit error {best['hf_nmse']:.3f}" if best.get("hf_nmse") is not None else ""
        out.append(f"{case}: most accurate is {best['label']} ({best['psnr_db']:.2f} dB, {gain}{nmse}).")
    return out
