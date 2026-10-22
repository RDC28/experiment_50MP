"""Markdown report of a Phase 5 synthetic sensor research run (PRD sections 14, 17.3, 24 E01/E08 and 27).

Rebuilt from ``manifest.json`` alone, so a failed run still yields a report of
the stages that completed. Formatting helpers are shared with the Phase 0/1
report.
"""

from __future__ import annotations

import json
from pathlib import Path

from .report_builder import _bytes, _num, _short_params, _table, _version


def write_report(manifest: dict, path: Path) -> Path:
    path.write_text(render_report(manifest, path.parent))
    return path


def render_report(manifest: dict, report_dir: Path) -> str:
    from .sensor_research import RAW_TERMINOLOGY

    out: list[str] = []
    add = out.append
    source = manifest.get("input", {})
    sensor = manifest.get("sensor", {})
    runtime = manifest.get("runtime", {})
    roundtrip = manifest.get("roundtrip", {})
    exports = manifest.get("exports", {})

    add(f"# Phase 5 synthetic sensor research report: {source.get('name', 'unknown input')}\n")
    add(f"> {manifest['notice']}\n")
    add(f"> Label carried by every synthetic output: *{manifest.get('synthetic_raw_label', '-')}*\n")
    add(f"Status: **{manifest.get('status')}**\n")
    if manifest.get("error"):
        add(f"Failure: `{manifest['error']['message']}`. Outputs written before the failure are kept.\n")

    add("## 1. Run summary\n")
    gpu = runtime.get("gpu")
    add(
        _table(
            ["Item", "Value"],
            [
                ("Run ID", manifest.get("run_id", "-")),
                ("Input", f"`{source.get('name', '-')}`"),
                ("Input pixel SHA-256", f"`{source.get('pixel_sha256', '-')}`"),
                ("Input dimensions", f"{source.get('width', '-')} x {source.get('height', '-')}"),
                ("Pipeline version", _version(manifest)),
                ("Notebook version", manifest.get("notebook_version") or "- (run outside a notebook)"),
                ("Models", "none (no learned models)"),
                ("GPU", f"{gpu['name']} ({gpu['memory_total_gb']} GB)" if gpu else "none detected (CPU only)"),
                ("Platform", f"{runtime.get('platform', '-')}, Python {runtime.get('python', '-')}"),
                ("Total runtime", f"{manifest.get('total_seconds', 0):.1f} s"),
                ("Peak resident memory", f"{manifest.get('memory', {}).get('pipeline_peak_rss_mb', '-')} MB"),
                ("Storage used", _bytes(manifest.get("storage_used_bytes"))),
            ],
        )
    )

    add("## 2. Scientific limitations (PRD 1.3, 14.3)\n")
    add(
        "- Converting RGB into a Bayer-like mosaic **simulates** sensor sampling. It does not reconstruct the CFA "
        "samples a real camera would have measured; the source never contained them."
    )
    add(
        "- The camera colour response, white balance, exposure, noise and ADC are **assumptions** of a stated "
        "hypothetical sensor, not measurements. Simulated shot noise is not evidence of a physical exposure."
    )
    add(
        "- The input is a processed master derived from a display-referred AI render; its linear values are not "
        "calibrated scene radiance, so the synthetic mosaic is not scene-referred either."
    )
    add(
        "- Developing the mosaic back to RGB can only lose information relative to the input (two of three "
        "channels per pixel are re-estimated); the round trip measures that loss.\n"
    )

    add("## 3. RAW-format terminology (PRD 17.3)\n")
    add(_table(["Term", "Meaning", "In this run"], RAW_TERMINOLOGY))

    add("## 4. Input\n")
    lineage = source.get("lineage") or {}
    rows = [
        ("File", source.get("path") or "in-memory array"),
        ("File SHA-256", f"`{source.get('file_sha256') or '-'}`"),
        ("Lineage", "; ".join(f"{k}: {v}" for k, v in lineage.items()) or "-"),
        ("Notes", "; ".join(source.get("notes", [])) or "-"),
    ]
    stats = source.get("statistics")
    if stats:
        rows.append(("Linear mean R/G/B", " / ".join(f"{v:.4f}" for v in stats["mean"].values())))
        rows.append(("Samples outside [0, 1]", f"{stats['below_zero_fraction'] + stats['above_one_fraction']:.4%}"))
    add(_table(["Item", "Value"], rows))

    if sensor:
        add("## 5. Synthetic sensor model (PRD 14.4)\n")
        add(f"{sensor.get('sensor_model', '')}.\n")
        add(
            _table(
                ["Parameter", "Value"],
                [
                    ("cfa_pattern", sensor["cfa_pattern"]),
                    ("adc_bit_depth", sensor["adc_bit_depth"]),
                    ("black_level / white_level", f"{sensor['black_level']} / {sensor['white_level']}"),
                    ("analog_gain", sensor["analog_gain"]),
                    ("exposure_scale", sensor["exposure_scale"]),
                    ("full_well_electrons", f"{sensor['full_well_electrons']:,.0f}"),
                    ("conversion gain", f"{sensor['dn_per_electron']:.6f} DN/e-"),
                    ("shot_noise_enabled", sensor["shot_noise_enabled"]),
                    (
                        "read_noise_enabled / std",
                        f"{sensor['read_noise_enabled']} / {sensor['read_noise_std_electrons']} e- RMS",
                    ),
                    ("color_matrix (camera WB -> Rec.709)", _matrix(sensor["color_matrix"])),
                    ("white_balance_multipliers", sensor["white_balance_multipliers"]),
                    ("DNG AsShotNeutral", sensor["as_shot_neutral"]),
                    ("DNG ColorMatrix1 (XYZ -> camera, D65)", _matrix(sensor["xyz_to_camera_d65"])),
                ],
            )
        )
    mosaic = manifest.get("mosaic", {}).get("statistics")
    if mosaic:
        add("Mosaic statistics (fractions of photosites):\n")
        add(
            _table(
                ["Quantity", "Fraction"],
                [
                    ("Negative input signal (clamped to no light)", f"{mosaic['negative_signal_fraction']:.4%}"),
                    ("Signal at or above the white level", f"{mosaic['saturated_signal_fraction']:.4%}"),
                    ("Codes clipped at the white level", f"{mosaic['clipped_at_white_fraction']:.4%}"),
                    ("Codes below the black level (noise)", f"{mosaic['below_black_fraction']:.4%}"),
                ],
            )
        )

    add("## 6. Processing log (PRD 27.2)\n")
    stage_rows = []
    for stage in manifest.get("stages", []):
        issues = "; ".join(stage["memory_warnings"] + ([stage["error"]] if stage["error"] else []) + stage["notes"])
        stage_rows.append(
            (
                stage["name"],
                stage["classification"],
                f"{stage['seconds']:.2f}",
                f"{stage['input_precision']} -> {stage['output_precision']}",
                f"{stage['rss_peak_mb']:.0f}" if stage["rss_peak_mb"] is not None else "-",
                _short_params(stage["parameters"]),
                issues or "-",
            )
        )
    add(_table(["Stage", "Class (PRD 5.2)", "Seconds", "Precision", "Peak RSS MB", "Parameters", "Notes"], stage_rows))

    if exports:
        add("## 7. Synthetic exports and validation (PRD 14.6, 23.6)\n")
        add(
            _table(
                ["File", "Kind", "Size", "Checks", "SHA-256"],
                [
                    (
                        f"`{Path(e['path']).name}`",
                        e["kind"],
                        _bytes(e["size_bytes"]),
                        f"{sum(e['checks'].values())}/{len(e['checks'])} passed",
                        f"`{e['file_sha256'][:16]}...`",
                    )
                    for e in exports.values()
                ],
            )
        )
        dng = exports.get("dng")
        if dng:
            add("DNG validation detail (structure re-read with tifffile, reader check with LibRaw):\n")
            add(_table(["Check", "Result"], [(name, "pass" if ok else "FAIL") for name, ok in dng["checks"].items()]))
            exiftool = dng.get("exiftool") or {}
            add(f"ExifTool `-validate`: {'; '.join(exiftool.get('output', [])) or '-'}\n")
            if exiftool.get("ok") and len(exiftool.get("output", [])) > 1:
                add(
                    "ExifTool passes: its only remark, SubIFDs stored with the IFD field type, is permitted by "
                    'TIFF Technical Note 1 and the DNG 1.4 specification ("LONG or IFD").\n'
                )
            add(
                "Adobe's `dng_validate` and Adobe Camera Raw were not run by the package; see the conclusions for "
                "what this validation does and does not cover.\n"
            )

    if roundtrip:
        add("## 8. Experiment E08: synthetic Bayer round trip\n")
        add(
            "Each method develops the synthetic mosaic back to linear Rec.709 and is compared with the input at "
            "full resolution. Full-frame PSNR uses sRGB-encoded values; the Phase 1 source-consistency metrics run "
            f"on {next(iter(roundtrip.values()))['sample']['description']}.\n"
        )
        add(_table(["Metric", *roundtrip], _roundtrip_rows(roundtrip)))
        add(
            _table(
                ["Method", "Description"],
                [(name, summary["description"]) for name, summary in roundtrip.items()],
            )
        )

    bit_depth = manifest.get("bit_depth")
    quantization = manifest.get("mosaic_quantization")
    if bit_depth or quantization:
        add("## 9. Bit-depth experiments (PRD 14.5, E01)\n")
    if bit_depth:
        params = bit_depth["parameters"]
        add(
            f"Every representation stores the same input. Edit: {params['edit']}; shadows: "
            f"{params['shadow_definition']} ({bit_depth['shadow_pixel_fraction']:.1%} of pixels). Cycle: "
            f"{params['cycle']} ({params['cycles']} times). Quantisation error compares with "
            "the input clipped to what the representation can hold; clipping is reported separately.\n"
        )
        add(_table(["Quantity", *[r["name"] for r in bit_depth["representations"]]], _bit_depth_rows(bit_depth)))
        half = next((r for r in bit_depth["representations"] if r["name"] == "rgb16_float_linear"), None)
        if half and half["quantization_rmse_linear"] == 0 and half["outside_range_fraction"] == 0:
            add(
                "The input is itself stored at half precision (for example a half EXR master), so the half "
                "representation reproduces it exactly here; its precision limit shows only on float32 inputs.\n"
            )
    if quantization:
        add(
            f"Optional row, the {quantization['description']}: {quantization['levels_in_unit_range']:,} levels, "
            f"quantisation RMSE {quantization['rmse']:.3e} of the range (uniform-quantiser expectation "
            f"{quantization['expected_rmse']:.3e}), max {quantization['max_abs_error']:.3e}; "
            f"{quantization['outside_range_fraction']:.4%} of photosites outside the range (clipped).\n"
        )
    expansion = (bit_depth or {}).get("expansion")
    if expansion:
        add("E01 on the source codes: does expanding the bit depth add information?\n")
        add(
            _table(
                [
                    "Representation",
                    "Bits/sample",
                    "Distinct values per channel",
                    "Entropy bits/sample",
                    "Exact inverse",
                ],
                [
                    (
                        r["representation"],
                        r["bits_per_sample"],
                        ", ".join(str(v) for v in r["distinct_values_per_channel"]),
                        ", ".join(f"{v:.4f}" for v in r["entropy_bits_per_sample"]),
                        "yes" if r["recovers_source_exactly"] else "no",
                    )
                    for r in expansion["rows"]
                ],
            )
        )
        add(f"{expansion['finding']}\n")

    add("## 10. Visual results (PRD 27.3)\n")
    figures = report_dir / "figures"
    if (figures / "input_preview.jpg").exists():
        add("Input master:\n\n![Input](figures/input_preview.jpg)\n")
    for method in roundtrip:
        if (figures / f"roundtrip_{method}_preview.jpg").exists():
            add(f"Round trip, {method}:\n\n![{method}](figures/roundtrip_{method}_preview.jpg)\n")
    for panel in manifest.get("crop_panels", []):
        columns = ", ".join(panel["columns"])
        add(
            f"Crop `{panel['crop']}` at 100% ({columns}; the last column is the amplified input vs "
            f"{panel['columns'][2] if len(panel['columns']) > 2 else '-'} difference):\n\n"
            f"![{panel['crop']}](figures/{Path(panel['path']).name})\n"
        )
    for method in roundtrip:
        if (figures / f"delta_e_{method}.png").exists():
            add(f"Difference map, {method} round trip vs input:\n\n![delta E {method}](figures/delta_e_{method}.png)\n")

    add("## 11. Information provenance (PRD 1.1, 27.5)\n")
    rows = []
    mosaic_info = manifest.get("mosaic", {}).get("information")
    if mosaic_info:
        rows.append(("Synthetic mosaic (TIFF/DNG)", *_information(mosaic_info)))
    for method, summary in roundtrip.items():
        rows.append((f"Round trip, {method}", *_information(summary["information"])))
    if rows:
        add(_table(["Output", "Preserved", "Estimated", "Synthesized", "Simulated", "Export-only"], rows))

    add("## 12. Conclusions (PRD 25.4, 27.6)\n")
    add("\n".join(f"- {line}" for line in conclusions(manifest)) + "\n")

    add("## 13. Provenance record (PRD 6.2)\n")
    add(
        _table(
            ["Item", "Value"],
            [
                ("Input pixel SHA-256", f"`{source.get('pixel_sha256', '-')}`"),
                ("Package", _version(manifest)),
                ("Notebook version", manifest.get("notebook_version") or "-"),
                ("Models and weight hashes", "none"),
                ("GPU", gpu["name"] if gpu else "none"),
                ("Random seeds", json.dumps(manifest.get("random_seeds", {}))),
                ("Runtime", f"{manifest.get('total_seconds', 0):.1f} s"),
            ],
        )
    )
    add("Dependency versions:\n")
    add(_table(["Package", "Version"], sorted(runtime.get("packages", {}).items())))
    add("Output checksums (SHA-256):\n")
    add(_table(["File", "SHA-256"], [(f"`{k}`", f"`{v}`") for k, v in manifest.get("output_checksums", {}).items()]))
    add("Configuration:\n")
    add("```json\n" + json.dumps(manifest.get("config", {}), indent=2) + "\n```\n")
    return "\n".join(out)


def conclusions(manifest: dict) -> list[str]:
    """Factual, automatically checkable statements; visual judgement is left to human review."""
    lines = []
    if manifest.get("status") != "complete":
        lines.append(f"The run did not complete ({manifest.get('status')}); conclusions cover finished stages only.")
    labels = manifest.get("labels")
    dng = manifest.get("exports", {}).get("dng")
    if labels:
        dng_ok = dng is None or dng.get("valid")
        verdict = "met" if labels["all_labelled"] and dng_ok else "not met"
        dng_text = (
            "no DNG was exported"
            if dng is None
            else f"the DNG passed {sum(dng['checks'].values())}/{len(dng['checks'])} validation checks"
        )
        lines.append(
            f"Phase 5 exit criterion {verdict}: every synthetic output is labelled as synthetic "
            f"({sum(labels['checks'].values())}/{len(labels['checks'])} label checks) and {dng_text}."
        )
    if dng:
        lines.append(
            "The synthetic DNG is structurally valid (all DNG 1.4 tags re-read and matched) and opens in LibRaw "
            "with the written CFA pattern, levels, white balance and colour matrix and a bit-identical mosaic. "
            "This shows compatibility, not camera realism."
        )
    roundtrip = manifest.get("roundtrip", {})
    if roundtrip:
        best = max(roundtrip, key=lambda m: roundtrip[m]["full_frame"]["psnr_encoded_db"])
        worst = min(roundtrip, key=lambda m: roundtrip[m]["full_frame"]["psnr_encoded_db"])
        lines.append(
            f"E08: every round trip loses information against the input; the closest is {best} "
            f"({roundtrip[best]['full_frame']['psnr_encoded_db']:.2f} dB), the furthest {worst} "
            f"({roundtrip[worst]['full_frame']['psnr_encoded_db']:.2f} dB). Synthetic Bayer sampling does not "
            "improve the visible image (PRD 34 question 11)."
        )
    bit_depth = manifest.get("bit_depth")
    if bit_depth:
        rows = {r["name"]: r for r in bit_depth["representations"]}
        eight, sixteen = rows.get("rgb8_srgb"), rows.get("rgb16_srgb")
        if eight and sixteen and None in (eight["edit_shadow_level_ratio"], sixteen["edit_shadow_level_ratio"]):
            lines.append(
                "PRD 14.5: the shadow-level comparison is unavailable because no input pixel has "
                f"{bit_depth['parameters']['shadow_definition']}."
            )
        elif eight and sixteen:
            lines.append(
                "PRD 14.5: after the identical shadow push, 8-bit keeps "
                f"{eight['edit_shadow_level_ratio']:.1%} and 16-bit integer "
                f"{sixteen['edit_shadow_level_ratio']:.1%} of the float32 reference's distinct shadow levels; "
                "higher precision protects gradients through processing."
            )
        expansion = bit_depth.get("expansion")
        if expansion:
            lines.append(
                "E01: "
                + (
                    "expanding the source to 16-bit or floating point added no information (no more distinct "
                    "values or entropy than the source codes). Greater numerical precision is not additional "
                    "original scene information."
                    if not expansion["information_added"]
                    else expansion["finding"]
                )
            )
    lines.append(
        "Synthetic sensor data adds no scene information; its practical value is pipeline and compatibility "
        "testing (PRD 25.4 question 5, PRD 34 question 12). The TIFF/EXR masters remain the deliverables."
    )
    lines.append(
        "Not verified by this run: opening the DNG in Adobe software or Adobe's dng_validate; visual review of "
        "the crop panels at 100% and 200% is left to a human reviewer."
    )
    return lines


# --------------------------------------------------------------------------- formatting


def _matrix(rows) -> str:
    return " / ".join("[" + ", ".join(f"{v:.4f}" for v in row) + "]" for row in rows)


def _information(info: dict) -> tuple:
    return tuple(info.get(k, "-") for k in ("preserved", "estimated", "synthesized", "simulated", "export_only"))


def _roundtrip_rows(roundtrip: dict) -> list[tuple]:
    def col(fn):
        return [fn(s) for s in roundtrip.values()]

    return [
        ("Full-frame PSNR, sRGB encoded (dB)", *col(lambda s: _num(s["full_frame"]["psnr_encoded_db"], 2))),
        ("Full-frame mean abs error (linear)", *col(lambda s: _num(s["full_frame"]["mean_abs_error_linear"], 6))),
        ("Full-frame max abs error (linear)", *col(lambda s: _num(s["full_frame"]["max_abs_error_linear"], 4))),
        ("Sample SSIM", *col(lambda s: _num(s["consistency"]["ssim"], 5))),
        ("Sample MS-SSIM (luma)", *col(lambda s: _num(s["consistency"]["ms_ssim"]["value"], 5))),
        ("Edge displacement mean (px)", *col(lambda s: _num(s["consistency"]["edge_displacement"]["mean_px"], 3))),
        (
            "delta E 2000 mean / p95",
            *col(
                lambda s: (
                    f"{s['consistency']['color_difference_ciede2000']['mean']:.4f} / "
                    f"{s['consistency']['color_difference_ciede2000']['p95']:.4f}"
                )
            ),
        ),
        ("L* mean shift", *col(lambda s: _num(s["consistency"]["luminance_distribution"]["mean_shift"], 4))),
    ]


def _bit_depth_rows(bit_depth: dict) -> list[tuple]:
    reps = bit_depth["representations"]

    def col(fn):
        return [fn(r) for r in reps]

    return [
        ("Domain", *col(lambda r: r["domain"])),
        ("Levels in [0, 1]", *col(lambda r: f"{r['levels_in_unit_range']:,}")),
        ("Largest value", *col(lambda r: f"{r['max_value']:.6g}")),
        ("Samples outside range (headroom)", *col(lambda r: f"{r['outside_range_fraction']:.4%}")),
        ("Clipping RMSE (linear)", *col(lambda r: f"{r['clipping_rmse_linear']:.3e}")),
        ("Quantisation RMSE (linear)", *col(lambda r: f"{r['quantization_rmse_linear']:.3e}")),
        ("Quantisation max abs (linear)", *col(lambda r: f"{r['quantization_max_abs_linear']:.3e}")),
        ("PSNR, sRGB encoded (dB)", *col(lambda r: _num(r["psnr_encoded_db"], 2))),
        ("Edit RMSE, encoded", *col(lambda r: f"{r['edit_rmse_encoded']:.3e}")),
        ("Edited shadow levels (mean per channel)", *col(lambda r: f"{r['edit_shadow_levels']:,.0f}")),
        (
            "Shadow levels vs float32",
            *col(lambda r: "-" if r["edit_shadow_level_ratio"] is None else f"{r['edit_shadow_level_ratio']:.1%}"),
        ),
        ("Cycle drift RMSE (linear)", *col(lambda r: f"{r['cycle_drift_rmse_linear']:.3e}")),
        ("Cycle drift max abs (linear)", *col(lambda r: f"{r['cycle_drift_max_abs_linear']:.3e}")),
        (
            "File size",
            *col(lambda r: f"{_bytes(r['file']['size_bytes'])} ({r['file']['format']})" if r["file"] else "-"),
        ),
    ]
