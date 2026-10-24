"""Markdown run report and figures (PRD section 27).

The report is rebuilt from ``manifest.json`` alone, so a partial manifest from
a failed run still produces a readable report of what completed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import baselines
from . import color_management as cm
from .exporters import write_review_image


def write_preview(encoded16: np.ndarray, path: Path, max_edge: int) -> Path:
    """Downscale a 16-bit sRGB image in linear light and write an 8-bit review JPEG."""
    height, width = encoded16.shape[:2]
    scale = min(1.0, max_edge / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    preview = np.empty((size[1], size[0], 3), dtype=np.float32)
    for channel in range(3):
        plane = cm.srgb_uint16_to_linear(encoded16[..., channel])[..., None]
        preview[..., channel] = baselines.downsample_area(plane, *size)[..., 0]
        del plane
    encoded, _ = cm.encode_srgb_uint16(preview)
    return write_review_image(path, cm.srgb_uint16_to_uint8(encoded))


def histogram_figure(histograms: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4", "Y": "#555555", "luma": "#111111"}
    bins = np.arange(histograms["bins"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), dpi=110)
    for name in ("R", "G", "B", "Y"):
        if name in histograms:
            axes[0].plot(bins, np.asarray(histograms[name]) + 1, color=colors[name], linewidth=1, label=name)
    axes[1].fill_between(bins, np.asarray(histograms["luma"]) + 1, 1, color=colors["luma"], alpha=0.6, linewidth=0)
    axes[0].set_title("Channel histograms (encoded values)")
    axes[1].set_title("Luma histogram")
    for ax in axes:
        ax.set_xlim(0, histograms["bins"] - 1)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1)
        ax.set_xlabel("code value (256 bins)")
        ax.set_ylabel("pixels + 1")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def heatmap_figure(values: np.ndarray, path: Path, title: str, max_edge: int, *, label: str = "delta E 2000") -> Path:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    height, width = values.shape
    scale = min(1.0, max_edge / max(width, height))
    if scale < 1.0:
        values = cv2.resize(
            values, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
        )
    upper = max(float(np.percentile(values, 99.5)), 1.0)
    fig, ax = plt.subplots(figsize=(10, 10 * values.shape[0] / values.shape[1]), dpi=200)
    image = ax.imshow(values, cmap="inferno", vmin=0, vmax=upper, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    colorbar_axes = make_axes_locatable(ax).append_axes("right", size="2.5%", pad=0.1)
    fig.colorbar(image, cax=colorbar_axes, label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def write_report(manifest: dict, path: Path) -> Path:
    path.write_text(render_report(manifest, path.parent))
    return path


def render_report(manifest: dict, report_dir: Path) -> str:
    out: list[str] = []
    add = out.append
    source = manifest.get("source", {})
    target = manifest.get("target_resolution", {})
    runtime = manifest.get("runtime", {})
    baselines_done = manifest.get("baselines", {})
    sr_record = manifest.get("super_resolution") or {}
    methods = {**baselines_done, **{f"sr_{m}": s for m, s in sr_record.get("modes", {}).items()}}
    title = "Phase 0 to 2 report" if sr_record else "Phase 0/1 baseline report"

    add(f"# {title}: {Path(source.get('original_path', 'unknown source')).name}\n")
    add(f"> {manifest['notice']}\n")
    add(f"Status: **{manifest.get('status')}**\n")
    if manifest.get("error"):
        add(f"Failure: `{manifest['error']['message']}`. Outputs written before the failure are kept.\n")

    add("## 1. Run summary\n")
    gpu = runtime.get("gpu")
    rows = [
        ("Run ID", manifest.get("run_id", "-")),
        ("Source", f"`{Path(source.get('original_path', '-')).name}`"),
        ("Source SHA-256", f"`{source.get('sha256', '-')}`"),
        ("Input dimensions", _dims(source.get("width"), source.get("height"))),
        (
            "Output dimensions",
            _dims(target.get("width"), target.get("height")) + (f" ({target['megapixels']:.3f} MP)" if target else ""),
        ),
        ("Pipeline version", _version(manifest)),
        ("Notebook version", manifest.get("notebook_version") or "- (run outside a notebook)"),
        ("Models", _models_text(manifest)),
        ("GPU", f"{gpu['name']} ({gpu['memory_total_gb']} GB)" if gpu else "none detected (CPU only)"),
        ("Platform", f"{runtime.get('platform', '-')}, Python {runtime.get('python', '-')}"),
        ("Final image", _final_summary(manifest.get("final_output"))),
        ("Total runtime", f"{manifest.get('total_seconds', 0):.1f} s"),
        ("Peak resident memory", f"{manifest.get('memory', {}).get('pipeline_peak_rss_mb', '-')} MB"),
        ("Storage used", _bytes(manifest.get("storage_used_bytes"))),
    ]
    add(_table(["Item", "Value"], rows))

    add("## 2. Scientific limitations\n")
    add(
        "- The source is a rendered, display-referred image. Linearising it gives linear-light display-referred "
        "values, not scene-referred radiance; the EXR master is synthetic linear RGB."
    )
    add(
        "- Interpolation estimates samples between source pixels. It adds no information beyond the source "
        "sampling limit; every output pixel is an estimate."
    )
    if sr_record:
        add(
            "- Pretrained super-resolution predicts detail above the source sampling limit from priors learned on "
            "other photographs. That detail is synthesised, not recovered, and may be wrong even when it looks "
            "plausible; a sharper result is not accepted for looking sharper (PRD 10.5)."
        )
    add("- 16-bit and floating-point encodings add numerical precision, not dynamic range or recovered detail.")
    add(
        "- Source-consistency metrics show agreement with the source after reduction; they cannot prove that "
        "detail above the source resolution is correct.\n"
    )

    validation = manifest.get("validation")
    if validation:
        add("## 3. Input validation (PRD 3.4)\n")
        add(_table(["Check", "Result"], _validation_rows(validation)))
        if validation.get("warnings"):
            add("Warnings:\n")
            add("\n".join(f"- {w}" for w in validation["warnings"]) + "\n")
        add("![Source histograms](figures/source_histograms.png)\n")

    if target:
        add("## 4. Target resolution (PRD 3.5)\n")
        add(
            _table(
                ["Quantity", "Value"],
                [
                    ("Mode", target["mode"]),
                    ("Requested pixels", f"{target['target_pixels']:,}"),
                    ("Achieved pixels", f"{target['pixels']:,}"),
                    ("Pixel-count error", f"{target['pixel_count_error']:+.4%}"),
                    ("Aspect-ratio error", f"{target['aspect_ratio_error']:+.5%}"),
                    ("Nominal linear scale", f"{target['nominal_scale']:.5f}"),
                    ("Scale x / y", f"{target['scale_x']:.5f} / {target['scale_y']:.5f}"),
                ],
            )
        )

    add("## 5. Processing log (PRD 27.2)\n")
    stage_rows = []
    for stage in manifest.get("stages", []):
        issues = "; ".join(stage["memory_warnings"] + ([stage["error"]] if stage["error"] else []) + stage["notes"])
        stage_rows.append(
            (
                stage["name"],
                stage["classification"],
                "yes" if stage["enabled"] else "no",
                f"{stage['seconds']:.2f}",
                f"{stage['input_precision']} -> {stage['output_precision']}",
                f"{stage['rss_peak_mb']:.0f}" if stage["rss_peak_mb"] is not None else "-",
                _short_params(stage["parameters"]),
                issues or "-",
            )
        )
    add(
        _table(
            [
                "Stage",
                "Class (PRD 5.2)",
                "Enabled",
                "Seconds",
                "Precision",
                "Peak RSS MB",
                "Parameters",
                "Warnings, errors, notes",
            ],
            stage_rows,
        )
    )

    memory = manifest.get("memory", {})
    if memory.get("estimates"):
        add("## 6. Memory (PRD 21.3, 28.1)\n")
        add("Base-image estimates at the output size (decimal MB, excluding temporaries):\n")
        add(
            _table(
                ["Representation", "Bytes/pixel", "Estimated MB"],
                [(r["representation"], r["bytes_per_pixel"], r["estimated_mb"]) for r in memory["estimates"]],
            )
        )
        if memory.get("benchmark"):
            add("Measured allocation benchmark:\n")
            add(
                _table(
                    ["Representation", "Estimated MB", "Measured RSS MB", "Allocate s"],
                    [
                        (r["representation"], r["estimated_mb"], r["measured_rss_mb"], r["allocate_seconds"])
                        for r in memory["benchmark"]
                    ],
                )
            )
        add(
            f"Pipeline peak resident memory: {memory.get('pipeline_peak_rss_mb', '-')} MB; process lifetime peak: "
            f"{memory.get('process_lifetime_peak_rss_mb', '-')} MB.\n"
        )

    add("## 7. Visual results (PRD 27.3)\n")
    figures = report_dir / "figures"
    if (figures / "source_preview.jpg").exists():
        add("Original source:\n\n![Source](figures/source_preview.jpg)\n")
    final = manifest.get("final_output")
    if final:
        add(_final_section(final, figures))
    if manifest.get("subject"):
        add(f"Main subject: {subject_text(manifest['subject'])}\n")
    for method in methods:
        if (figures / f"{method}_preview.jpg").exists():
            kind = "pretrained super-resolution" if method.startswith("sr_") else "baseline"
            add(
                f"{method.capitalize()} {kind} at output size (review downscale):\n\n"
                f"![{method}](figures/{method}_preview.jpg)\n"
            )
    labels = {c["name"]: c.get("label") for c in manifest.get("crops", [])}
    for panel in manifest.get("crop_panels", []):
        name = Path(panel["path"]).name
        shows = f", {labels[panel['crop']]}" if labels.get(panel["crop"]) else ""
        add(
            f"Crop `{panel['crop']}`{shows} (source, every method at 100% output pixels, amplified difference):\n\n"
            f"![{panel['crop']}](figures/{name})\n"
        )
    for mode in sr_record.get("modes", {}):
        if (figures / f"sr_difference_{mode}.png").exists():
            add(
                f"Difference heatmap, sr_{mode} vs Lanczos at output resolution (where detail was synthesised or "
                f"drifted):\n\n![sr_{mode} vs Lanczos](figures/sr_difference_{mode}.png)\n"
            )
    for method in methods:
        if (figures / f"delta_e_{method}.png").exists():
            add(
                f"Difference map, {method} reduced to source resolution vs source:\n\n"
                f"![delta E {method}](figures/delta_e_{method}.png)\n"
            )

    if methods:
        add("## 8. Quantitative results (PRD 27.4)\n")
        add("Source consistency, measured on the 16-bit TIFF master after area reduction to source size:\n")
        add(_table(["Metric"] + list(methods), _metric_rows(methods)))
        if any(s.get("source_projection") for s in sr_record.get("modes", {}).values()):
            add(
                "SR modes with a low-frequency correction were pulled towards the source under the same area "
                "reduction these metrics use, so their PSNR, SSIM and delta E are partly true by construction. Edge "
                "displacement and the interpolation-difference statistics below remain independent checks.\n"
            )
        add(f"LPIPS: {next(iter(methods.values())).get('source_consistency', {}).get('lpips_note', '-')}.\n")
        add("Image statistics and clipping:\n")
        add(_table(["Quantity"] + list(methods), _statistics_rows(methods)))
        if manifest.get("crop_panels"):
            add("Crop high-frequency energy (variance of the luma Laplacian, encoded domain):\n")
            add(
                _table(
                    ["Crop"] + list(methods),
                    [
                        (
                            p["crop"],
                            *[_num(p["methods"].get(m, {}).get("high_frequency_energy"), 6) for m in methods],
                        )
                        for p in manifest["crop_panels"]
                    ],
                )
            )
        add("Export validation (PRD 23.6):\n")
        add(_table(["File", "Kind", "Size", "Compression", "Checks", "SHA-256"], _export_rows(manifest)))
    if sr_record:
        add(_sr_section(sr_record))

    add("## 9. Information provenance (PRD 27.5)\n")
    source_pixels = (source.get("width") or 0) * (source.get("height") or 0)
    share = f"{source_pixels / target['pixels']:.1%}" if target and source_pixels else "-"
    add(
        _table(
            ["Contribution", "In this run"],
            [
                (
                    "Original source data",
                    f"All image structure. The source holds {source_pixels:,} samples per channel, "
                    f"{share} of the output pixel count.",
                ),
                (
                    "Interpolated values",
                    "Every baseline sample (classical, deterministic kernels)"
                    + (
                        "; in the SR masters, the final resample from the x4 grid and the low-frequency correction."
                        if sr_record.get("modes")
                        else "."
                    ),
                ),
                ("Learned inference", _inference_text(sr_record)),
                ("Simulated sensor effects", "None."),
                ("Added noise or grain", "None."),
                (
                    "File encoding",
                    "16-bit sRGB integer TIFF (clips interpolation overshoot); half or float linear EXR "
                    "(keeps overshoot). Precision only, no new information.",
                ),
            ],
        )
    )

    add("## 10. Conclusions (PRD 27.6)\n")
    add("\n".join(f"- {line}" for line in conclusions(manifest)) + "\n")

    add("## 11. Provenance record (PRD 6.2)\n")
    add(
        _table(
            ["Item", "Value"],
            [
                ("Source SHA-256", f"`{source.get('sha256', '-')}`"),
                ("Decoded pixel SHA-256", f"`{source.get('pixel_sha256', '-')}`"),
                ("Package", _version(manifest)),
                ("Notebook version", manifest.get("notebook_version") or "-"),
                ("Models and weight hashes", _models_text(manifest, hashes=True)),
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


def _models_text(manifest: dict, *, hashes: bool = False) -> str:
    models = manifest.get("models") or []
    if not models:
        return "none (classical interpolation only)"
    parts = []
    for model in models:
        weights = model["weights"]
        text = f"{weights['name']} ({weights['architecture']}, {weights['license']})"
        if hashes:
            text += f", SHA-256 `{weights['sha256']}`, {weights['url']}"
        parts.append(text + f" on {model['device']['name']} ({model['precision']})")
    return "; ".join(parts)


def _inference_text(sr_record: dict) -> str:
    modes = sr_record.get("modes", {})
    if not modes:
        return "None."
    used = ", ".join(f"sr_{m}: {s['weights']}" for m, s in modes.items())
    return (
        f"Detail above the source sampling limit in the SR masters ({used}) is synthesised by pretrained networks "
        "from learned priors; it is not recovered information. The baselines contain none."
    )


def _sr_section(sr_record: dict) -> str:
    """PRD 10 and experiment E02: how each SR mode was produced and its seam and hallucination checks."""
    out = ["## 8b. Pretrained super-resolution (PRD 10, experiment E02)\n"]
    if sr_record.get("skipped"):
        out.append(f"Skipped: {sr_record['skipped']}.\n")
        return "\n".join(out)
    out.append(f"Backend `{sr_record['backend']}`; scale handling: {sr_record['scale_handling']}.\n")
    rows = []
    for mode, s in sr_record.get("modes", {}).items():
        tiling = s.get("tiling", {})
        disagreement = tiling.get("overlap_disagreement", {})
        seams = s.get("seams", {})
        projection = s.get("source_projection") or {}
        difference = s.get("interpolation_difference") or {}
        model = s.get("model", {})
        rows.append(
            (
                f"sr_{mode}",
                s["preset"]["label"],
                s["weights"],
                f"{model.get('device', {}).get('name', '-')} {model.get('precision', '')}".strip(),
                f"{tiling.get('tile_size', '-')} / {tiling.get('tile_overlap', '-')}",
                f"{tiling.get('tiles', '-')} ({tiling.get('model_calls', '-')} calls)",
                f"{tiling.get('inference_seconds', 0):.0f}",
                _num(model.get("peak_device_memory_mb"), 0),
                "; ".join(f["message"] for f in tiling.get("fallbacks", [])) or "none",
                f"{_num(disagreement.get('blend_error_p99_abs_max'), 3)} "
                f"({_num(disagreement.get('low_frequency_p99_abs_max'), 2)} / "
                f"{_num(disagreement.get('p99_abs_max'), 2)})",
                _num(seams.get("max_score"), 3),
                {True: "pass", False: "FAIL"}.get(seams.get("passed"), "-"),
                f"{projection.get('mean_abs_residual_linear_before', '-')} -> "
                f"{projection.get('mean_abs_residual_linear_after', '-')}"
                if projection
                else "off",
                f"{difference.get('mean', '-')} / {difference.get('p99', '-')}",
                range_text(s),
            )
        )
    out.append(
        _table(
            [
                "Output",
                "Mode",
                "Weights",
                "Device",
                "Tile / overlap px",
                "Tiles",
                "Inference s",
                "Peak device MB",
                "Memory fallbacks",
                "Blend error p99 (tile disagreement low-freq. / raw), 8-bit",
                "Hard-seam height max, 8-bit",
                "Seam check",
                "Low-frequency residual (linear) before -> after",
                "vs Lanczos mean / p99 (8-bit luma)",
                "Linear min: resampled -> anti-ringing -> corrected",
            ],
            rows,
        )
    )
    out.append(
        "Blend error: inside each crossfade, the low-frequency (Gaussian of 2 source pixels) difference between the "
        "two tiles times the smaller blending weight, which bounds how far blending moves the image from the "
        "better-placed tile (the soft-seam measure). Tile disagreement itself, low-frequency and raw, also counts "
        "texture the networks synthesised differently where one tile has almost no weight. Hard-seam height: the "
        "luma step across each tile-boundary line beyond the step across nearby lines, on samples where the "
        "content is flat (0 means no hard seam). Both gates are provisional until calibrated in pilot runs "
        "(PRD 25.3)" + _limits_text(sr_record) + ". Linear min: the lowest linear value after the Lanczos-3 resample "
        "from the network's x4 grid, after the anti-ringing clamp to the range of the network pixels under the "
        "kernel's main lobe, and after the gain-form low-frequency correction (negative values clip to black in "
        "the TIFF and show as dark halos).\n"
    )
    out.append("Mode intents:\n")
    out += [
        f"- `sr_{mode}`: {s['preset']['intent']} {s['preset']['warning']}".rstrip()
        for mode, s in sr_record.get("modes", {}).items()
    ]
    out.append("")
    return "\n".join(out)


def _limits_text(sr_record: dict) -> str:
    limits = next((s["seams"]["limits"] for s in sr_record.get("modes", {}).values() if "seams" in s), None)
    if not limits:
        return ""
    return (
        f"; a mode passes when the blend error p99 is at most {limits['blend_error_p99_abs_8bit']} and the "
        f"hard-seam height at most {limits['max_score']} code values"
    )


def conclusions(manifest: dict) -> list[str]:
    """Factual, automatically checkable statements; visual judgements are left to human review."""
    lines = []
    target = manifest.get("target_resolution")
    done = manifest.get("baselines", {})
    config = manifest.get("config", {})
    if manifest.get("status") != "complete":
        lines.append(f"The run did not complete ({manifest.get('status')}); conclusions cover finished stages only.")
    if target:
        within = abs(target["pixel_count_error"]) <= config.get("pixel_count_tolerance", 0.001) or (
            target["mode"] == "exact" and target["pixels"] == target["target_pixels"]
        )
        lines.append(
            f"Output size {target['width']}x{target['height']} ({target['pixels']:,} pixels) "
            f"{'meets' if within else 'misses'} the configured pixel-count target."
        )
        is_full = target["target_pixels"] >= 50_000_000
        exports_ok = (
            bool(done)
            and all(e.get("valid") for s in done.values() for e in s.get("exports", {}).values())
            and all(s.get("exports") for s in done.values())
        )
        if is_full and within and exports_ok:
            lines.append(
                "Phase 0 exit criterion met: a valid 50 MP master was created and exported from the "
                "source by classical interpolation, and every export passed read-back validation."
            )
        elif not is_full:
            lines.append(
                "Phase 0 exit criterion not assessed: this run used a reduced target "
                f"({target['target_pixels'] / 1e6:g} MP, preset {config.get('runtime_preset')})."
            )
        else:
            lines.append("Phase 0 exit criterion not met: at least one export is missing or failed validation.")
    if done:
        measured = all("source_consistency" in s for s in done.values())
        reproducible = [s["reproducibility"]["identical"] for s in done.values() if "reproducibility" in s]
        if reproducible:
            lines.append(
                f"Reproducibility: {sum(reproducible)}/{len(reproducible)} baselines produced bit-identical "
                "float32 output on a repeat run."
            )
        else:
            lines.append(
                "Reproducibility was not re-run in this preset; compare the recorded working_pixel_sha256 "
                "values across runs (the research preset repeats each baseline)."
            )
        if measured and (not reproducible or all(reproducible)):
            lines.append(
                "Phase 1 baselines are measurable: source-consistency metrics are reported for every baseline."
            )
    lines += _ranking_conclusions(manifest)
    lines += _sr_conclusions(manifest)
    lines += _final_conclusions(manifest.get("final_output"))
    lines.append(
        "Visual acceptance (PRD 25.2: seams, ringing, halos, banding) needs human review of the crop "
        "panels at 100% and 200%."
    )
    if manifest.get("super_resolution", {}).get("modes"):
        lines.append(
            "Detail in the SR masters above the source sampling limit was inferred by pretrained networks; "
            "source consistency shows it agrees with the source after reduction, not that it is correct."
        )
    else:
        lines.append(
            "No new detail was inferred: interpolation cannot recover information above the source sampling "
            "limit, which is the reference that later pretrained SR phases must beat."
        )
    return lines


# (label, metric getter, higher is better, digits) for the ranking in the conclusions
RANKED_METRICS = (
    ("Source PSNR (dB)", lambda m: m["psnr_db"], True, 2),
    ("SSIM", lambda m: m["ssim"], True, 5),
    ("MS-SSIM", lambda m: m["ms_ssim"]["value"], True, 5),
    ("Edge displacement mean (px)", lambda m: m["edge_displacement"]["mean_px"], False, 3),
    ("delta E 2000 mean", lambda m: m["color_difference_ciede2000"]["mean"], False, 4),
    ("L* Wasserstein-1", lambda m: m["luminance_distribution"]["wasserstein_1"], False, 4),
)


def rank_outputs(manifest: dict) -> list[dict]:
    """Every measured output (baselines and SR modes) ordered best first on each reported metric."""
    measured = {
        name: summary["source_consistency"]
        for name, summary in {
            **manifest.get("baselines", {}),
            **{f"sr_{m}": s for m, s in (manifest.get("super_resolution") or {}).get("modes", {}).items()},
        }.items()
        if "source_consistency" in summary
    }
    rankings = []
    for label, getter, higher, digits in RANKED_METRICS:
        values = {}
        for name, metrics in measured.items():
            try:
                value = getter(metrics)
            except (KeyError, TypeError):
                continue
            if value is not None:
                values[name] = float(value)
        order = sorted(values, key=lambda n: -values[n] if higher else values[n])
        rankings.append(
            {"metric": label, "higher_is_better": higher, "digits": digits, "order": order, "values": values}
        )
    return rankings


def _ranking_conclusions(manifest: dict) -> list[str]:
    rankings = [r for r in rank_outputs(manifest) if len(r["order"]) >= 2]
    if not rankings:
        return []
    corrected = {
        f"sr_{m}"
        for m, s in (manifest.get("super_resolution") or {}).get("modes", {}).items()
        if s.get("source_projection")
    }
    lines = []
    for r in rankings:
        ordered = ", ".join(
            f"{name}{'*' if name in corrected else ''} {r['values'][name]:.{r['digits']}f}" for name in r["order"]
        )
        lines.append(
            f"{r['metric']} ({'higher' if r['higher_is_better'] else 'lower'} is better), best first: {ordered}."
        )
    if corrected:
        lines.append(
            "* Pulled towards the source by the low-frequency correction under the same area reduction these "
            "metrics use, so its PSNR, SSIM, MS-SSIM, delta E and L* agreement are partly true by construction; "
            "edge displacement is the independent check. Source consistency also cannot show whether synthesised "
            "detail is correct."
        )
    stats = {
        name: s["statistics"]
        for name, s in {
            **manifest.get("baselines", {}),
            **{f"sr_{m}": s for m, s in (manifest.get("super_resolution") or {}).get("modes", {}).items()},
        }.items()
        if "statistics" in s
    }
    if len(stats) >= 2:
        overshoot = {n: st["below_zero_fraction"] + st["above_one_fraction"] for n, st in stats.items()}
        most = max(overshoot, key=overshoot.get)
        lowest = min(stats, key=lambda n: min(stats[n]["min"].values()))
        lines.append(
            f"Samples outside [0, 1] (kernel overshoot, a ringing indicator) are most frequent in {most} "
            f"({overshoot[most]:.4%}); the lowest linear value is {min(stats[lowest]['min'].values()):.4f} "
            f"({lowest}). The TIFF clips them, the EXR keeps them."
        )
    return lines


def _final_conclusions(final: dict | None) -> list[str]:
    if not final:
        return []
    if not final.get("method"):
        return [f"No final image was saved: {final.get('reason')}."]
    files = final.get("files", {})
    names = ", ".join(f"`{Path(f['path']).name}`" for f in files.values()) or "no files"
    line = f"Final 50 MP image: {final['method']} ({final['reason']}); saved as {names}."
    return [line, *final.get("notes", [])]


def _final_summary(final: dict | None) -> str:
    if not final:
        return "-"
    if not final.get("method"):
        return f"none ({final.get('reason')})"
    tiff = final.get("files", {}).get("tiff16")
    return f"{final['method']}" + (f", `{Path(tiff['path']).name}`" if tiff else "")


def _final_section(final: dict, figures: Path) -> str:
    out = ["### Final 50 MP image\n"]
    if not final.get("method"):
        out.append(f"No final image: {final.get('reason')}.\n")
        return "\n".join(out)
    out.append(
        f"**{final['method']}** ("
        + ("requested in the configuration" if final["rule"] == "override" else "automatic rule")
        + f"): {final['reason']}.\n"
    )
    if (figures / "final_preview.jpg").exists():
        out.append("![Final image](figures/final_preview.jpg)\n")
    rows = [
        (
            f"`{Path(f['path']).name}`",
            f"{f['width']} x {f['height']}",
            _bytes(f["size_bytes"]),
            f"{f['how']} of `{f['from']}`, SHA-256 verified",
        )
        for f in final.get("files", {}).values()
    ]
    if rows:
        out.append(_table(["File", "Dimensions", "Size", "Saved as"], rows))
    return "\n".join(out)


def subject_text(subject: dict) -> str:
    face = subject.get("primary_face")
    if face:
        text = (
            f"{len(subject['faces'])} face(s) found; the primary face is {face['width']} x {face['height']} source px "
            f"at ({face['x']}, {face['y']})"
        )
        if subject.get("high_information_face"):
            text += f", at least the provisional {subject['tier1_min_face_px']} px of a high-information face"
    elif subject.get("center_of_interest"):
        x, y = subject["center_of_interest"]
        text = f"no face found; centre of interest at ({x}, {y}): {subject['interest_basis']}"
    else:
        text = "no face and no distinct centre of interest found"
    return f"{text}. Detection: {subject['note']}. Never identity evidence."


def range_text(summary: dict) -> str:
    resample = summary.get("tiling", {}).get("resample_range") or {}
    projection = summary.get("source_projection") or {}
    parts = [_num(resample.get("min_linear_before"), 4), _num(resample.get("min_linear_after"), 4)]
    parts.append(_num(projection.get("min_linear_after"), 4) if projection else "off")
    return " -> ".join(parts)


def _sr_conclusions(manifest: dict) -> list[str]:
    record = manifest.get("super_resolution")
    if not record:
        return []
    if record.get("skipped"):
        return [f"Phase 2 super-resolution was skipped: {record['skipped']}."]
    modes = record.get("modes", {})
    target = manifest.get("target_resolution") or {}
    lines = []
    finished = {m: s for m, s in modes.items() if s.get("exports") and "seams" in s}
    complete = (
        manifest.get("status") == "complete"
        and finished
        and len(finished) == len(modes)
        and all(e.get("valid") for s in finished.values() for e in s["exports"].values())
    )
    seams_ok = all(s["seams"].get("passed") for s in finished.values())
    if not target.get("target_pixels", 0) >= 50_000_000:
        lines.append(
            "Phase 2 exit criterion not assessed: this run used a reduced target "
            f"({target.get('target_pixels', 0) / 1e6:g} MP)."
        )
    elif complete and seams_ok:
        lines.append(
            "Phase 2 exit criterion met by the automated checks: full-resolution SR ran for every mode, every "
            "export passed validation and no tile boundary exceeded the seam gates. Confirm visually."
        )
    else:
        lines.append(
            "Phase 2 exit criterion not met: "
            + ("a seam gate failed." if complete else "at least one SR mode did not finish or failed validation.")
        )
    fallbacks = sum(len(s.get("tiling", {}).get("fallbacks", [])) for s in modes.values())
    if fallbacks:
        lines.append(f"Memory fallback was used {fallbacks} time(s); see the SR table for details.")
    for mode, s in modes.items():
        resample = s.get("tiling", {}).get("resample_range") or {}
        if resample.get("min_linear_before") is not None and resample["min_linear_before"] < 0:
            final_min = min(s["statistics"]["min"].values()) if "statistics" in s else None
            lines.append(
                f"sr_{mode}: Lanczos resampling of the network output rang down to {resample['min_linear_before']:.4f} "
                f"linear ({resample['below_zero_fraction_before']:.4%} of samples below 0); the anti-ringing clamp "
                f"raised the minimum to {resample['min_linear_after']:.4f} and the master's minimum is "
                f"{_num(final_min, 4)}, so no dark halos are clipped into the TIFF."
            )
    measured = {f"sr_{m}": s["source_consistency"] for m, s in modes.items() if "source_consistency" in s}
    base = {m: s["source_consistency"] for m, s in manifest.get("baselines", {}).items() if "source_consistency" in s}
    if measured and base:
        reference = "lanczos" if "lanczos" in base else next(iter(base))
        ref_edges = base[reference]["edge_displacement"]["mean_px"]
        for name, metrics in measured.items():
            edges = metrics["edge_displacement"]["mean_px"]
            difference = modes[name[3:]].get("interpolation_difference", {})
            lines.append(
                f"{name}: source PSNR {metrics['psnr_db']:.2f} dB vs {base[reference]['psnr_db']:.2f} dB for "
                f"{reference}; edge displacement {_num(edges, 3)} px vs {_num(ref_edges, 3)} px; mean departure "
                f"from Lanczos {difference.get('mean', '-')} 8-bit luma codes (p99 {difference.get('p99', '-')})."
            )
    return lines


# --------------------------------------------------------------------------- formatting helpers


def _table(header: list[str], rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in row) + " |")
    return "\n".join(lines) + "\n"


def _dims(width, height) -> str:
    return f"{width} x {height}" if width and height else "-"


def _bytes(value) -> str:
    if value is None:
        return "-"
    return f"{value / 1e6:,.1f} MB"


def _num(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value == float("inf"):
        return "inf"
    return f"{value:.{digits}f}"


def _version(manifest: dict) -> str:
    package = manifest.get("package", {})
    git = package.get("git") or {}
    commit = git.get("commit")
    text = f"synthetic_raw_lab {package.get('version', '?')}"
    if commit:
        text += f", commit `{commit[:12]}`" + (" (uncommitted changes)" if git.get("dirty") else "")
    return text


def _short_params(parameters: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in parameters.items()) or "-"


def _validation_rows(validation: dict) -> list[tuple]:
    file, dims, samples = validation["file"], validation["dimensions"], validation["samples"]
    color, clipping, compression = validation["color"], validation["clipping"], validation["compression"]
    alpha, metadata = validation["alpha"], validation["metadata"]
    clip_text = ", ".join(
        f"{n} {c['at_min_fraction']:.3%} / {c['at_max_fraction']:.3%}" for n, c in clipping["channels"].items()
    )
    return [
        ("Format", f"{file['format']} ({file['size_bytes']:,} bytes), decoded by {file['decoder']}"),
        ("Read-only stored copy", f"`{Path(file['stored_copy']).name}` (read-only: {file['stored_read_only']})"),
        (
            "Dimensions",
            f"{dims['width']} x {dims['height']} ({dims['megapixels']:.2f} MP), aspect {dims['aspect_ratio_reduced']}",
        ),
        (
            "Bit depth",
            f"{samples['bit_depth']}-bit {samples['dtype']} (effective {samples['effective_bit_depth']}-bit)",
        ),
        ("Channels", f"{samples['channels_in_file']} in file, {samples['color_channels']} colour"),
        ("Alpha", "none" if not alpha["present"] else ("opaque" if alpha["opaque"] else f"min {alpha['min']:.4f}")),
        ("ICC profile", color["icc_profile_description"] or ("present" if color["icc_profile_present"] else "none")),
        ("Source interpretation", f"{color['resolved']['description']} ({color['resolved']['origin']})"),
        ("Working signal", color["working_signal"]),
        ("EXIF", json.dumps(metadata["exif"]) if metadata["exif_present"] else "none"),
        ("Embedded text keys", ", ".join(metadata["text_keys"]) or "none"),
        ("Clipping (at min / at max)", clip_text),
        ("JPEG quality estimate", compression["jpeg_quality_estimate"] or "-"),
        ("8x8 blockiness ratio", f"{compression['blockiness_ratio']:.3f} (about 1.0 means no block artefacts)"),
        (
            "Integrity",
            "structure verified and fully decoded" if validation["integrity"]["structure_verified"] else "decoded",
        ),
        ("Duplicates", ", ".join(d["match"] for d in validation["duplicates"]) or "none"),
    ]


def _metric_rows(done: dict) -> list[tuple]:
    def col(fn):
        return [fn(s["source_consistency"]) if "source_consistency" in s else "-" for s in done.values()]

    return [
        ("PSNR (dB)", *col(lambda m: _num(m["psnr_db"], 2))),
        ("SSIM", *col(lambda m: _num(m["ssim"], 5))),
        ("MS-SSIM (luma)", *col(lambda m: _num(m["ms_ssim"]["value"], 5))),
        ("Edge displacement mean (px)", *col(lambda m: _num(m["edge_displacement"]["mean_px"], 3))),
        ("Edge displacement p95 (px)", *col(lambda m: _num(m["edge_displacement"]["p95_px"], 3))),
        ("Edge pixel ratio", *col(lambda m: _num(m["edge_displacement"]["edge_pixel_ratio"], 3))),
        ("delta E 2000 mean", *col(lambda m: _num(m["color_difference_ciede2000"]["mean"], 4))),
        ("delta E 2000 p95", *col(lambda m: _num(m["color_difference_ciede2000"]["p95"], 4))),
        ("L* mean shift", *col(lambda m: _num(m["luminance_distribution"]["mean_shift"], 4))),
        ("L* std ratio", *col(lambda m: _num(m["luminance_distribution"]["std_ratio"], 4))),
        ("L* Wasserstein-1", *col(lambda m: _num(m["luminance_distribution"]["wasserstein_1"], 4))),
    ]


def _statistics_rows(done: dict) -> list[tuple]:
    def col(fn):
        return [fn(s) if "statistics" in s else "-" for s in done.values()]

    return [
        ("Linear mean R/G/B", *col(lambda s: " / ".join(f"{v:.4f}" for v in s["statistics"]["mean"].values()))),
        ("Linear min", *col(lambda s: f"{min(s['statistics']['min'].values()):.4f}")),
        ("Linear max", *col(lambda s: f"{max(s['statistics']['max'].values()):.4f}")),
        ("Samples below 0", *col(lambda s: f"{s['statistics']['below_zero_fraction']:.4%}")),
        ("Samples above 1", *col(lambda s: f"{s['statistics']['above_one_fraction']:.4%}")),
        ("Clipped in TIFF", *col(lambda s: _clipped_fraction(s.get("tiff_clipping")))),
        ("Float32 pixel SHA-256", *col(lambda s: f"`{s['working_pixel_sha256'][:16]}...`")),
    ]


def _clipped_fraction(clipping: dict | None) -> str:
    if not clipping:
        return "-"
    return f"{clipping['below_zero_fraction'] + clipping['above_one_fraction']:.4%}"


def _export_rows(manifest: dict) -> list[tuple]:
    rows = []
    exports = list(manifest.get("intermediates", []))
    reconstructions = list(manifest.get("baselines", {}).values())
    reconstructions += list(manifest.get("super_resolution", {}).get("modes", {}).values())
    for summary in reconstructions:
        exports += list(summary.get("exports", {}).values())
    for e in exports:
        passed = sum(e["checks"].values())
        rows.append(
            (
                f"`{Path(e['path']).name}`",
                e["kind"],
                _bytes(e["size_bytes"]),
                e["compression"],
                f"{passed}/{len(e['checks'])} passed",
                f"`{e['file_sha256'][:16]}...`",
            )
        )
    return rows
