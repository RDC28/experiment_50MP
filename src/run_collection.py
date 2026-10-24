"""Phase 6 run collection: ingest the run folders or zips of notebooks 00 to 06 and compare them (PRD 26, 27, 28).

Every notebook writes ``manifest.json`` and ``report.md`` into its run folder, and its archive zip holds those
folders. :func:`collect` takes any mix of run folders, folders of runs, zips and manifest files (for example
the repository's ``evidence/colab_runs`` and the zips the captain downloads from Colab), recognises each
manifest by its schema and normalises it into a :class:`Run`:

* the outputs it measured (baselines, SR modes, specialised and final masters) with the same source-consistency
  fields whichever notebook wrote them;
* performance: total runtime, SR time, peak memory, GPU, storage;
* the notebook-specific evidence the final report needs (identity per face tier, crowd flags, night metrics
  before and after, the universal route and its improvement verdict, Phase 5 round-trip losses, ablation and
  reference-test records).

The comparison tables group runs by source image (SHA-256), and :func:`review_sheet` cuts identical crops from
every run of one source: at 100% from the full-resolution masters when they can be found (in place, or under
``masters_roots`` such as a Drive copy), otherwise from each run's review previews, which is labelled.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import ablation, exporters, reference_test
from .evaluation_config import AggregationParams

NOTEBOOKS = {
    "synthetic_raw_lab.phase01.manifest/1": "00",
    "synthetic_raw_lab.human_crowd.manifest/1": "01",
    "synthetic_raw_lab.phase3_night.manifest/1": "02",
    "synthetic_raw_lab.phase4_universal.manifest/1": "03",
    "synthetic_raw_lab.phase5.manifest/1": "05",
    ablation.SCHEMA: "06 ablation",
    reference_test.SCHEMA: "06 reference",
}
NOTEBOOK_NAMES = {
    "00": "00 Feasibility and baselines",
    "01": "01 Human and crowd",
    "02": "02 Night scenery",
    "03": "03 Universal",
    "05": "05 Synthetic sensor research",
    "06 ablation": "06 Ablation",
    "06 reference": "06 Reference test",
}
SR_STAGE_PREFIXES = ("super_resolution_", "human_upscale_")
_SKIP_SR_STAGES = ("super_resolution_setup",)


@dataclass
class Output:
    """One measured image of a run, with notebook-independent source-consistency fields."""

    name: str
    label: str
    kind: str  # baseline | sr | specialised | final
    consistency: dict = field(default_factory=dict)
    seam_passed: bool | None = None
    tiff: str | None = None
    preview: str | None = None
    detail: dict = field(default_factory=dict)


@dataclass
class Run:
    run_id: str
    notebook: str
    schema: str
    folder: Path
    manifest: dict
    status: str
    source: dict
    label: str = ""
    parent: str | None = None
    outputs: list[Output] = field(default_factory=list)
    final: dict = field(default_factory=dict)
    performance: dict = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return self.source.get("sha256", "")

    @property
    def short(self) -> str:
        return self.run_id.split("_")[0][:15]

    @property
    def has_report(self) -> bool:
        return (self.folder / "report.md").exists()


# --------------------------------------------------------------------------- collection


def safe_extract(archive: Path, target: Path) -> Path:
    """Extract a zip into ``target``, refusing members that would land outside it."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (target / member.filename).resolve()
            if root != destination and root not in destination.parents:
                raise ValueError(f"{archive.name}: member {member.filename!r} would extract outside the folder")
        bundle.extractall(target)
    return target


def manifest_files(paths, work_dir: Path) -> list[Path]:
    """Every ``manifest.json`` under the given folders, zips and files (zips extract into ``work_dir``)."""
    found: list[Path] = []
    for entry in paths:
        path = Path(entry).expanduser()
        if path.is_file() and path.suffix.lower() == ".zip":
            found += sorted(safe_extract(path, Path(work_dir) / path.stem).rglob("manifest.json"))
        elif path.is_file() and path.name.endswith(".json"):
            found.append(path)
        elif path.is_dir():
            found += sorted(path.rglob("manifest.json"))
            for archive in sorted(path.rglob("*.zip")):
                found += sorted(safe_extract(archive, Path(work_dir) / archive.stem).rglob("manifest.json"))
        else:
            raise FileNotFoundError(f"no run folder, zip or manifest at {path}")
    return found


def collect(paths, work_dir: str | Path, params: AggregationParams | None = None) -> list[Run]:
    """Load and normalise every recognised run; a run seen twice (folder and zip) is kept once."""
    params = params or AggregationParams()
    runs: dict[str, Run] = {}
    for path in manifest_files(paths, Path(work_dir)):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict) or manifest.get("schema") not in NOTEBOOKS or "run_id" not in manifest:
            continue
        run = normalise(manifest, path.parent, params)
        if run.run_id not in runs or (run.has_report and not runs[run.run_id].has_report):
            runs[run.run_id] = run
    _link_delegated(runs)
    return sorted(runs.values(), key=lambda r: (r.label, r.notebook, r.run_id))


def _link_delegated(runs: dict[str, Run]) -> None:
    """A notebook 03 run hands human and night scenes to the notebook 01 or 02 pipeline; mark those runs."""
    for run in runs.values():
        if run.notebook != "03":
            continue
        for attempt in run.manifest.get("route", {}).get("attempts", []) or []:
            child = runs.get(attempt.get("run_id"))
            if child is not None:
                child.parent = run.run_id


def normalise(manifest: dict, folder: Path, params: AggregationParams) -> Run:
    notebook = NOTEBOOKS[manifest["schema"]]
    source = dict(manifest.get("source") or {})
    if not source and manifest.get("provenance"):  # notebook 01 keeps the source in its provenance record
        provenance = manifest["provenance"]
        dims = provenance.get("source_dimensions") or [None, None]
        source = {"sha256": provenance.get("source_sha256", ""), "original_path": provenance.get("source_name", "")}
        source.update(width=dims[0], height=dims[1])
    run = Run(
        run_id=manifest["run_id"],
        notebook=notebook,
        schema=manifest["schema"],
        folder=folder,
        manifest=manifest,
        status=str(manifest.get("status", "unknown")),
        source=source,
    )
    name = Path(str(source.get("original_path") or "")).name or "source"
    run.label = params.label(run.sha256) or (f"{name} ({run.sha256[:8]})" if run.sha256 else name)
    {"00": _phase01, "01": _human, "02": _night, "03": _universal}.get(notebook, lambda r: None)(run)
    final = manifest.get("final_output") or {}
    if final:
        run.final = {"method": final.get("method"), "reason": final.get("reason")}
        tiff = (final.get("files") or {}).get("tiff16") or {}
        if tiff.get("path"):
            for output in run.outputs:
                if output.kind == "final":
                    output.tiff = output.tiff or tiff["path"]
    run.performance = performance(manifest)
    return run


def consistency(record: dict | None) -> dict:
    """Source-consistency fields from either a ``quality_metrics.source_consistency`` record or a notebook 03
    ``measure`` record."""
    if not record:
        return {}
    if "color_difference_ciede2000" in record:
        edge = record.get("edge_displacement") or {}
        ms = record.get("ms_ssim")
        return {
            "psnr_db": record.get("psnr_db"),
            "ssim": record.get("ssim"),
            "ms_ssim": ms.get("value") if isinstance(ms, dict) else ms,
            "delta_e_mean": record["color_difference_ciede2000"].get("mean"),
            "delta_e_p95": record["color_difference_ciede2000"].get("p95"),
            "edge_mean_px": edge.get("mean_px"),
            "edge_p95_px": edge.get("p95_px"),
        }
    inner = record.get("source_consistency", record)
    edge = inner.get("edge_displacement") or {}
    return {
        "psnr_db": inner.get("psnr_db"),
        "ssim": inner.get("ssim"),
        "ms_ssim": None,
        "delta_e_mean": inner.get("delta_e_mean"),
        "delta_e_p95": inner.get("delta_e_p95"),
        "edge_mean_px": edge.get("mean_px"),
        "edge_p95_px": edge.get("p95_px"),
    }


def _preview(run: Run, name: str) -> str | None:
    path = run.folder / "figures" / f"{name}_preview.jpg"
    return str(path) if path.exists() else None


def _tiff(exports: dict | None) -> str | None:
    return ((exports or {}).get("tiff16") or {}).get("path")


def _sr_outputs(run: Run, modes: dict) -> None:
    for mode, summary in (modes or {}).items():
        if "source_consistency" not in summary and "seams" not in summary:
            continue
        run.outputs.append(
            Output(
                f"sr_{mode}",
                f"{mode} SR",
                "sr",
                consistency(summary.get("source_consistency")),
                (summary.get("seams") or {}).get("passed"),
                _tiff(summary.get("exports")),
                _preview(run, f"sr_{mode}"),
                {
                    "inference_seconds": (summary.get("tiling") or {}).get("inference_seconds"),
                    "difference_vs_lanczos": summary.get("interpolation_difference"),
                    "statistics": summary.get("statistics"),
                    "tiff_clipping": summary.get("tiff_clipping"),
                },
            )
        )


def _phase01(run: Run) -> None:
    m = run.manifest
    for name, record in (m.get("baselines") or {}).items():
        run.outputs.append(
            Output(
                name,
                name.capitalize(),
                "baseline",
                consistency(record.get("source_consistency")),
                None,
                _tiff(record.get("exports")),
                _preview(run, name),
                {"statistics": record.get("statistics"), "tiff_clipping": record.get("tiff_clipping")},
            )
        )
    _sr_outputs(run, (m.get("super_resolution") or {}).get("modes"))
    final = (m.get("final_output") or {}).get("method")
    energies: dict[str, list[float]] = {}
    for panel in m.get("crop_panels") or []:
        for name, method in (panel.get("methods") or {}).items():
            if method.get("high_frequency_energy") is not None:
                energies.setdefault(name, []).append(method["high_frequency_energy"])
    for output in run.outputs:
        output.detail["crop_hf_energy"] = energies.get(output.name, [])
        if output.name == final:
            output.detail["final"] = True


def _human(run: Run) -> None:
    m = run.manifest
    run.outputs.append(
        Output(
            "human_final",
            "notebook 01 final",
            "final",
            consistency(m.get("source_consistency")),
            None,
            _tiff(m.get("exports")),
            _preview(run, "final"),
            {
                "identity": m.get("identity"),
                "failure_events": m.get("failure_events"),
                "crowd": m.get("crowd"),
                "sr_backend": (m.get("super_resolution") or {}).get("backend"),
            },
        )
    )


def _night(run: Run) -> None:
    m = run.manifest
    sr_modes = (m.get("super_resolution") or {}).get("modes") or {}
    for name, record in (m.get("comparisons") or {}).items():
        mode = sr_modes.get(name.removeprefix("sr_")) if name.startswith("sr_") else None
        run.outputs.append(
            Output(
                name,
                record.get("label", name),
                "sr" if name.startswith("sr_") else "baseline",
                consistency(record.get("source_consistency")),
                (mode or {}).get("seams", {}).get("passed") if mode else None,
                None,
                _preview(run, name),
                {"night_metrics": record.get("night_metrics")},
            )
        )
    final = m.get("final") or {}
    night = m.get("night") or {}
    run.outputs.append(
        Output(
            "night_final",
            "notebook 02 final (Layer 5B)",
            "final",
            consistency(final.get("source_consistency")),
            None,
            _tiff(final.get("exports")),
            _preview(run, "final"),
            {
                "input": (night.get("input") or {}).get("slot"),
                "metrics_before": night.get("metrics_before"),
                "metrics_after": night.get("metrics_after"),
                "comparison": night.get("comparison"),
                "fallback_fraction": night.get("fallback_fraction"),
                "source_night_metrics": m.get("source_night_metrics"),
            },
        )
    )


def _universal(run: Run) -> None:
    m = run.manifest
    route = m.get("route") or {}
    evaluation = m.get("evaluation") or {}
    for name, candidate in ((route.get("selection") or {}).get("candidates") or {}).items():
        run.outputs.append(
            Output(
                name,
                candidate.get("label", name),
                "sr" if name.startswith("sr_") else "baseline",
                consistency(candidate.get("measurements")),
                None,
                None,
                _preview(run, name),
                {"measurements": candidate.get("measurements"), "verdict": candidate.get("verdict")},
            )
        )
    run.outputs.append(
        Output(
            "universal_final",
            f"notebook 03 final ({route.get('category', '?')})",
            "final",
            consistency(evaluation.get("final")),
            None,
            _tiff(route.get("exports")),
            _preview(run, "final"),
            {
                "category": route.get("category"),
                "route": route.get("pipeline"),
                "method": route.get("method"),
                "evaluation": evaluation,
                "scene": (m.get("scene") or {}).get("classification"),
            },
        )
    )


def performance(manifest: dict) -> dict:
    stages = manifest.get("stages") or []
    sr = sum(
        s.get("seconds") or 0
        for s in stages
        if s.get("name", "").startswith(SR_STAGE_PREFIXES) and s.get("name") not in _SKIP_SR_STAGES
    )
    runtime = manifest.get("runtime") or {}
    gpu = runtime.get("gpu") or {}
    models = manifest.get("models") or []
    peaks = [m.get("peak_device_memory_mb") for m in models if isinstance(m, dict) and m.get("peak_device_memory_mb")]
    config = manifest.get("config") or {}
    target = manifest.get("target_resolution") or (manifest.get("route") or {}).get("target") or {}
    return {
        "total_seconds": manifest.get("total_seconds"),
        "sr_seconds": sr or None,
        "peak_rss_mb": (manifest.get("memory") or {}).get("pipeline_peak_rss_mb"),
        "peak_device_mb": max(peaks) if peaks else None,
        "gpu": gpu.get("name") or "none",
        "in_colab": runtime.get("in_colab"),
        "preset": config.get("runtime_preset"),
        "output_megapixels": target.get("megapixels"),
        "storage_mb": (manifest.get("storage_used_bytes") or 0) / 1e6 or None,
    }


# --------------------------------------------------------------------------- tables


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "pass" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def inventory_table(runs: list[Run]) -> tuple[list[str], list[tuple]]:
    header = ["source", "notebook", "run", "status", "final image", "report"]
    rows = [
        (
            r.label,
            NOTEBOOK_NAMES.get(r.notebook, r.notebook) + (f" (from {r.parent[:15]})" if r.parent else ""),
            f"`{r.run_id}`",
            r.status,
            (r.final.get("method") or "-") if r.final else "-",
            "yes" if r.has_report else "no",
        )
        for r in runs
    ]
    return header, rows


def outputs_table(runs: list[Run]) -> tuple[list[str], list[tuple]]:
    """Every measured output of every run, source consistency after area reduction to the source size."""
    header = ["source", "notebook", "run", "output", "PSNR dB", "SSIM", "dE00 mean", "edge mean px", "seam gate"]
    rows = []
    for run in runs:
        for output in run.outputs:
            c = output.consistency
            rows.append(
                (
                    run.label,
                    run.notebook,
                    run.short,
                    output.name + (" (final)" if output.kind == "final" or output.detail.get("final") else ""),
                    fmt(c.get("psnr_db")),
                    fmt(c.get("ssim"), 5),
                    fmt(c.get("delta_e_mean"), 3),
                    fmt(c.get("edge_mean_px"), 3),
                    fmt(output.seam_passed) if output.seam_passed is not None else "-",
                )
            )
    return header, rows


def performance_table(runs: list[Run]) -> tuple[list[str], list[tuple]]:
    """PRD 28.1 performance of every run."""
    header = [
        "source",
        "notebook",
        "run",
        "GPU",
        "preset",
        "output MP",
        "total s",
        "SR s",
        "peak RSS MB",
        "peak GPU MB",
        "storage MB",
    ]
    rows = []
    for run in runs:
        p = run.performance
        rows.append(
            (
                run.label,
                run.notebook,
                run.short,
                p["gpu"],
                p["preset"] or "-",
                fmt(p["output_megapixels"], 1),
                fmt(p["total_seconds"], 0),
                fmt(p["sr_seconds"], 0),
                fmt(p["peak_rss_mb"], 0),
                fmt(p["peak_device_mb"], 0),
                fmt(p["storage_mb"], 0),
            )
        )
    return header, rows


def night_table(runs: list[Run]) -> tuple[list[str], list[tuple]]:
    """PRD 23.5 night metrics before and after the Layer 5B processing of every notebook 02 run."""
    header = ["source", "run", "input", "metric", "before", "after"]
    rows = []
    for run in runs:
        final = next((o for o in run.outputs if o.name == "night_final"), None)
        if not final or not final.detail.get("metrics_before"):
            continue
        before, after = final.detail["metrics_before"], final.detail["metrics_after"] or {}
        for key, label in (
            ("banding_score", "banding score"),
            ("shadow_chroma_noise_codes", "shadow chroma noise (codes)"),
            ("black_crush_fraction", "black crush fraction"),
        ):
            rows.append(
                (
                    run.label,
                    run.short,
                    final.detail.get("input"),
                    label,
                    fmt(before.get(key), 4),
                    fmt(after.get(key), 4),
                )
            )
        clip_b = (before.get("highlight_clipping_fraction") or {}).get("any_channel")
        clip_a = (after.get("highlight_clipping_fraction") or {}).get("any_channel")
        rows.append(
            (
                run.label,
                run.short,
                final.detail.get("input"),
                "highlight clipping (any channel)",
                fmt(clip_b, 5),
                fmt(clip_a, 5),
            )
        )
        halo_b, halo_a = (before.get("halo") or {}).get("fraction"), (after.get("halo") or {}).get("fraction")
        rows.append(
            (run.label, run.short, final.detail.get("input"), "halo ring fraction", fmt(halo_b, 4), fmt(halo_a, 4))
        )
    return header, rows


def identity_table(runs: list[Run]) -> tuple[list[str], list[tuple]]:
    """Notebook 01 identity consistency per face tier and the recovery events (PRD 11.3, 11.6)."""
    header = ["source", "run", "tier", "faces", "mean SSIM", "mean dE00", "recovery events"]
    rows = []
    for run in runs:
        final = next((o for o in run.outputs if o.name == "human_final"), None)
        if not final or not final.detail.get("identity"):
            continue
        events = final.detail.get("failure_events") or []
        for tier in final.detail["identity"].get("per_tier", []):
            tier_events = sum(1 for e in events if e.get("tier") == tier.get("tier"))
            rows.append(
                (
                    run.label,
                    run.short,
                    tier.get("tier"),
                    tier.get("face_count"),
                    fmt(tier.get("mean_ssim"), 4),
                    fmt(tier.get("mean_ciede2000"), 3),
                    tier_events,
                )
            )
    return header, rows


def markdown_table(header: list[str], rows: list[tuple]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |" for row in rows]
    return "\n".join(lines)


# --------------------------------------------------------------------------- visual review sheet


def locate(path: str | None, roots=()) -> Path | None:
    """A master recorded in a manifest: in place, else a file of the same name under one of ``roots``."""
    if not path:
        return None
    candidate = Path(path)
    if candidate.exists():
        return candidate
    for root in roots:
        matches = sorted(Path(root).expanduser().rglob(candidate.name))
        if matches:
            return matches[0]
    return None


def crop_boxes(runs: list[Run], params: AggregationParams) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Normalised review crops for one source: the configured boxes, else the crops and faces the runs recorded."""
    if params.review_crops:
        return [(f"crop_{i}", tuple(box)) for i, box in enumerate(params.review_crops, start=1)]
    found: list[tuple[str, tuple]] = []
    for run in runs:
        width, height = run.source.get("width"), run.source.get("height")
        if not width or not height:
            continue
        m = run.manifest
        recorded = list(m.get("crops") or []) + list((m.get("route") or {}).get("crops") or [])
        recorded += [
            {
                "name": f"face_{f.get('face_index', i)}",
                "x": f["box"][0],
                "y": f["box"][1],
                "width": f["box"][2],
                "height": f["box"][3],
            }
            for i, f in enumerate((m.get("identity") or {}).get("faces", []))
            if f.get("box")
        ]
        for crop in recorded:
            box = (crop["x"] / width, crop["y"] / height, crop["width"] / width, crop["height"] / height)
            if all(_iou(box, other) < 0.3 for _, other in found):
                found.append((crop.get("name", f"crop_{len(found) + 1}"), box))
    return found[: params.review_crop_count]


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _read_rgb8(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        from . import color_management as cm

        data = tifffile.imread(path)
        return cm.srgb_uint16_to_uint8(data) if data.dtype == np.uint16 else data.astype(np.uint8)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _cut(image: np.ndarray, box) -> np.ndarray:
    height, width = image.shape[:2]
    x0, y0 = int(round(box[0] * width)), int(round(box[1] * height))
    x1, y1 = max(x0 + 1, int(round((box[0] + box[2]) * width))), max(y0 + 1, int(round((box[1] + box[3]) * height)))
    return np.ascontiguousarray(image[y0:y1, x0:x1])


def review_sheet(
    runs: list[Run],
    out_path: str | Path,
    params: AggregationParams | None = None,
    masters_roots=(),
    source_image: str | Path | None = None,
) -> dict:
    """The identical-crops sheet for the runs of one source: one row per run output, one column per crop.

    Each row is cut from the output's full-resolution TIFF when it can be found (100% output pixels), else
    from its review preview (marked ``preview``). The first row is the source, from ``source_image`` or the
    run's stored source copy when available, else its preview. Returns the figure path and the row sources.
    """
    params = params or AggregationParams()
    boxes = crop_boxes(runs, params)
    if not boxes:
        raise ValueError("no crops: give aggregation.review_crops or include runs that recorded crops")
    rows: list[tuple[str, list[np.ndarray]]] = []
    provenance: list[dict] = []
    source_path = Path(source_image) if source_image else None
    if source_path is None:
        for run in runs:
            stored = locate(run.source.get("stored_copy"), masters_roots)
            if stored is not None:
                source_path = stored
                break
    if source_path is None:
        source_path = next(
            (
                run.folder / "figures" / "source_preview.jpg"
                for run in runs
                if (run.folder / "figures" / "source_preview.jpg").exists()
            ),
            None,
        )
    if source_path is not None:
        image = _read_rgb8(Path(source_path))
        rows.append(("source", [_cut(image, box) for _, box in boxes]))
        provenance.append({"row": "source", "from": str(source_path)})
    for run in runs:
        for output in run.outputs:
            master = locate(output.tiff, masters_roots)
            path, scale = (
                (master, "100%")
                if master is not None
                else (Path(output.preview) if output.preview else None, "preview")
            )
            if path is None:
                continue
            image = _read_rgb8(path)
            label = f"{run.notebook} {output.name}" + (" (preview)" if scale == "preview" else "")
            rows.append((label, [_cut(image, box) for _, box in boxes]))
            provenance.append(
                {"row": label, "run": run.run_id, "output": output.name, "from": str(path), "scale": scale}
            )
            del image
    if not rows:
        raise ValueError("no masters or previews found for these runs")
    sheet = ablation.review_sheet(rows, [name for name, _ in boxes], params.review_crop_px)
    path = exporters.write_review_image(Path(out_path), sheet)
    return {"path": str(path), "crops": [{"name": n, "box": list(b)} for n, b in boxes], "rows": provenance}


def by_source(runs: list[Run]) -> dict[str, list[Run]]:
    groups: dict[str, list[Run]] = {}
    for run in runs:
        if run.notebook.startswith("06 reference"):
            continue
        groups.setdefault(run.sha256 or run.label, []).append(run)
    return groups
