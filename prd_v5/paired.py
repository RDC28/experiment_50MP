"""Sequential analytic source-resolution runs with one shared hypothesis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from prd_v5.comparison import compare_renders, validate_region
from prd_v5.config import AnalyticRunConfig
from prd_v5.delivery import write_pair_html
from prd_v5.fidelity import SRCheckpoint
from prd_v5.fields import validate_selections
from prd_v5.hypothesis import CameraField
from prd_v5.learned import ResidualAdapter, resolve_mode


def run_analytic_pair(
    source_2k: str | Path,
    source_4k: str | Path,
    output_dir: str | Path,
    config: AnalyticRunConfig,
    *,
    camera_selections: tuple[CameraField, ...] = (),
    isp_selections: tuple[CameraField, ...] = (),
    sr_checkpoint: SRCheckpoint | None = None,
    inverse_mode: str = "ANALYTIC",
    residual_adapter: ResidualAdapter | None = None,
    comparison_region: tuple[int, int, int, int] | None = None,
) -> dict:
    """Run caller-labelled variants sequentially using exactly the same config.

    Labels do not certify dimensions, registration or common scene content.
    Actual source dimensions/hashes and geometry remain in each child manifest.
    A failed child leaves diagnostic outputs but no pair completion manifest.
    Optional comparison_region adds native-pixel crop metrics alongside the whole
    visible raster comparison. Bounds must fit both developed TIFFs.
    Existing output directories are never reused. Full-scale work is Colab-only.
    """
    if not isinstance(config, AnalyticRunConfig):
        raise ValueError("Expected one validated AnalyticRunConfig")
    learned_state = resolve_mode(inverse_mode, residual_adapter, config.camera)
    validate_region(comparison_region)
    camera_selections = tuple(camera_selections)
    isp_selections = tuple(isp_selections)
    validate_selections(config.camera, camera_selections)
    validate_selections(config.isp, isp_selections, isp=True)
    checkpoint_evidence = sr_checkpoint.verify() if sr_checkpoint is not None else None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    runs = {}
    # Each call closes disk-backed working arrays before the next call starts.
    for label, source in (("2k", source_2k), ("4k", source_4k)):
        child = config.run(
            source,
            output / label,
            camera_selections=camera_selections,
            isp_selections=isp_selections,
            sr_checkpoint=sr_checkpoint,
            inverse_mode=inverse_mode,
            residual_adapter=residual_adapter,
        )
        manifest = output / label / "manifest.json"
        runs[label] = {
            "learned_raw": child["stages"]["learned_raw"],
            "optics": child["stages"]["optics"],
            "distortion": child["stages"]["distortion"],
            "manifest": f"{label}/manifest.json",
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "source_sha256": child["source"]["sha256"],
            "source_dimensions": [child["source"]["width"], child["source"]["height"]],
            "output_dimensions": [child["geometry"]["width"], child["geometry"]["height"]],
        }
    comparison = compare_renders(
        output / "2k/developed.tiff", output / "4k/developed.tiff", strip_rows=config.strip_rows
    )
    crop_comparison = (
        compare_renders(
            output / "2k/developed.tiff",
            output / "4k/developed.tiff",
            strip_rows=config.strip_rows,
            region=comparison_region,
        )
        if comparison_region is not None
        else None
    )
    report = {
        "schema_version": 1,
        "pipeline": "sequential experimental learned residual pair"
        if residual_adapter is not None
        else (
            "sequential checkpoint SR plus analytic source-resolution pair"
            if sr_checkpoint is not None
            else "sequential analytic source-resolution pair; no learned inference"
        ),
        "shared_config": json.loads(config.to_json()),
        "inverse_isp": {**learned_state, "status": runs["4k"]["learned_raw"]["status"]},
        "shared_camera": asdict(config.camera),
        "shared_checkpoint": checkpoint_evidence,
        "replay": {
            "additional_checkpoint_required": sr_checkpoint is not None or residual_adapter is not None,
            "limitation": (
                "Shared config stores analytic settings only; replay requires SR bytes "
                "and any caller-loaded residual adapter"
            ),
        },
        "runs": runs,
        "comparison": {
            **comparison,
            "limitations": [
                "2k and 4k are caller labels; scene identity and registration are not verified",
                "Shared hypotheses and random stream seeds do not guarantee identical noise residuals",
                "Unequal raster shapes skip metrics; equal shapes do not establish scene registration",
                "Scientific source-support versus generated-detail comparison remains pending",
            ],
        },
    }
    if crop_comparison is not None:
        report["crop_comparison"] = {
            **crop_comparison,
            "limitations": report["comparison"]["limitations"],
        }
    write_pair_html(output, report)
    (output / "pair_manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
