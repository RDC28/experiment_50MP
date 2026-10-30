"""Sequential analytic source-resolution runs with one shared hypothesis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from prd_v5.colour_metrics import calibrated_delta_e
from prd_v5.comparison import compare_bayer, compare_renders, validate_region
from prd_v5.config import AnalyticRunConfig
from prd_v5.delivery import digest, write_pair_html
from prd_v5.fidelity import SRCheckpoint
from prd_v5.fields import validate_selections
from prd_v5.geometry import GeometryPlanner
from prd_v5.hypothesis import CameraField
from prd_v5.inputs import InputInspector
from prd_v5.learned import ResidualAdapter, resolve_mode
from prd_v5.mtf import tiff_edge_mtf


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
    colour_references: dict | None = None,
    mtf_edge: dict | None = None,
) -> dict:
    """Run caller-labelled variants sequentially using exactly the same config.

    Labels do not certify dimensions, registration or common scene content.
    Actual source dimensions/hashes and geometry remain in each child manifest.
    A failed child leaves diagnostic outputs but no pair completion manifest.
    Optional comparison_region adds native-pixel crop metrics alongside the whole
    visible raster comparison. Bounds must fit both planned rasters before any run.
    Existing output directories are never reused. Full-scale work is Colab-only.
    Optional colour_references maps 2k/4k to {path, calibration} records for
    independently trusted aligned uint16 sRGB TIFFs. Calibration is caller
    attestation; pair disagreement is never used as colour truth.
    Optional mtf_edge supplies the same native region, normal_axis and
    edge_evidence for each developed TIFF; it does not certify registration.
    """
    if not isinstance(config, AnalyticRunConfig):
        raise ValueError("Expected one validated AnalyticRunConfig")
    learned_state = resolve_mode(inverse_mode, residual_adapter, config.camera)
    validate_region(comparison_region)
    colour_references = {} if colour_references is None else colour_references
    if not isinstance(colour_references, dict) or set(colour_references) - {"2k", "4k"}:
        raise ValueError("colour_references must map only 2k/4k labels")
    for record in colour_references.values():
        if not isinstance(record, dict) or set(record) != {"path", "calibration"}:
            raise ValueError("Each colour reference requires exactly path and calibration")
        if not isinstance(record["path"], (str, Path)) or not str(record["path"]).strip():
            raise ValueError("Colour reference path must be nonempty")
    if mtf_edge is not None:
        if not isinstance(mtf_edge, dict) or set(mtf_edge) != {"region", "normal_axis", "edge_evidence"}:
            raise ValueError("mtf_edge requires exactly region, normal_axis and edge_evidence")
        validate_region(mtf_edge["region"])
        if mtf_edge["normal_axis"] not in ("x", "y"):
            raise ValueError("mtf_edge normal_axis must be x or y")
        evidence = mtf_edge["edge_evidence"]
        if evidence is not None and (not isinstance(evidence, str) or not evidence.strip()):
            raise ValueError("mtf_edge edge_evidence must be a nonempty string")
    camera_selections = tuple(camera_selections)
    isp_selections = tuple(isp_selections)
    validate_selections(config.camera, camera_selections)
    validate_selections(config.isp, isp_selections, isp=True)
    checkpoint_evidence = sr_checkpoint.verify() if sr_checkpoint is not None else None
    if comparison_region is not None:
        for source in (source_2k, source_4k):
            record = InputInspector().inspect(source, assume_unprofiled_srgb=config.assume_unprofiled_srgb)
            plan = GeometryPlanner().plan(
                record.width,
                record.height,
                target_megapixels=config.target_megapixels,
                mode=config.geometry_mode,
                native_fit_policy=config.native_fit_policy,
            )
            if comparison_region[2] > plan.height or comparison_region[3] > plan.width:
                raise ValueError("region exceeds a visible raster")
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
            colour_reference=colour_references.get(label),
        )
        manifest = output / label / "manifest.json"
        runs[label] = {
            "exposure_sweep": child["stages"]["exposure_sweep"],
            "wb_sweep": child["stages"]["wb_sweep"],
            "demosaic_sweep": child["stages"]["demosaic_sweep"],
            "tone_sweep": child["stages"]["tone_sweep"],
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
    raw_comparison = {}
    for stage, filename, domain in (
        ("pre_noise", "clean_bayer.npy", "Expected electrons before stochastic and body effects"),
        ("post_noise", "noisy_bayer.npy", "ADC-referred electrons including clipping and quantisation"),
    ):
        paths = [output / label / filename for label in ("2k", "4k")]
        raw_comparison[stage] = {
            "domain": domain,
            "whole_raster": compare_bayer(*paths, cfa_pattern=config.camera.cfa_pattern, strip_rows=config.strip_rows),
        }
        if comparison_region is not None:
            raw_comparison[stage]["crop"] = compare_bayer(
                *paths, cfa_pattern=config.camera.cfa_pattern, strip_rows=config.strip_rows, region=comparison_region
            )
    colour_comparison = {}
    for label in ("2k", "4k"):
        reference = colour_references.get(label, {})
        candidate = output / label / "developed.tiff"
        evidence = {
            "whole_raster": calibrated_delta_e(
                candidate,
                reference.get("path"),
                calibration=reference.get("calibration"),
                strip_rows=config.strip_rows,
            ),
        }
        if comparison_region is not None:
            evidence["crop"] = calibrated_delta_e(
                candidate,
                reference.get("path"),
                calibration=reference.get("calibration"),
                strip_rows=config.strip_rows,
                region=comparison_region,
            )
        if reference:
            path = Path(reference["path"])
            evidence["reference"] = {"path": str(path), "sha256": digest(path), "size_bytes": path.stat().st_size}
        colour_comparison[label] = evidence
    mtf_comparison = {
        label: tiff_edge_mtf(output / label / "developed.tiff", **(mtf_edge or {})) for label in ("2k", "4k")
    }
    report = {
        "mtf_comparison": mtf_comparison,
        "colour_comparison": colour_comparison,
        "raw_comparison": raw_comparison,
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
