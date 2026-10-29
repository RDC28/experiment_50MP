"""Single analytic MVP run with disk-backed stages and a completion manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from prd_v5.analytic import AnalyticISP
from prd_v5.config import AnalyticRunConfig
from prd_v5.decode import decode_source
from prd_v5.dng import write_dng
from prd_v5.fidelity import SRCheckpoint, reconstruct
from prd_v5.fields import validate_selections
from prd_v5.geometry import GeometryPlanner
from prd_v5.hypothesis import CameraField, CameraHypothesis
from prd_v5.inputs import InputInspector
from prd_v5.learned import ResidualAdapter, apply_residual, resolve_mode
from prd_v5.raster import rasterize
from prd_v5.registry import analytic_registry
from prd_v5.report import render_report
from prd_v5.roundtrip import develop_roundtrip
from prd_v5.runtime import runtime_record
from prd_v5.sensor import BayerSensor, SensorEffects


def run_analytic(
    source_path: str | Path,
    output_dir: str | Path,
    camera: CameraHypothesis,
    isp: AnalyticISP | None = None,
    *,
    sensor_effects: SensorEffects | None = None,
    target_megapixels: float = 50.0,
    geometry_mode: str = "PRESERVE_SOURCE_ASPECT_50MP",
    native_fit_policy: str = "LETTERBOX_SENSOR_MARGIN",
    assume_unprofiled_srgb: bool = False,
    strip_rows: int = 64,
    max_decode_bytes: int = 256 * 1024 * 1024,
    max_develop_bytes: int = 512 * 1024 * 1024,
    camera_selections: tuple[CameraField, ...] = (),
    isp_selections: tuple[CameraField, ...] = (),
    sr_checkpoint: SRCheckpoint | None = None,
    inverse_mode: str = "ANALYTIC",
    residual_adapter: ResidualAdapter | None = None,
) -> dict:
    """Run explicit bilinear/analytic MVP, never an implicit learned fallback.

    An optional verified SR checkpoint reconstructs encoded sRGB before geometry
    and analytic unprocessing. Saved config.json describes only the analytic
    settings; replaying SR also requires the checkpoint in the stage evidence.

    Requires a new output directory to protect prior results. On failure partial
    deliverables remain for diagnosis without a completion manifest. Temporary
    arrays are closed and removed on success or failure. The same frozen camera
    can be supplied to sequential source-resolution runs. Full-scale execution
    belongs in Colab with explicit memory budgets, never on the development Mac.
    """
    if sensor_effects is None:
        sensor_effects = SensorEffects()
    resolve_mode(inverse_mode, residual_adapter, camera)
    if isp is None:
        isp = AnalyticISP()
    # Save executable settings, including actual hypotheses rather than a seed
    # alone. Paths remain caller-supplied so bundles can be replayed elsewhere.
    config = AnalyticRunConfig(
        camera=camera,
        sensor_effects=sensor_effects,
        isp=isp,
        target_megapixels=target_megapixels,
        geometry_mode=geometry_mode,
        native_fit_policy=native_fit_policy,
        assume_unprofiled_srgb=assume_unprofiled_srgb,
        strip_rows=strip_rows,
        max_decode_bytes=max_decode_bytes,
        max_develop_bytes=max_develop_bytes,
    )
    selections = validate_selections(camera, camera_selections)
    selections.update(validate_selections(isp, isp_selections, isp=True))
    for value in (strip_rows, max_decode_bytes, max_develop_bytes):
        if type(value) is not int or value <= 0:
            raise ValueError("Strip height and memory budgets must be positive integers")
    source = InputInspector().inspect(source_path, assume_unprofiled_srgb=assume_unprofiled_srgb)
    geometry = GeometryPlanner().plan(
        source.width,
        source.height,
        target_megapixels=target_megapixels,
        mode=geometry_mode,
        native_fit_policy=native_fit_policy,
    )
    if geometry.generated_bands:
        raise ValueError("Geometry requires outpainting, but no generator is available")
    if source.width * source.height * 8 > max_decode_bytes:
        raise ValueError("Source decoder memory budget exceeded before allocation")
    if geometry.width * geometry.height * 64 > max_develop_bytes:
        raise ValueError("LibRaw development memory budget exceeded before allocation")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    stages = {}
    arrays = []
    with TemporaryDirectory(prefix="working-", dir=output) as scratch:

        def allocate(name, dtype, shape):
            array = np.memmap(Path(scratch) / name, mode="w+", dtype=dtype, shape=shape)
            arrays.append(array)
            return array

        try:
            decoded = allocate("source", np.float32, (source.height, source.width, 3))
            stages["decode"] = decode_source(
                source_path,
                decoded,
                assume_unprofiled_srgb=assume_unprofiled_srgb,
                max_decode_bytes=max_decode_bytes,
                strip_rows=strip_rows,
            )
            raster_source = decoded
            raster_plan = geometry
            if sr_checkpoint is not None:
                sr_path = Path(scratch) / "reconstructed.npy"
                stages["super_resolution"] = reconstruct(decoded, sr_path, sr_checkpoint)
                raster_source = np.load(sr_path, mmap_mode="r", allow_pickle=False)
                arrays.append(raster_source)
                raster_plan = replace(
                    geometry,
                    source_size=(source.width * sr_checkpoint.scale, source.height * sr_checkpoint.scale),
                    scale=geometry.scale / sr_checkpoint.scale,
                )
            shape = (geometry.height, geometry.width)
            rgb = allocate("raster", np.float32, (*shape, 3))
            support = allocate("support", np.bool_, shape)
            stages["raster"] = rasterize(raster_source, rgb, support, raster_plan)
            if sr_checkpoint is not None:
                stages["raster"]["learned_super_resolution"] = sr_checkpoint.backend
            clean = allocate("clean", np.float32, (*shape, 3))
            stages["analytic"] = isp.process(rgb, clean, strip_rows=strip_rows)
            stages["learned_raw"] = apply_residual(rgb, clean, camera, residual_adapter, mode=inverse_mode)
            adc = allocate("adc", np.uint16, shape)
            stages["sensor"] = BayerSensor(camera, sensor_effects).process(clean, adc, strip_rows=strip_rows)
            stages["dng"] = write_dng(output / "capture.dng", adc, camera, geometry, isp, strip_rows=strip_rows)
            stages["roundtrip"] = develop_roundtrip(
                output / "capture.dng",
                output / "developed.tiff",
                adc,
                camera,
                geometry,
                isp,
                max_working_bytes=max_develop_bytes,
            )
            # Export only after the independent image gate, without materializing
            # the full mask in RAM. NPY preserves boolean dtype and raster shape.
            clipped_pixels = 0
            packaged_clipping = np.lib.format.open_memmap(
                output / "source_clipped.npy", mode="w+", dtype=np.bool_, shape=shape
            )
            try:
                for start in range(0, shape[0], strip_rows):
                    stop = start + strip_rows
                    clipped = np.any(rgb[start:stop] >= 1, axis=-1) & support[start:stop]
                    packaged_clipping[start:stop] = clipped
                    clipped_pixels += int(np.count_nonzero(clipped))
                packaged_clipping.flush()
            finally:
                packaged_clipping._mmap.close()
            packaged_support = np.lib.format.open_memmap(
                output / "source_support.npy", mode="w+", dtype=np.bool_, shape=shape
            )
            try:
                for start in range(0, shape[0], strip_rows):
                    packaged_support[start : start + strip_rows] = support[start : start + strip_rows]
                packaged_support.flush()
            finally:
                packaged_support._mmap.close()
        finally:
            for array in reversed(arrays):
                array._mmap.close()
    (output / "config.json").write_text(config.to_json(), encoding="utf-8")
    artifacts = {}
    for name in ("capture.dng", "developed.tiff", "source_support.npy", "source_clipped.npy", "config.json"):
        path = output / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        artifacts[name] = {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}
    report = {
        "schema_version": 1,
        "pipeline": "experimental learned residual over analytic baseline"
        if residual_adapter is not None
        else (
            "verified checkpoint SR with bilinear geometry and analytic inverse ISP"
            if sr_checkpoint is not None
            else "analytic MVP with explicit bilinear resampling; no learned inference"
        ),
        "replay": {
            "config_scope": "analytic settings only",
            "additional_checkpoint_required": sr_checkpoint is not None or residual_adapter is not None,
            "inverse_isp": stages["learned_raw"],
            "residual_replay": "Requires same caller-loaded adapter; config stores analytic baseline only",
            "checkpoint_evidence": "stages.super_resolution.checkpoint" if sr_checkpoint is not None else None,
        },
        "runtime": runtime_record(),
        "source": source.to_dict(),
        "geometry": geometry.to_dict(),
        "camera": asdict(camera),
        "registry": {**analytic_registry(camera, source, geometry, isp), **sensor_effects.registry(), **selections},
        "isp": asdict(isp),
        "stages": stages,
        "artifacts": artifacts,
        "masks": {
            "source_clipped": {
                "artifact": "source_clipped.npy",
                "dtype": "bool",
                "shape": [geometry.height, geometry.width],
                "coordinates": "full sensor raster, before DNG crop or TIFF development",
                "true": "source-supported resampled sRGB pixel has at least one channel at 1",
                "false": "no endpoint detected, or unsupported sensor margin",
                "clipped_pixels": clipped_pixels,
                "limitations": (
                    "Endpoint indicator after ICC and bilinear resampling, not recovered highlight truth; "
                    "resampling can dilute source endpoints and below-endpoint clipping is undetected; "
                    "with SR this describes reconstructed samples, not original endpoints"
                ),
            },
            "source_support": {
                "artifact": "source_support.npy",
                "dtype": "bool",
                "shape": [geometry.height, geometry.width],
                "coordinates": "full sensor raster, before DNG crop or TIFF development",
                "true": "pixel center lies within source-supported geometry",
                "false": "sensor margin with no source support",
                "supported_pixels": stages["raster"]["source_supported_pixels"],
                "limitations": "Geometry support only; not recovered-truth, clipping or learned-detail confidence",
            },
        },
        "limits": {
            "strip_rows": strip_rows,
            "max_decode_bytes": max_decode_bytes,
            "max_develop_bytes": max_develop_bytes,
        },
        "scientific_acceptance": "not measured",
        "pending": [
            "complete PRD configuration",
            "fidelity",
            "learned RAW",
            "sensor nonlinearity and dual conversion gain",
            "full optics",
            "final metadata",
            "Colab full-scale validation",
        ],
    }
    # Render before the completion marker. Hash the HTML in the manifest without
    # making the report depend on its own digest.
    report_path = output / "report.html"
    report_path.write_text(render_report(report), encoding="utf-8")
    artifacts["report.html"] = {
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "size_bytes": report_path.stat().st_size,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
