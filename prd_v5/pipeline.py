"""Single analytic MVP run with disk-backed stages and a completion manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from prd_v5.analytic import AnalyticISP
from prd_v5.decode import decode_source
from prd_v5.dng import write_dng
from prd_v5.geometry import GeometryPlanner
from prd_v5.hypothesis import CameraHypothesis
from prd_v5.inputs import InputInspector
from prd_v5.raster import rasterize
from prd_v5.roundtrip import develop_roundtrip
from prd_v5.sensor import BayerSensor


def run_analytic(
    source_path: str | Path,
    output_dir: str | Path,
    camera: CameraHypothesis,
    isp: AnalyticISP | None = None,
    *,
    target_megapixels: float = 50.0,
    assume_unprofiled_srgb: bool = False,
    strip_rows: int = 64,
    max_decode_bytes: int = 256 * 1024 * 1024,
    max_develop_bytes: int = 512 * 1024 * 1024,
) -> dict:
    """Run explicit bilinear/analytic MVP, never an implicit learned fallback.

    Requires a new output directory to protect prior results. On failure partial
    deliverables remain for diagnosis without a completion manifest. Temporary
    arrays are closed and removed on success or failure. The same frozen camera
    can be supplied to sequential source-resolution runs. Full-scale execution
    belongs in Colab with explicit memory budgets, never on the development Mac.
    """
    if isp is None:
        isp = AnalyticISP()
    for value in (strip_rows, max_decode_bytes, max_develop_bytes):
        if type(value) is not int or value <= 0:
            raise ValueError("Strip height and memory budgets must be positive integers")
    source = InputInspector().inspect(source_path, assume_unprofiled_srgb=assume_unprofiled_srgb)
    geometry = GeometryPlanner().plan(source.width, source.height, target_megapixels=target_megapixels)
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
            shape = (geometry.height, geometry.width)
            rgb = allocate("raster", np.float32, (*shape, 3))
            support = allocate("support", np.bool_, shape)
            stages["raster"] = rasterize(decoded, rgb, support, geometry)
            clean = allocate("clean", np.float32, (*shape, 3))
            stages["analytic"] = isp.process(rgb, clean, strip_rows=strip_rows)
            adc = allocate("adc", np.uint16, shape)
            stages["sensor"] = BayerSensor(camera).process(clean, adc, strip_rows=strip_rows)
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
    artifacts = {}
    for name in ("capture.dng", "developed.tiff", "source_support.npy"):
        path = output / name
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        artifacts[name] = {"sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}
    report = {
        "schema_version": 1,
        "pipeline": "analytic MVP with explicit bilinear resampling; no learned inference",
        "source": source.to_dict(),
        "geometry": geometry.to_dict(),
        "camera": asdict(camera),
        "registry": camera.registry(),
        "isp": asdict(isp),
        "stages": stages,
        "artifacts": artifacts,
        "masks": {
            "source_support": {
                "artifact": "source_support.npy",
                "dtype": "bool",
                "shape": [geometry.height, geometry.width],
                "coordinates": "full sensor raster, before DNG crop or TIFF development",
                "true": "pixel center lies within source-supported geometry",
                "false": "sensor margin with no source support",
                "supported_pixels": stages["raster"]["source_supported_pixels"],
                "limitations": "Geometry support only; not recovered-truth, clipping or learned-detail confidence",
            }
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
            "full sensor",
            "full optics",
            "final metadata",
            "Colab full-scale validation",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
