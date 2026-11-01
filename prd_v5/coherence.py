"""Read exported DNG metadata independently of the intended tag records."""

from __future__ import annotations

import numpy as np
import tifffile

# PRD catalogue names that differ from their corresponding standard tag names.
ALIASES = {
    "Black level": "BlackLevel",
    "White level": "WhiteLevel",
    "ISO": "PhotographicSensitivity",
    "Shutter time": "ExposureTime",
    "Aperture": "FNumber",
}
GROUPS = {
    "ColorMatrix1/2": ("ColorMatrix1", "ColorMatrix2"),
    "ForwardMatrix1/2": ("ForwardMatrix1", "ForwardMatrix2"),
    "CameraCalibration1/2": ("CameraCalibration1", "CameraCalibration2"),
    "CalibrationIlluminant1/2": ("CalibrationIlluminant1", "CalibrationIlluminant2"),
}


def metadata_coherence(path, registry, tags):
    """Compare all emitted tags with both manifest and complete registry.

    Numeric comparison allows 1e-6 absolute error for the writer's rational
    serialization. Integer/string tags remain exact. This checks serialization,
    not physical calibration or parameters without a standard DNG representation.
    Only tag directories are read; no pixel raster is loaded.
    """
    checked, failures, mapped = [], [], set()

    def compare(label, expected, actual, kind):
        if kind == "ASCII":
            matches = isinstance(expected, str) and expected == actual
        else:
            try:
                left = np.asarray(expected, dtype=float).reshape(-1)
                right = np.asarray(actual, dtype=float).reshape(-1)
                tolerance = 1e-6 if kind in {"RATIONAL", "SRATIONAL", "DOUBLE"} else 0
                matches = (
                    left.shape == right.shape
                    and np.all(np.isfinite(left))
                    and np.all(np.isfinite(right))
                    and np.allclose(left, right, rtol=0, atol=tolerance)
                )
            except (TypeError, ValueError):
                matches = False
        checked.append(label)
        if not matches:
            failures.append(label)

    actuals, kinds = {}, {}
    with tifffile.TiffFile(path) as container:
        page = container.pages[0]
        for name, record in tags.items():
            kind = record["dng_type"].split("[")[0]
            kinds[name] = kind
            tag = page.tags.get(tifffile.TIFF.TAGS[name])
            actual = None if tag is None else tag.value
            if actual is not None and kind != "ASCII":
                actual = np.asarray(tuple(actual) if isinstance(actual, bytes) else actual).reshape(-1)
                if kind in {"RATIONAL", "SRATIONAL"}:
                    pairs = actual.reshape(-1, 2)
                    actual = np.divide(
                        pairs[:, 0], pairs[:, 1], out=np.full(len(pairs), np.nan), where=pairs[:, 1] != 0
                    )
            actuals[name] = actual
            compare(f"manifest.{name}", record["value"], actual, kind)
            key = f"dng.{name}"
            mapped.add(key)
            compare(key, registry.get(key, {}).get("value"), actual, kind)
            for field in (name, *(alias for alias, target in ALIASES.items() if target == name)):
                if field in registry:
                    mapped.add(field)
                    compare(field, registry[field]["value"], actual, kind)
        # An intentionally omitted optional tag must remain absent in the file.
        for key, record in registry.items():
            if key.startswith("dng.") and key[4:] not in tags:
                name = key[4:]
                mapped.add(key)
                checked.append(key)
                if record["value"] is not None or page.tags.get(tifffile.TIFF.TAGS[name]) is not None:
                    failures.append(key)
        for field, names in GROUPS.items():
            if field in registry and any(name in tags for name in names):
                mapped.add(field)
                values = registry[field]["value"]
                if not isinstance(values, (list, tuple)) or len(values) != len(names):
                    checked.append(field)
                    failures.append(field)
                    continue
                for name, expected in zip(names, values, strict=True):
                    compare(f"{field}:{name}", expected, actuals.get(name), kinds.get(name, "DOUBLE"))
    return {
        "status": "failed" if failures else "passed",
        "reason": "Exported DNG tags compared with manifest and registry; physical calibration is not established",
        "scope": "Emitted standard tags and mapped registry fields; some simulator parameters have no DNG tag",
        "numeric_absolute_tolerance": 1e-6,
        "checked": checked,
        "mismatches": failures,
        "unmapped_registry_fields": sorted(set(registry) - mapped),
    }
