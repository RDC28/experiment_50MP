"""Provenance for the executed analytic path, without inferred capture truth."""

from dataclasses import asdict

from prd_v5.analytic import AnalyticISP
from prd_v5.geometry import GeometryPlan
from prd_v5.hypothesis import CameraField, CameraHypothesis
from prd_v5.inputs import SourceRecord


def analytic_registry(
    camera: CameraHypothesis, source: SourceRecord, geometry: GeometryPlan, isp: AnalyticISP
) -> dict[str, dict]:
    """Return detached records; namespace non-camera values to avoid collisions.

    These describe supplied hypotheses and arithmetic, not parameter estimation.
    No private source EXIF or filesystem path is copied into field records.
    Confidence and posterior remain null because no inference was performed.
    """
    records = camera.registry()

    def add(name, value, units, provenance, method, validation):
        records[name] = asdict(CameraField(name, value, units, "AUTO", provenance, None, None, method, validation))

    for name, units in (
        ("sha256", None),
        ("size_bytes", "bytes"),
        ("width", "pixels"),
        ("height", "pixels"),
        ("image_format", None),
        ("encoded_bits_per_channel", "bits"),
        ("icc_sha256", None),
    ):
        add(
            f"source.{name}",
            getattr(source, name),
            units,
            "OBSERVED",
            "Streaming file hash or stored source header; not capture metadata",
            "InputInspector header checks; source must remain unchanged during run",
        )
    add(
        "source.colour_policy",
        source.colour_policy,
        None,
        "SYNTHETIC",
        "Explicit decoding policy; an unprofiled sRGB assumption is not observed colour truth",
        "Decoder policy admission; calibrated colour accuracy not measured",
    )
    for name, value in asdict(isp).items():
        add(
            f"isp.{name}",
            value,
            "ratio" if name in ("white_balance", "exposure_gain") else None,
            "SYNTHETIC",
            "Supplied analytic forward-render hypothesis; no camera calibration inferred",
            "AnalyticISP domain and matrix conditioning checks; radiance accuracy not measured",
        )
    for name in ("mode", "fit_policy", "requested_megapixels"):
        add(
            f"geometry.{name}",
            getattr(geometry, name),
            "MP" if name == "requested_megapixels" else None,
            "SYNTHETIC",
            "Explicit geometry request",
            "GeometryPlanner request-domain checks",
        )
    for name in (
        "width",
        "height",
        "actual_megapixels",
        "scale",
        "offset_xy",
        "source_crop",
        "content_box",
        "default_crop_origin",
        "default_crop_size",
        "active_area",
    ):
        units = "MP" if name == "actual_megapixels" else "ratio" if name == "scale" else "pixels"
        add(
            f"geometry.{name}",
            getattr(geometry, name),
            units,
            "DERIVED",
            "GeometryPlanner transform; source_crop uses source pixels, other positions use sensor pixels",
            "Arithmetic geometry checks; not evidence of recovered scene content",
        )
    return records
