"""Executable subset of the notebook's PRD field catalogue."""

from prd_v5.fields import resolve_camera
from prd_v5.hypothesis import CameraField, CameraHypothesis

CAMERA_FIELDS = {
    "CFA pattern": "cfa_pattern",
    "ADC bit depth": "adc_bits",
    "Black level": "black_level",
    "Full-well capacity": "full_well_e",
    "ISO": "iso",
    "Shutter time": "shutter_s",
    "Aperture": "aperture",
    "Random seed": "seed",
}


def resolve_camera_panel(base: CameraHypothesis, requests: list[dict], modes: dict):
    """Resolve supported fields; reject unsupported edits before pixel work.

    Explicit catalogue values require USER mode and are labelled USER_SET.
    AUTO/FABRICATE without values use the declared synthetic base hypothesis.
    ESTIMATE has no estimator and therefore fails rather than inventing evidence.
    Read-noise amplitude remains in CAMERA_VALUES; its PRD row combines multiple
    distributions and is deliberately not treated as a scalar override.
    """
    names = [field["name"] for field in requests]
    if len(set(names)) != len(names) or set(modes) != set(names):
        raise ValueError("Panel requires unique fields and exactly one mode per field")
    selected_modes, candidates = {}, {}
    registry = base.registry()
    for field in requests:
        name = field["name"]
        mode = modes[name]
        if mode not in {"AUTO", "ESTIMATE", "FABRICATE", "USER"}:
            raise ValueError(f"Invalid panel mode: {name}")
        value = field["value"]
        target = CAMERA_FIELDS.get(name)
        if target is None:
            if value is not None or mode != field["mode"]:
                raise ValueError(f"Unsupported catalogue override: {name}")
            continue
        selected_modes[target] = mode
        if value is not None:
            if mode != "USER":
                raise ValueError(f"Explicit panel value requires USER mode: {name}")
            candidates[target] = (
                CameraField(
                    target,
                    value,
                    registry[target]["units"],
                    "USER",
                    "USER_SET",
                    None,
                    None,
                    "User-supplied notebook catalogue hypothesis",
                    "CameraHypothesis domain checks",
                ),
            )
    return resolve_camera(base, selected_modes, candidates)
