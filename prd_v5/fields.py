"""Resolve field modes from actual evidence, without inventing estimators."""

import copy
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from prd_v5.hypothesis import CameraField, FieldMode

if TYPE_CHECKING:
    from prd_v5.analytic import AnalyticISP
    from prd_v5.hypothesis import CameraHypothesis

# Identity aliases used by the PRD and existing camera registry.
USER_ONLY_FIELDS = frozenset({"gps", "serial_number", "camera_serial", "lens_serial", "timestamp", "shutter_count"})


def resolve_field(
    name: str,
    units: str | None,
    mode: FieldMode,
    candidates: Iterable[CameraField] = (),
    *,
    fabricate_unobservable_metadata: bool = False,
    unobservable_metadata: bool = False,
) -> CameraField:
    """Select supplied evidence and retain the requested mode in the result.

    AUTO prefers user input, observation, derivation, estimation, then declared
    priors/synthetic hypotheses. ESTIMATE never falls back to a prior. FABRICATE
    selects only explicitly supplied synthetic/prior values. Missing evidence is
    UNSPECIFIED, not a fabricated estimate. No model or sampling runs here.

    Callers must domain-validate candidates before use in camera/ISP configuration.
    The metadata fabrication gate applies in AUTO as well as FABRICATE. Identity
    fields require USER mode even when the fabrication gate is enabled.
    """
    if type(fabricate_unobservable_metadata) is not bool or type(unobservable_metadata) is not bool:
        raise ValueError("Metadata policy switches must be booleans")
    empty = CameraField(
        name,
        None,
        units,
        mode,
        "UNSPECIFIED",
        None,
        None,
        "No eligible supplied evidence for requested field mode",
        "Unresolved value must not be passed as an effective camera parameter",
    )
    if name.casefold() in USER_ONLY_FIELDS and mode != "USER":
        raise ValueError(f"{name} requires USER mode")
    priorities = {
        "AUTO": ("USER_SET", "OBSERVED", "DERIVED", "ESTIMATED", "PRIOR_SAMPLED", "SYNTHETIC"),
        "ESTIMATE": ("OBSERVED", "DERIVED", "ESTIMATED"),
        "FABRICATE": ("PRIOR_SAMPLED", "SYNTHETIC"),
        "USER": ("USER_SET",),
    }[mode]
    supplied = {}
    for candidate in candidates:
        if not isinstance(candidate, CameraField) or candidate.name != name or candidate.units != units:
            raise ValueError("Candidates must be CameraFields with matching name and units")
        if candidate.provenance in supplied:
            raise ValueError("Ambiguous duplicate candidate provenance")
        supplied[candidate.provenance] = candidate
    for provenance in priorities:
        if (
            unobservable_metadata
            and not fabricate_unobservable_metadata
            and provenance in ("PRIOR_SAMPLED", "SYNTHETIC")
        ):
            continue
        candidate = supplied.get(provenance)
        if candidate is not None and candidate.value is not None:
            # Detach nested JSON values as well as the immutable record itself.
            return replace(copy.deepcopy(candidate), mode=mode)
    return empty


def resolve_camera(
    base: "CameraHypothesis",
    modes: "Mapping[str, FieldMode]",
    candidates: "Mapping[str, Iterable[CameraField]] | None" = None,
) -> tuple["CameraHypothesis", tuple[CameraField, ...]]:
    """Resolve executable camera inputs atomically against a shared hypothesis.

    Unspecified modes use AUTO with the base's explicit synthetic hypothesis.
    Supplied candidate lists replace that default evidence for their field.
    Missing estimates/user values fail rather than silently reuse the base.
    Derived electronics are recomputed by CameraHypothesis, never overridden.
    Returned records describe input selection only; existing runner provenance
    does not automatically incorporate them. No estimation or sampling runs.
    """
    from dataclasses import fields

    from prd_v5.hypothesis import CameraHypothesis

    if not isinstance(base, CameraHypothesis):
        raise ValueError("base must be a CameraHypothesis")
    names = {field.name for field in fields(base)}
    supplied = {} if candidates is None else dict(candidates)
    unknown = (set(modes) | set(supplied)) - names
    if unknown:
        raise ValueError(f"Unknown or derived camera inputs: {sorted(unknown)}")
    registry = base.registry()
    resolved = []
    for field in fields(base):
        default = CameraField(**registry[field.name])
        result = resolve_field(
            field.name,
            default.units,
            modes.get(field.name, "AUTO"),
            supplied.get(field.name, (default,)),
        )
        if result.value is None:
            raise ValueError(f"Unresolved camera input: {field.name} ({result.mode})")
        resolved.append(result)
    # Construct only after every field resolves; this enforces coupled domains.
    camera = CameraHypothesis(**{record.name: record.value for record in resolved})
    return camera, tuple(resolved)


def resolve_isp(
    base: "AnalyticISP",
    modes: Mapping[str, FieldMode],
    candidates: Mapping[str, Iterable[CameraField]] | None = None,
) -> tuple["AnalyticISP", tuple[CameraField, ...]]:
    """Resolve analytic ISP evidence before constructing a validated hypothesis.

    Names are unprefixed AnalyticISP inputs; gains use ratio units, other inputs
    use None, matching the analytic registry. Explicit lists replace base
    evidence. No estimator runs and no missing value falls back implicitly.
    Selection records remain separate from the runner's synthetic registry.
    """
    from dataclasses import fields

    from prd_v5.analytic import AnalyticISP

    if not isinstance(base, AnalyticISP):
        raise ValueError("base must be an AnalyticISP")
    names = {field.name for field in fields(base)}
    supplied = {} if candidates is None else dict(candidates)
    unknown = (set(modes) | set(supplied)) - names
    if unknown:
        raise ValueError(f"Unknown ISP inputs: {sorted(unknown)}")
    resolved = []
    for field in fields(base):
        units = "ratio" if field.name in ("white_balance", "exposure_gain") else None
        default = CameraField(
            field.name,
            getattr(base, field.name),
            units,
            "AUTO",
            "SYNTHETIC",
            None,
            None,
            "Supplied analytic forward-render hypothesis; no camera calibration inferred",
            "AnalyticISP domain and matrix conditioning checks; radiance accuracy not measured",
        )
        result = resolve_field(field.name, units, modes.get(field.name, "AUTO"), supplied.get(field.name, (default,)))
        if result.value is None:
            raise ValueError(f"Unresolved ISP input: {field.name} ({result.mode})")
        resolved.append(result)
    isp = AnalyticISP(**{record.name: record.value for record in resolved})
    return isp, tuple(resolved)
