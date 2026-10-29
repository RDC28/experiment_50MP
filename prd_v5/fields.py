"""Resolve field modes from actual evidence, without inventing estimators."""

import copy
from collections.abc import Iterable
from dataclasses import replace

from prd_v5.hypothesis import CameraField, FieldMode

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
