"""Licence-gated stubs for the PRD_v3 experimental milestone.

SUPIR, ExpandDiff, Vid2HDR, RAW-Flow, SpiralDiff and spectral inference from an
AI image have no verified adapter or pinned checkpoint. Requesting one warns about
its licence, never downloads or imports a model, and returns the deterministic
fallback with the reason recorded. No result is ever labelled Sony alpha 1 II.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from synthraw.model_manifest import MODEL_MANIFEST
from synthraw.provenance import FieldRecord, Provenance

CAMERA_DOMAIN = "cross-camera prior; not a Sony alpha 1 II model"


@dataclass(frozen=True)
class ExperimentalBackend:
    """A PRD experimental option, its manifest entry and its deterministic fallback."""

    name: str
    toggle: str
    manifest_name: str
    fallback: str


EXPERIMENTAL_BACKENDS = {
    backend.name: backend
    for backend in (
        ExperimentalBackend("SUPIR", 'SR_BACKEND="SUPIR"', "SUPIR", "SWINIR"),
        ExperimentalBackend("EXPANDDIFF", 'HDR_BACKEND="EXPANDDIFF"', "EXPANDDIFF", "DETERMINISTIC"),
        ExperimentalBackend("VID2HDR", 'HDR_BACKEND="VID2HDR"', "VID2HDR", "DETERMINISTIC"),
        ExperimentalBackend("RAW_FLOW", "USE_RAW_FLOW", "RAW_FLOW", "PHYSICS"),
        ExperimentalBackend("SPIRALDIFF", "USE_SPIRALDIFF", "SPIRALDIFF", "PHYSICS"),
        ExperimentalBackend(
            "INFER_SPECTRAL_RESPONSE_FROM_AI_IMAGE",
            "INFER_SPECTRAL_RESPONSE_FROM_AI_IMAGE",
            "SPECTRAL_FROM_AI_IMAGE",
            "MATRIX_ONLY",
        ),
    )
}


@dataclass(frozen=True)
class ExperimentalDecision:
    """The recorded outcome of requesting one experimental backend."""

    requested: str
    backend: str
    status: str
    licence: str
    licence_accepted: bool
    fallback_reason: str
    camera_domain: str = CAMERA_DOMAIN


def request_experimental(name: str, *, accept_licence: bool = False) -> ExperimentalDecision:
    """Warn about the licence first, then fall back; nothing is downloaded or imported.

    Even a pinned manifest entry falls back because no adapter has been verified
    against its checkpoint camera domain (PRD_v3 learned inverse-ISP rules).
    """
    backend = EXPERIMENTAL_BACKENDS[name]
    spec = MODEL_MANIFEST[backend.manifest_name]
    warnings.warn(f"{name} licence restrictions: {spec.licence}", UserWarning, stacklevel=2)
    reason = (
        f"{name} is an experimental licence-gated stub ({spec.status}), unavailable without a verified adapter. "
        f"Using the {backend.fallback} fallback."
    )
    return ExperimentalDecision(name, backend.fallback, spec.status, spec.licence, accept_licence, reason)


def spectral_inference_records(enabled: bool, *, accept_licence: bool = False) -> list[FieldRecord]:
    """EXPERIMENTAL / LOW CONFIDENCE spectral inference; never replaces calibrated values."""
    if not isinstance(enabled, bool):
        raise ValueError("INFER_SPECTRAL_RESPONSE_FROM_AI_IMAGE must be a boolean")
    decision = (
        request_experimental("INFER_SPECTRAL_RESPONSE_FROM_AI_IMAGE", accept_licence=accept_licence)
        if enabled
        else None
    )
    method = "PRD_v3 EXPERIMENTAL / LOW CONFIDENCE spectral inference gate; no model import or download"
    return [
        FieldRecord("spectral_inference_requested", enabled, Provenance.DETERMINISTIC, method),
        FieldRecord(
            "spectral_inference_backend",
            decision.backend if decision else None,
            Provenance.DETERMINISTIC,
            method,
            dependencies=["spectral_inference_requested"],
        ),
        FieldRecord(
            "spectral_inference_fallback_reason",
            decision.fallback_reason if decision else None,
            Provenance.DETERMINISTIC,
            method,
            dependencies=["spectral_inference_requested"],
            disclaimer="Spectral response is never inferred from an AI image in place of calibrated curves.",
        ),
    ]
