"""Audit HDR truth claims against original-source uncertainty, without models."""

from __future__ import annotations

import numpy as np


def validate_hdr_claims(source_clipped, source_support, claimed_truth=None, *, strip_rows=64):
    """Reject recovered-truth claims at source endpoints or unsupported pixels.

    Inputs are aligned two-dimensional boolean arrays (including NPY memmaps).
    source_clipped must describe original rendered-source endpoints, not endpoints
    recomputed after SR or inverse tone mapping. A claim mask is an assertion to
    audit, never evidence that its remaining pixels are radiometrically correct.
    Missing reference truth always leaves numerical HDR fidelity not measured.
    """
    if type(strip_rows) is not int or strip_rows <= 0:
        raise ValueError("strip_rows must be a positive integer")
    arrays = [source_clipped, source_support]
    if claimed_truth is not None:
        arrays.append(claimed_truth)
    shape = source_clipped.shape
    if len(shape) != 2 or not all(shape):
        raise ValueError("HDR masks must be nonempty two-dimensional arrays")
    if any(array.shape != shape or array.dtype != np.bool_ for array in arrays):
        raise ValueError("HDR masks must be aligned boolean arrays")
    clipped_count = unsupported_count = claimed_count = forbidden_count = 0
    for top in range(0, shape[0], strip_rows):
        clipped = source_clipped[top : top + strip_rows]
        support = source_support[top : top + strip_rows]
        clipped_count += int(np.count_nonzero(clipped))
        unsupported_count += int(np.count_nonzero(~support))
        if claimed_truth is not None:
            claims = claimed_truth[top : top + strip_rows]
            claimed_count += int(np.count_nonzero(claims))
            forbidden_count += int(np.count_nonzero(claims & (clipped | ~support)))
    return {
        "status": "failed" if forbidden_count else "not measured",
        "reason": (
            "Recovered-truth claims intersect clipped source or unsupported geometry"
            if forbidden_count
            else "No aligned calibrated HDR reference; radiometric accuracy remains not measured"
        ),
        "claim_policy": "failed" if forbidden_count else "passed: no forbidden recovered-truth claims",
        "pixels": int(shape[0] * shape[1]),
        "source_endpoint_pixels": clipped_count,
        "unsupported_pixels": unsupported_count,
        "claimed_truth_pixels": claimed_count,
        "forbidden_claim_pixels": forbidden_count,
        "numerical_hdr_fidelity": "not measured",
        "limitations": (
            "Endpoint mask also includes black endpoints; non-endpoint values are not proof of truth. "
            "Generated photosite saturation, expanded dynamic range and model confidence do not establish recovery."
        ),
    }
