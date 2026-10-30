"""Explicit single-illuminant D65 profile selection for the analytic branch."""

import numpy as np

from prd_v5.analytic import AnalyticISP
from prd_v5.dng import SRGB_TO_XYZ_D65
from prd_v5.fields import resolve_isp
from prd_v5.hypothesis import CameraField


def resolve_d65_profile(
    base: AnalyticISP, xyz_to_camera, *, label: str
) -> tuple[AnalyticISP, tuple[CameraField, ...], dict]:
    """Use a user-supplied XYZ-to-camera matrix without inferring calibration.

    Only D65 is supported: no chromatic adaptation, dual-illuminant interpolation,
    analog balance or camera calibration matrices are implicitly applied. Supply
    a matrix already composed for this simplified model. WB remains separate.
    """
    if not isinstance(base, AnalyticISP):
        raise ValueError("base must be an AnalyticISP")
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a nonempty user profile description")
    matrix = np.asarray(xyz_to_camera, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1e6:
        raise ValueError("xyz_to_camera must be a finite well-conditioned 3x3 matrix")
    # XYZ = inv(ColorMatrix) * camera; linear sRGB = inv(sRGB_to_XYZ) * XYZ.
    camera_to_srgb = np.linalg.solve(np.asarray(SRGB_TO_XYZ_D65), np.linalg.inv(matrix))
    report = {
        "status": "effective",
        "method": "user_supplied_d65_xyz_to_camera",
        "label": label.strip(),
        "illuminant": "D65",
        "xyz_to_camera": matrix.tolist(),
        "camera_to_srgb": camera_to_srgb.tolist(),
        "confidence": None,
        "limitations": (
            "User hypothesis, not verified camera calibration or automatic profile selection. "
            "No chromatic adaptation, dual-illuminant interpolation or auxiliary DNG transforms. "
            "DNG export retains the synthetic virtual-profile designation."
        ),
    }
    candidate = CameraField(
        "camera_to_srgb",
        camera_to_srgb.tolist(),
        None,
        "USER",
        "USER_SET",
        None,
        report,
        "Explicit user D65 matrix converted from XYZ-to-camera",
        report["limitations"],
    )
    isp, records = resolve_isp(base, {"camera_to_srgb": "USER"}, {"camera_to_srgb": (candidate,)})
    return isp, records, report
