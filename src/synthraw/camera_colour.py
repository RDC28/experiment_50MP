"""Matrix camera colour and physics unprocessing for PRD_v3's camera milestone.

Matrices approximate colour response; they do not recover spectral sensitivities.
The default identity camera-to-working transform is a virtual prior, not Sony calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthraw.provenance import FieldRecord, Provenance

_SRGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
)
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def _matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.array(value, dtype=float, copy=True)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 3x3 camera matrix")
    if np.linalg.cond(matrix) > 1e6:
        raise ValueError("Camera matrix is singular or ill-conditioned")
    matrix.setflags(write=False)
    return matrix


class CameraColourModel:
    """Invert a supplied camera-to-linear-sRGB fit, with explicit matrix provenance."""

    def __init__(self, camera_to_working: FieldRecord | None = None) -> None:
        if camera_to_working is None:
            camera_to_working = FieldRecord(
                "camera_to_working",
                np.eye(3).tolist(),
                Provenance.FABRICATED,
                "Virtual identity camera-to-linear-sRGB prior",
                disclaimer="Not measured from a Sony Alpha 1 II sensor.",
            )
        camera_to_working.validate()
        self.camera_to_working = _matrix(camera_to_working.value)
        self.working_to_camera = _matrix(np.linalg.inv(self.camera_to_working))
        # Snapshot the record to keep metadata consistent if the caller edits its record.
        self.source = FieldRecord.from_dict(camera_to_working.to_dict())
        self.source.value = self.camera_to_working.tolist()

    def metadata(self) -> dict[str, FieldRecord]:
        """Generate DNG XYZ-to-camera matrices using Bradford white adaptation.

        ColorMatrix1 accepts XYZ under Standard A; ColorMatrix2 accepts D65 XYZ.
        Both describe the same hypothetical linear camera, before white balancing.
        """
        d65 = _SRGB_TO_XYZ @ np.ones(3)
        standard_a = np.array([1.09850, 1.0, 0.35585])
        adaptation = np.linalg.solve(_BRADFORD, np.diag((_BRADFORD @ d65) / (_BRADFORD @ standard_a)) @ _BRADFORD)
        xyz_to_camera = self.working_to_camera @ np.linalg.inv(_SRGB_TO_XYZ)
        values = {
            "ColorMatrix1": (xyz_to_camera @ adaptation).tolist(),
            "ColorMatrix2": xyz_to_camera.tolist(),
            "CalibrationIlluminant1": 17,
            "CalibrationIlluminant2": 21,
            "camera_matrix_condition_number": float(np.linalg.cond(self.camera_to_working)),
        }
        values["ColorMatrix1_condition_number"] = float(np.linalg.cond(values["ColorMatrix1"]))
        values["ColorMatrix2_condition_number"] = float(np.linalg.cond(values["ColorMatrix2"]))
        records = {self.source.field: FieldRecord.from_dict(self.source.to_dict())}
        records.update(
            {
                name: FieldRecord(
                    name,
                    value,
                    Provenance.DETERMINISTIC,
                    "DNG XYZ-to-camera transform; Bradford Standard A to D65 adaptation",
                    dependencies=[self.source.field],
                    disclaimer="Matrix approximation only; source provenance remains in camera_to_working.",
                )
                for name, value in values.items()
            }
        )
        return records


@dataclass
class InverseISPResult:
    """Unclipped camera RGB and applied gain metadata."""

    camera_rgb: np.ndarray
    records: dict[str, FieldRecord]


class InverseISPEngine:
    """Brooks-style matrix, inverse WB and inverse digital gain (PRD_v3)."""

    def reconstruct(
        self,
        scene_linear_rgb: np.ndarray,
        model: CameraColourModel,
        *,
        white_balance_gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
        digital_gain: float = 1.0,
        use_raw_flow: bool = False,
        use_spiraldiff: bool = False,
    ) -> InverseISPResult:
        """Unprocess linear sRGB; WB gains must be in camera channel coordinates.

        Signed out-of-gamut values and HDR headroom are preserved. The downstream
        electron simulator must explicitly handle negative sensor expectations.
        No inverse tone curve is applied here: HDR reconstruction already did it.
        Experimental learned toggles are gated stubs: they retain PHYSICS and
        report the fallback and cross-camera domain without loading any model.
        """
        if not isinstance(use_raw_flow, bool) or not isinstance(use_spiraldiff, bool):
            raise ValueError("Learned inverse-ISP toggles must be booleans")
        rgb = np.asarray(scene_linear_rgb, dtype=float)
        gains = np.asarray(white_balance_gains, dtype=float)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or not rgb.size or not np.isfinite(rgb).all():
            raise ValueError("Expected nonempty finite HxWx3 scene-linear RGB")
        if gains.shape != (3,) or not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("White-balance gains must be three finite positive values")
        if not np.isscalar(digital_gain) or not np.isfinite(digital_gain) or digital_gain <= 0:
            raise ValueError("digital_gain must be finite and positive")
        camera = (rgb @ model.working_to_camera.T) / gains / digital_gain
        if not np.isfinite(camera).all():
            raise ValueError("Inverse ISP overflowed")
        records = model.metadata()
        for name, value in {
            "AsShotNeutral": (gains[1] / gains).tolist(),
            "inverse_isp_white_balance_gains": gains.tolist(),
            "inverse_isp_digital_gain": float(digital_gain),
        }.items():
            records[name] = FieldRecord(
                name,
                value,
                Provenance.DETERMINISTIC if name == "AsShotNeutral" else Provenance.FABRICATED,
                "Supplied camera-channel gain hypothesis; neutral normalized to green",
                dependencies=["inverse_isp_white_balance_gains"] if name == "AsShotNeutral" else [],
                user_override=name != "AsShotNeutral",
            )
        requested = [name for name, enabled in (("RAW_FLOW", use_raw_flow), ("SPIRALDIFF", use_spiraldiff)) if enabled]
        selection = {
            "inverse_isp_backend": "PHYSICS",
            "inverse_isp_requested_priors": requested,
            "inverse_isp_fallback_reason": (
                "Learned inverse-ISP adapters are unavailable; no supplied checkpoint camera domain has been verified. "
                "Using the PHYSICS baseline without a learned prior."
                if requested
                else None
            ),
            "inverse_isp_camera_domain": "cross-camera" if requested else "virtual matrix camera",
        }
        for name, value in selection.items():
            records[name] = FieldRecord(
                name,
                value,
                Provenance.DETERMINISTIC,
                "PRD_v3 learned inverse-ISP gate; no model import or checkpoint download",
                dependencies=["inverse_isp_requested_priors"] if name != "inverse_isp_requested_priors" else [],
            )
        return InverseISPResult(camera, records)
