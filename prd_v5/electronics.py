"""Explicit experimental ISO-switched electronics, without inferred calibration."""

from dataclasses import asdict, dataclass

from prd_v5.fields import resolve_camera
from prd_v5.hypothesis import CameraField, CameraHypothesis


@dataclass(frozen=True)
class DualConversionProfile:
    """Two supplied effective charge/read regimes, switched at ISO >= threshold.

    The low-gain full well sets the photon scale and total DN per electron at
    every ISO. Switching changes only read noise and the in-pixel conversion
    gain regime, whose charge capacity must hold the ADC-saturating charge.
    This is a simplified effective model, not simultaneous dual-gain readout or
    measured Sony electronics.
    """

    label: str
    switch_iso: int
    low_full_well_e: float
    low_read_noise_e: float
    high_full_well_e: float
    high_read_noise_e: float

    def __post_init__(self):
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must describe the supplied profile")
        if type(self.switch_iso) is not int or not 101 <= self.switch_iso <= 32000:
            raise ValueError("switch_iso must be an integer in 101..32000")
        for regime in ("low", "high"):
            CameraHypothesis(
                full_well_e=getattr(self, f"{regime}_full_well_e"),
                read_noise_e=getattr(self, f"{regime}_read_noise_e"),
            )
        if self.high_full_well_e < self.low_full_well_e * 100 / self.switch_iso:
            raise ValueError("high_full_well_e must hold the ADC-saturating charge at switch_iso")

    def resolve(self, base: CameraHypothesis, *, accept_experimental: bool = False):
        """Return immutable camera, selection records and strict-JSON evidence.

        Resolve once before paired runs. Save the selected camera in normal run
        config and pass records as camera_selections to retain profile evidence.
        No defaults, random streams, ISO or identity fields are changed.
        """
        if not isinstance(base, CameraHypothesis):
            raise ValueError("base must be a CameraHypothesis")
        if accept_experimental is not True:
            raise ValueError("Explicit experimental acceptance is required")
        regime = "high" if base.iso >= self.switch_iso else "low"
        values = {"full_well_e": self.low_full_well_e, "read_noise_e": getattr(self, f"{regime}_read_noise_e")}
        report = {
            "status": "experimental",
            "profile": asdict(self),
            "selected_regime": regime,
            "iso": base.iso,
            "selection_rule": "high when ISO >= switch_iso; otherwise low",
            "charge_capacity_e": getattr(self, f"{regime}_full_well_e"),
            "adc_saturation_e": self.low_full_well_e * min(1, 100 / base.iso),
            "zero_effect": all(getattr(base, name) == value for name, value in values.items()),
            "limitations": (
                "User-supplied effective electronics hypothesis, not Sony calibration. "
                "Total gain remains ISO/100 over the low-gain well; switching never changes photon counts. "
                "No simultaneous dual readout. "
                "Physical photon-transfer and full-scale validation pending."
            ),
        }
        candidates = {
            name: (
                CameraField(
                    name,
                    value,
                    base.registry()[name]["units"],
                    "USER",
                    "USER_SET",
                    None,
                    report,
                    "Explicit experimental ISO-switched electronics profile",
                    report["limitations"],
                ),
            )
            for name, value in values.items()
        }
        camera, records = resolve_camera(base, dict.fromkeys(values, "USER"), candidates)
        return camera, records, report
