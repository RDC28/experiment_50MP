"""Explicit exposure hypotheses for PRD_v3's camera milestone and metadata schema."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from synthraw.camera_profile import CameraProfile
from synthraw.config import ExperimentConfig
from synthraw.provenance import FieldRecord, Provenance

# Nominal full stops. Longest shutter and aperture bounds are simulation hypotheses,
# not claims about a particular lens or the camera's bulb operating envelope.
SHUTTER_STOPS = tuple(1 / x for x in (32000, 16000, 8000, 4000, 2000, 1000, 500, 250, 125, 60, 30, 15, 8, 4, 2)) + (
    1.0,
    2.0,
    4.0,
    8.0,
    15.0,
    30.0,
)
APERTURE_STOPS = (1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0)
ISO_STOPS = (50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400)


@dataclass(frozen=True)
class ExposureCandidate:
    """One joint hypothesis; weights are relative heuristic support."""

    exposure_time: float
    f_number: float
    iso: int
    weight: float


@dataclass
class ExposurePosterior:
    """Joint alternatives plus scalar DNG fields for the leading hypothesis."""

    candidates: tuple[ExposureCandidate, ...]
    records: dict[str, FieldRecord]


class ExposurePosteriorEstimator:
    """Rank standard-stop hypotheses, never identify an actual capture exposure."""

    def estimate(
        self,
        profile: CameraProfile,
        *,
        mode: str = "POSTERIOR",
        scene_ev100: float = 10.0,
        manual: tuple[float, float, int] | None = None,
        electronic_shutter: bool = False,
        expanded_iso: bool = False,
        candidate_count: int = 8,
        baseline_exposure: float = 0.0,
        exposure_bias: float = 0.0,
    ) -> ExposurePosterior:
        """Use an explicit EV100 scene-scale hypothesis, not image brightness metering.

        Relative scores combine EV agreement with weak f/5.6, ISO 100, 1/125 s
        priors. They are not calibrated probabilities. Manual values must also
        be standard stops. PROFILE_DEFAULT is the fabricated 1/125, f/5.6, ISO100.
        """
        if mode not in ("POSTERIOR", "MANUAL", "PROFILE_DEFAULT"):
            raise ValueError("Unknown exposure mode")
        if not isinstance(electronic_shutter, bool) or not isinstance(expanded_iso, bool):
            raise ValueError("Exposure toggles must be booleans")
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 2:
            raise ValueError("candidate_count must be an integer >= 2")
        for name, value in (
            ("scene_ev100", scene_ev100),
            ("BaselineExposure", baseline_exposure),
            ("ExposureBiasValue", exposure_bias),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if abs(baseline_exposure) > 5 or abs(exposure_bias) > 5:
            raise ValueError("Exposure offsets must be within -5 to +5 EV")
        if (manual is not None) != (mode == "MANUAL"):
            raise ValueError("MANUAL requires a manual triple; other modes do not accept one")
        camera = profile.camera
        iso_min, iso_max = camera["iso_expanded_range" if expanded_iso else "iso_standard_range"].value
        minimum = camera[
            "electronic_shutter_min_seconds" if electronic_shutter else "mechanical_shutter_min_seconds"
        ].value
        triples = [
            (t, n, iso)
            for t in SHUTTER_STOPS
            if t >= minimum
            for n in APERTURE_STOPS
            for iso in ISO_STOPS
            if iso_min <= iso <= iso_max
        ]
        if not triples:
            raise ValueError("Profile contains no supported standard-stop exposures")
        inferred = mode == "POSTERIOR"
        if inferred:

            def score(triple: tuple[float, float, int]) -> float:
                t, n, iso = triple
                ev = math.log2(n * n / t) - math.log2(iso / 100)
                return (
                    -0.5 * ((ev - scene_ev100 + exposure_bias) / 1.5) ** 2
                    - 0.08 * math.log2(n / 5.6) ** 2
                    - 0.12 * math.log2(iso / 100) ** 2
                    - 0.025 * math.log2(t * 125) ** 2
                )

            ranked = sorted(triples, key=score, reverse=True)[:candidate_count]
            maximum = score(ranked[0])
            weights = [math.exp(score(triple) - maximum) for triple in ranked]
            total = sum(weights)
            candidates = tuple(
                ExposureCandidate(*triple, weight / total) for triple, weight in zip(ranked, weights, strict=True)
            )
        else:
            selected = manual if mode == "MANUAL" else (1 / 125, 5.6, 100)
            if selected not in triples or any(isinstance(value, bool) for value in selected):
                raise ValueError("Exposure triple must use standard stops within profile limits")
            candidates = (ExposureCandidate(*selected, 1.0),)
        records = {}
        for field, attribute, unit in (
            ("ExposureTime", "exposure_time", "s"),
            ("FNumber", "f_number", None),
            ("PhotographicSensitivity", "iso", "ISO"),
        ):
            records[field] = FieldRecord(
                field,
                getattr(candidates[0], attribute),
                Provenance.INFERRED if inferred else Provenance.FABRICATED,
                "Heuristic joint EV100 exposure posterior" if inferred else "Explicit exposure hypothesis",
                unit=unit,
                alternatives=[{"value": getattr(c, attribute), "weight": c.weight} for c in candidates[1:]],
                dependencies=["scene_ev100", "exposure_joint_candidates"] if inferred else [],
                user_override=mode == "MANUAL",
                disclaimer="Hypothetical synthetic exposure; weights are relative, not calibrated probabilities.",
            )
        for field, value in (
            ("BaselineExposure", baseline_exposure),
            ("ExposureBiasValue", exposure_bias),
            ("scene_ev100", scene_ev100),
        ):
            records[field] = FieldRecord(
                field, value, Provenance.FABRICATED, "Explicit simulation exposure prior", unit="EV"
            )
        records["exposure_joint_candidates"] = FieldRecord(
            "exposure_joint_candidates",
            [
                {
                    "ExposureTime": c.exposure_time,
                    "FNumber": c.f_number,
                    "PhotographicSensitivity": c.iso,
                    "weight": c.weight,
                }
                for c in candidates
            ],
            Provenance.INFERRED if inferred else Provenance.FABRICATED,
            "Joint alternatives normalized over the retained candidate set",
        )
        return ExposurePosterior(candidates, records)


# Headroom kept between the brightest reconstructed value and full well over gain.
EXPOSURE_PEAK_MARGIN_EV = 0.5
# LibRaw exp_shift spans 0.25..8, so a developer can undo between -2 and +3 stops of placement.
MIN_DEVELOPMENT_SHIFT_EV = -2.0
MAX_DEVELOPMENT_SHIFT_EV = 3.0


def exposure_and_gain(
    config: ExperimentConfig,
    exposure: Mapping[str, FieldRecord],
    camera: Mapping[str, FieldRecord],
    *,
    hdr_peak: float | None = None,
    peak_margin_ev: float = EXPOSURE_PEAK_MARGIN_EV,
) -> dict[str, FieldRecord]:
    """Radiance-to-photoelectron scale, analogue gain and the BaselineExposure that undoes the placement.

    The leading ISO hypothesis over the profile base ISO (lowest standard ISO) is the analogue
    gain, so the DNG ISO and the simulated shot/read noise agree. With ``hdr_peak`` (the largest
    value entering the sensor, in units where 1.0 is SDR white) the scale places that peak
    ``peak_margin_ev`` below full well divided by the gain, so the brightest reconstructed pixel
    stays below WhiteLevel while the rest of the frame is exposed as brightly as that allows.
    Without a peak the top of the nominal HDR headroom (2^HDR_HIGHLIGHT_EV) is placed there
    instead. ``applied_baseline_exposure_ev`` is the push, in EV, that brings SDR white back to
    display white after that placement; the DNG carries it as BaselineExposure. The placement is
    clamped to the span a developer can undo (MIN/MAX_DEVELOPMENT_SHIFT_EV); see
    ``placement_clamp_note`` for the fallback message when that happens.
    """
    iso = exposure["PhotographicSensitivity"]
    base_iso = camera["iso_standard_range"].value[0]
    gain = iso.value / base_iso
    headroom_ev = config.HDR_HIGHLIGHT_EV if config.HDR_ENABLED else 0.0
    if hdr_peak is not None:
        if not math.isfinite(hdr_peak) or hdr_peak <= 0:
            raise ValueError("hdr_peak must be finite and positive")
        peak = float(hdr_peak)
        method = "the brightest value entering the sensor sits peak_margin_ev below full well / gain"
    else:
        peak = 2.0**headroom_ev
        method = "2^HDR_HIGHLIGHT_EV over analogue gain: the top of the HDR headroom reaches full well / gain"
    placement_ev = min(MAX_DEVELOPMENT_SHIFT_EV, max(MIN_DEVELOPMENT_SHIFT_EV, math.log2(peak) + peak_margin_ev))
    return {
        "applied_analogue_gain": FieldRecord(
            "applied_analogue_gain",
            gain,
            iso.provenance,
            "PhotographicSensitivity hypothesis over the profile base ISO",
            dependencies=["PhotographicSensitivity", "iso_standard_range"],
            disclaimer=iso.disclaimer,
        ),
        "applied_hdr_peak": FieldRecord(
            "applied_hdr_peak",
            peak,
            Provenance.DETERMINISTIC if hdr_peak is not None else Provenance.FABRICATED,
            "Largest optical value placed by the exposure scale; 1.0 is SDR white"
            if hdr_peak is not None
            else "Nominal HDR headroom used because no reconstructed peak was supplied",
        ),
        "applied_exposure_scale": FieldRecord(
            "applied_exposure_scale",
            2.0**-placement_ev / gain,
            iso.provenance,
            method,
            dependencies=["applied_analogue_gain", "applied_hdr_peak"],
            disclaimer=iso.disclaimer,
        ),
        "applied_baseline_exposure_ev": FieldRecord(
            "applied_baseline_exposure_ev",
            placement_ev,
            Provenance.DETERMINISTIC,
            "log2(applied_hdr_peak) + margin, clamped to the developable shift span: the push that "
            "returns SDR white to display white",
            unit="EV",
            dependencies=["applied_hdr_peak"],
        ),
    }


def placement_clamp_note(
    gain_records: Mapping[str, FieldRecord], *, peak_margin_ev: float = EXPOSURE_PEAK_MARGIN_EV
) -> str | None:
    """Fallback message when ``exposure_and_gain`` clamped the placement, else None."""
    peak = gain_records["applied_hdr_peak"].value
    wanted = math.log2(peak) + peak_margin_ev
    applied = gain_records["applied_baseline_exposure_ev"].value
    if math.isclose(wanted, applied, abs_tol=1e-9):
        return None
    if wanted > applied:
        clipped = 2.0**applied
        return (
            f"Exposure placement clamped from {wanted:.2f} to {applied:.2f} EV (developers undo at most "
            f"{MAX_DEVELOPMENT_SHIFT_EV:g} EV): optical values above {clipped:.3g}x SDR white clip at full well"
        )
    return (
        f"Exposure placement clamped from {wanted:.2f} to {applied:.2f} EV (developers undo at least "
        f"{MIN_DEVELOPMENT_SHIFT_EV:g} EV): the frame is exposed darker than full well allows"
    )
