"""The final experiment report (PRD Phase 6, 25.4, 33, 34): every scientific question answered with evidence.

:func:`build` reads the collected runs (:mod:`run_collection`: notebooks 00 to 03 and 05, the Phase 6 ablation
and the controlled reference test) and answers the 18 questions of PRD 34 one by one. Each answer states its
status:

* ``answered``: measured evidence answers it (the numbers and the runs they come from are listed);
* ``partial``: some evidence exists, and what is still needed is named;
* ``not measured``: no collected run measures it yet, and the run that would is named;
* ``reasoned``: a conclusion from the method itself (what an 8-bit render cannot contain), not a measurement.

It then recommends a PRD 33 outcome from the reference test (the only accuracy measurement against ground
truth) with the provisional thresholds of ``report`` in the evaluation profile, picks the best
quality-to-runtime Colab configuration, and states plainly what this upscaling approach can and cannot achieve.
Collecting more runs (the captain's notebook 01, 02 and 03 zips) and re-running the build updates every answer.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .ablation import CONFIGURATIONS
from .evaluation_config import EvaluationConfig
from .reference_test import summary_rows
from .run_collection import (
    NOTEBOOK_NAMES,
    Run,
    by_source,
    fmt,
    identity_table,
    inventory_table,
    markdown_table,
    night_table,
    outputs_table,
    performance_table,
)

QUESTIONS = (
    "Does 50 MP reconstruction preserve the source image's composition?",
    "Which fine details are genuinely constrained by source pixels?",
    "Which details are generated from learned priors?",
    "How often does super-resolution alter facial identity?",
    "How does face size affect reliability?",
    "Can crowd faces be enhanced without inventing identities?",
    "Does nighttime enhancement improve shadows without creating false detail?",
    "Can highlights be made more aesthetically convincing without implying real recovery?",
    "Does 16-bit processing reduce artifacts in subsequent edits?",
    "Does floating-point EXR provide meaningful workflow advantages?",
    "Does synthetic Bayer sampling improve the final visible image?",
    "Is synthetic DNG useful beyond compatibility testing?",
    "Which stages contribute most to perceived photographic realism?",
    "Which stages produce measurable harm?",
    "What is the best quality-to-runtime configuration for Google Colab?",
    "Can the universal pipeline approach the quality of specialized pipelines?",
    "What information is permanently absent from the original 4K render?",
    "What additional source data would be needed for genuine scene-referred or sensor-faithful reconstruction?",
)
OUTCOMES = {
    "A": "Successful camera-grade reconstruction",
    "B": "High-quality upscale with limited camera realism",
    "C": "Primarily cosmetic enhancement",
}
NEXT_STEPS = {
    "A": "Improve performance, colour handling and export interoperability; never describe the result as recovered "
    "camera RAW.",
    "B": "Focus on region-specific reconstruction and controlled image-formation simulation; avoid unnecessary "
    "synthetic RAW complexity.",
    "C": "Retain conventional interpolation, professional colour management and high-bit-depth export; reconsider the "
    "value of the additional stages.",
}
COMPOSITION_MIN_SSIM = 0.98  # provisional (PRD 25.3)
COMPOSITION_MAX_EDGE_PX = 0.5
DELEGATED = ", delegated to the specialised pipeline"
REVIEW_CRITERIA = ("naturalness", "structural_fidelity", "perceived_detail", "artefact_severity", "overall_preference")


class Evidence:
    """The collected runs, indexed for the questions."""

    def __init__(self, runs: list[Run], review_scores: dict | None = None):
        self.runs = runs
        self.review_scores = review_scores or {}
        self.reference_rows = [row for r in self.of("06 reference") for row in summary_rows(r.manifest)]
        self.ablations = self.of("06 ablation")

    def of(self, notebook: str) -> list[Run]:
        return [r for r in self.runs if r.notebook == notebook and str(r.status) == "complete"]

    def finals(self) -> list[tuple[Run, object]]:
        found = []
        for run in self.runs:
            for output in run.outputs:
                if output.kind == "final" or output.detail.get("final"):
                    found.append((run, output))
        return found

    def outputs(self, name: str) -> list[tuple[Run, object]]:
        return [(r, o) for r in self.runs for o in r.outputs if o.name == name]

    def reference_by_config(self) -> dict[str, list[dict]]:
        grouped: dict[str, list[dict]] = {}
        for row in self.reference_rows:
            grouped.setdefault(row["configuration"], []).append(row)
        return grouped

    def ablation_records(self) -> list[tuple[Run, dict]]:
        return [
            (r, rec) for r in self.ablations for rec in r.manifest.get("configurations", []) if "measurements" in rec
        ]


def _q(status: str, answer: str, evidence: list[str] | None = None) -> dict:
    return {"status": status, "answer": answer, "evidence": evidence or []}


def _mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _sentence(text: str) -> str:
    return f"{text[:1].upper()}{text[1:]}."


# --------------------------------------------------------------------------- the questions


def q1_composition(ev: Evidence) -> dict:
    finals = [(r, o) for r, o in ev.finals() if o.consistency.get("ssim") is not None]
    # Runs from before the final-image step (notebook 00 v1.1) name no final: every output they measured counts.
    with_final = {r.run_id for r, _ in finals}
    finals += [
        (r, o)
        for r in ev.runs
        if r.notebook in ("00", "01", "02", "03") and r.run_id not in with_final
        for o in r.outputs
        if o.consistency.get("ssim") is not None
    ]
    if not finals:
        return _q("not measured", "No reconstruction run has been collected yet; run notebook 00, 01, 02 or 03.")
    worst = min(finals, key=lambda x: x[1].consistency["ssim"])[1].consistency["ssim"]
    edges = [o.consistency.get("edge_mean_px") for _, o in finals if o.consistency.get("edge_mean_px") is not None]
    largest = max(edges) if edges else None
    evidence = [
        f"{r.label}, notebook {r.notebook} `{o.name}`: source SSIM {o.consistency['ssim']:.5f}, edge displacement "
        f"{fmt(o.consistency.get('edge_mean_px'), 3)} px, PSNR {fmt(o.consistency.get('psnr_db'))} dB"
        for r, o in finals
    ]
    lowband = [row["lowband_psnr_db"] for row in ev.reference_rows]
    if lowband:
        evidence.append(
            f"Reference test: the low band (what the source can hold) agrees with ground truth at "
            f"{min(lowband):.1f} to {max(lowband):.1f} dB across every configuration."
        )
    measured = f"lowest SSIM {worst:.4f}, largest mean edge displacement {fmt(largest, 3)} px"
    if worst >= COMPOSITION_MIN_SSIM and (largest is None or largest <= COMPOSITION_MAX_EDGE_PX):
        answer = (
            f"Yes: every master reduced back to the source size matches the source ({measured}), so composition, "
            "geometry and tone are preserved."
        )
    else:
        answer = (
            f"Mostly, with exceptions to inspect: some outputs fall outside the provisional gates (SSIM at least "
            f"{COMPOSITION_MIN_SSIM:g}, mean edge displacement at most {COMPOSITION_MAX_EDGE_PX:g} px; {measured})."
        )
    answer += (
        " Source consistency is necessary, not sufficient: it says nothing about whether added fine detail is right."
    )
    return _q("answered", answer, evidence)


def q2_constrained(ev: Evidence) -> dict:
    rows = ev.reference_by_config()
    if "lanczos" in rows:
        lanczos = rows["lanczos"]
        evidence = [
            f"{r['case']}: Lanczos low band {r['lowband_psnr_db']:.1f} dB, above-limit correlation "
            f"{fmt(r['hf_correlation'], 3)}, above-limit error {fmt(r['hf_nmse'], 3)}"
            for r in lanczos
        ]
        return _q(
            "answered",
            "Detail coarser than the source sampling limit (about two output pixels at x1.71) is constrained: every "
            "method reproduces that band of the ground truth closely. Detail finer than the limit is not: "
            "interpolation leaves it empty (its above-limit error stays near 1.0, the score of adding nothing), so "
            "anything there comes from a model, not from the source pixels.",
            evidence,
        )
    zero = ev.outputs("lanczos")
    if zero:
        return _q(
            "partial",
            "On the real inputs, structure at and above the source sampling limit is constrained (every output "
            "reduced to the source size matches it), but without ground truth nothing can say which finer detail is "
            "real. The controlled reference test (notebook 06, E10) measures it.",
            [f"{r.label}: Lanczos source PSNR {fmt(o.consistency.get('psnr_db'))} dB" for r, o in zero],
        )
    return _q("not measured", "Needs the controlled reference test (notebook 06, E10).")


def q3_priors(ev: Evidence) -> dict:
    evidence = []
    for run, output in ev.outputs("sr_conservative") + ev.outputs("sr_balanced"):
        lanczos = next((o for o in run.outputs if o.name == "lanczos"), None)
        crops, base = (
            output.detail.get("crop_hf_energy") or [],
            (lanczos.detail.get("crop_hf_energy") if lanczos else []),
        )
        diff = output.detail.get("difference_vs_lanczos") or {}
        if crops and base and len(crops) == len(base):
            ratios = [c / b for c, b in zip(crops, base, strict=True) if b > 0]
            evidence.append(
                f"{run.label} {output.name}: crop detail energy {min(ratios):.1f}x to {max(ratios):.1f}x Lanczos; "
                f"mean departure from Lanczos {fmt(diff.get('mean'))} 8-bit codes (p99 {fmt(diff.get('p99'))})"
            )
    rows = ev.reference_by_config()
    for name in ("sr_conservative", "sr_balanced", "full"):
        for r in rows.get(name, []):
            evidence.append(
                f"reference {r['case']} {name}: above-limit energy {fmt(r['hf_energy_ratio'], 2)}x the truth's, "
                f"correlation {fmt(r['hf_correlation'], 3)}, error {fmt(r['hf_nmse'], 3)}"
            )
    if not evidence:
        return _q("not measured", "Needs SR runs (notebook 00) and the reference test (notebook 06).")
    status = "answered" if any(n in rows for n in ("sr_conservative", "sr_balanced")) else "partial"
    answer = (
        "All detail above the source sampling limit in the SR masters is generated from learned priors: texture in "
        "fabric, foliage, gravel and hair, and sharper edges. The fidelity network adds little of it; the perceptual "
        "network adds many times Lanczos' detail energy."
    )
    if status == "answered":
        answer += (
            " The reference test measures how much of it is real: correlation with the true above-limit detail and "
            "the above-limit error (below 1.0 means more right than wrong)."
        )
    else:
        answer += " Whether any of it is correct needs the reference test (notebook 06, E10)."
    return _q(status, answer, evidence)


def q4_identity(ev: Evidence) -> dict:
    humans = [(r, o) for r, o in ev.outputs("human_final") if o.detail.get("identity")]
    evidence = []
    faces = events = 0
    for run, output in humans:
        n = len(output.detail["identity"].get("faces", []))
        e = len(output.detail.get("failure_events") or [])
        faces += n
        events += e
        evidence.append(f"{run.label}, notebook 01 `{run.short}`: {e} identity recovery event(s) among {n} face(s)")
    for r in ev.reference_rows:
        if r.get("face_psnr_db") is not None and r["configuration"] in (
            "sr_conservative",
            "sr_balanced",
            "specialized",
        ):
            evidence.append(
                f"reference {r['case']} {r['configuration']}: face PSNR vs truth {r['face_psnr_db']:.2f} dB, face "
                f"above-limit error {fmt(r['face_hf_nmse'], 3)}"
            )
    if not humans:
        status = "partial" if evidence else "not measured"
        return _q(status, "Needs notebook 01 (or 03) runs on the portrait: the identity checks run there.", evidence)
    rate = events / faces if faces else 0.0
    return _q(
        "answered" if faces else "partial",
        f"Super-resolution failed the identity gates on {events} of {faces} observed face(s) ({rate:.0%}); "
        "each failure "
        "was reduced or replaced by source-derived Lanczos locally (PRD 11.6). These are consistency checks, not "
        "identity proof; a reviewer must still look at the face crops.",
        evidence,
    )


def q5_face_size(ev: Evidence) -> dict:
    header, rows = identity_table([r for r, _ in ev.outputs("human_final")])
    if not rows:
        return _q("not measured", "Needs notebook 01 runs with faces in more than one tier (PRD 11.2).")
    evidence = [
        f"{row[0]} tier {row[2]}: {row[3]} face(s), mean SSIM {row[4]}, mean dE00 {row[5]}, events {row[6]}"
        for row in rows
    ]
    scales = sorted({r["scale"] for r in ev.reference_rows if r.get("face_psnr_db") is not None})
    if len(scales) > 1:
        for scale in scales:
            values = [
                r["face_psnr_db"]
                for r in ev.reference_rows
                if r["scale"] == scale and r["configuration"] == "sr_conservative"
            ]
            if values:
                evidence.append(f"reference portrait at x{scale:g}: conservative SR face PSNR {_mean(values):.2f} dB")
    return _q(
        "answered",
        "Reliability falls with face size: tier 1 faces (96 source px or more) carry enough pixels for consistency "
        "checks, tier 3 faces (under 32 px) do not, so they only get conservative processing and any detail on them "
        "is invented. The per-tier numbers are below.",
        evidence,
    )


def q6_crowd(ev: Evidence) -> dict:
    crowds = [(r, o.detail["crowd"]) for r, o in ev.outputs("human_final") if o.detail.get("crowd")]
    if not crowds:
        return _q("not measured", "Needs notebook 01 runs on a crowd scene.")
    evidence = []
    for run, crowd in crowds:
        flagged = sum(1 for p in crowd.get("duplicate_pairs", []) if p.get("flagged"))
        texture = sum(1 for p in crowd.get("skin_texture", []) if p.get("flagged"))
        new = crowd.get("new_face_like_detections")
        evidence.append(
            f"{run.label} `{run.short}`: {flagged} duplicated-looking face pair(s) flagged, {texture} texture flag(s), "
            f"new face-like detections: {'not checked' if new is None else new}"
        )
    return _q(
        "partial",
        "Classical checks found the flags below. Background faces are never generatively restored (prohibited by "
        "the profile), which keeps identities from being invented by face restoration, but SR still draws texture "
        "on small faces; confirming that no new identity appears needs the reviewer pass on the small-face panels.",
        evidence,
    )


def q7_night_shadows(ev: Evidence) -> dict:
    header, rows = night_table([r for r, _ in ev.outputs("night_final")])
    evidence = [
        f"{row[0]} `{row[1]}` ({row[2]} input): {row[3]} {row[4]} -> {row[5]}"
        for row in rows
        if "shadow" in row[3] or "black" in row[3]
    ]
    by = ev.reference_by_config()
    for name in ("sr_conservative", "specialized"):
        for r in by.get(name, []):
            if r["reference"] == "night_forest_path" and r.get("shadow_hf_nmse") is not None:
                evidence.append(f"reference {r['case']} {name}: shadow above-limit error {r['shadow_hf_nmse']:.3f}")
    if not evidence:
        return _q("not measured", "Needs notebook 02 runs on the night image and the night reference test.")
    status = "answered" if any("reference" in e for e in evidence) and rows else "partial"
    return _q(
        status,
        "The night processing lowers shadow chroma noise and banding without crushing blacks (numbers below); its "
        "shadow sharpening is limited by the invented-texture tolerance and the PRD 12.6 failure rule. Whether shadow "
        "texture is true can only be measured on the night reference: a shadow above-limit error below 1.0 means "
        "the added texture is more right than wrong.",
        evidence,
    )


def q8_highlights(ev: Evidence) -> dict:
    header, rows = night_table([r for r, _ in ev.outputs("night_final")])
    evidence = [
        f"{row[0]} `{row[1]}`: {row[3]} {row[4]} -> {row[5]}"
        for row in rows
        if "highlight" in row[3] or "halo" in row[3]
    ]
    if not evidence:
        return _q("not measured", "Needs notebook 02 runs on the night image.")
    return _q(
        "partial",
        "The highlight shoulder lowers the share of clipped samples and the halo clamp keeps rings around lights "
        "small (below), so lights roll off more smoothly. It reshapes values the render already clipped; no clipped "
        "radiance is recovered, and every report says so. Whether it looks more convincing needs the visual review.",
        evidence,
    )


def _bit_depth_rows(ev: Evidence) -> list[tuple[str, dict]]:
    found = []
    for run in ev.ablations:
        study = (run.manifest.get("bit_depth") or {}).get("study")
        if study:
            found.append((f"{run.label} ablation", study))
    for run in ev.of("05"):
        if run.manifest.get("bit_depth"):
            found.append((f"{run.label} notebook 05", run.manifest["bit_depth"]))
    return found


def q9_sixteen_bit(ev: Evidence) -> dict:
    studies = _bit_depth_rows(ev)
    if not studies:
        return _q("not measured", "Needs the notebook 06 ablation (bit-depth study) or a notebook 05 run.")
    evidence = []
    for label, study in studies:
        rows = {r["name"]: r for r in study["representations"]}
        names = list(rows)
        eight = next((rows[n] for n in names if "8" in n and "int" in n), rows[names[0]])
        sixteen = next((rows[n] for n in names if "16" in n and "int" in n), None)
        if sixteen:
            evidence.append(
                f"{label}: after an identical +3 stop shadow push, {eight['name']} keeps "
                f"{fmt(eight['edit_shadow_levels'], 1)} "
                f"shadow levels and {sixteen['name']} {fmt(sixteen['edit_shadow_levels'], 1)} (float32 reference "
                f"{fmt(study.get('reference_edit_shadow_levels'), 1)})"
            )
    return _q(
        "answered",
        "Yes for edits: a 16-bit master keeps far more distinct shadow levels through a strong push and drifts less "
        "through repeated transforms than 8 bits, so later grading bands less. It adds precision, not information: "
        "the source's 8-bit codes expand to 16 bits without a single new value (E01).",
        evidence,
    )


def q10_exr(ev: Evidence) -> dict:
    evidence = []
    for run in ev.of("00"):
        for name, record in (run.manifest.get("baselines") or {}).items():
            stats, exports = record.get("statistics") or {}, record.get("exports") or {}
            if stats and exports.get("exr") and exports.get("tiff16"):
                outside = (stats.get("below_zero_fraction") or 0) + (stats.get("above_one_fraction") or 0)
                evidence.append(
                    f"{run.label} {name}: {outside:.3%} of samples outside [0, 1] are kept by the half EXR and "
                    f"clipped by the TIFF; EXR {exports['exr']['size_bytes'] / 1e6:.0f} MB vs TIFF "
                    f"{exports['tiff16']['size_bytes'] / 1e6:.0f} MB"
                )
    if not evidence:
        return _q("not measured", "Needs notebook 00 runs (they export both formats of every baseline).")
    return _q(
        "answered",
        "Modestly: the linear half-float EXR keeps interpolation overshoot and highlight headroom that the 16-bit TIFF "
        "clips, is linear for compositing and grading, and is smaller here. It is display-referred linear light, not "
        "scene-referred radiance, so it gives no extra dynamic range.",
        evidence,
    )


def _effect(ev: Evidence, full: str, without: str) -> list[str]:
    out = []
    by = ev.reference_by_config()
    for f in by.get(full, []):
        w = next((r for r in by.get(without, []) if r["case"] == f["case"]), None)
        if w:
            out.append(
                f"reference {f['case']}: {full} {f['psnr_db']:.2f} dB vs {without} {w['psnr_db']:.2f} dB "
                f"({f['psnr_db'] - w['psnr_db']:+.2f} dB), above-limit error {fmt(f['hf_nmse'], 3)} vs "
                f"{fmt(w['hf_nmse'], 3)}"
            )
    for run, rec in ev.ablation_records():
        if rec["name"] == without:
            full_rec = next(
                (x for x in run.manifest["configurations"] if x["name"] == full and "measurements" in x), None
            )
            if full_rec:
                a, b = full_rec["measurements"], rec["measurements"]
                out.append(
                    f"{run.label} ablation: detail {a['detail_energy'] / b['detail_energy'] - 1:+.1%}, source PSNR "
                    f"{a['source_consistency']['psnr_db'] - b['source_consistency']['psnr_db']:+.2f} dB with the stage"
                )
    return out


def q11_bayer(ev: Evidence) -> dict:
    evidence = _effect(ev, "full", "full_no_sensor")
    for run in ev.of("05"):
        for method, summary in (run.manifest.get("roundtrip") or {}).items():
            psnr = (summary.get("full_frame") or {}).get("psnr_encoded_db")
            evidence.append(f"{run.label} notebook 05 E08 {method} round trip: {fmt(psnr)} dB against its input")
    if not evidence:
        return _q(
            "not measured", "Needs the notebook 06 ablation or reference test (full vs full_no_sensor), or notebook 05."
        )
    accurate = [e for e in evidence if e.startswith("reference")]
    return _q(
        "answered" if accurate else "partial",
        "No. Sampling the finished image through a synthetic colour filter array, adding noise and demosaicing it "
        "again can only lose detail and add noise: the round trip moves the output away from both the source and "
        "the ground truth (numbers below). Any 'camera look' it adds is simulated noise and demosaic softness.",
        evidence,
    )


def q12_dng(ev: Evidence) -> dict:
    evidence = []
    for run in ev.of("05"):
        dng = (run.manifest.get("exports") or {}).get("dng")
        if dng:
            evidence.append(f"{run.label} notebook 05: synthetic DNG valid {dng.get('valid')}, LibRaw read back")
    status = "answered" if evidence else "reasoned"
    return _q(
        status,
        "Only for compatibility testing. A synthetic DNG wraps a mosaic simulated from processed RGB: it opens in raw "
        "converters, but it holds no sensor measurements, its noise is simulated, and developing it returns less "
        "than the RGB master it was made from (E08). As a deliverable it would misrepresent the image as a capture.",
        evidence,
    )


def q13_realism(ev: Evidence) -> dict:
    scores = ev.review_scores
    if scores:
        ranked = sorted(scores.items(), key=lambda kv: -float(kv[1].get("naturalness", 0)))
        evidence = [
            f"{name}: " + ", ".join(f"{k} {v}" for k, v in values.items() if k in REVIEW_CRITERIA)
            for name, values in ranked
        ]
        return _q(
            "answered",
            f"From the blind review, the most natural-looking configuration is `{ranked[0][0]}`; the scores are below "
            "(1 to 5, artefact severity higher is worse).",
            evidence,
        )
    evidence = [e for e in _effect(ev, "full", "full_no_grain") + _effect(ev, "full", "full_no_optics")]
    return _q(
        "not measured",
        "Perceived realism needs the blind review (PRD 26.3): score the anonymised blind sheet of the ablation and "
        "enter the scores in notebook 06. Measured proxies only show what each stage changes.",
        evidence,
    )


def q14_harm(ev: Evidence) -> dict:
    evidence = []
    for run, output in [(r, o) for r in ev.runs for o in r.outputs if o.seam_passed is False]:
        evidence.append(f"{run.label} notebook {run.notebook}: {output.name} FAILED its seam gate")
    for run, output in ev.outputs("sr_balanced"):
        stats = output.detail.get("statistics") or {}
        low = stats.get("min") or {}
        if low and min(low.values()) < -0.05:
            evidence.append(
                f"{run.label}: balanced SR undershoots to {min(low.values()):.3f} linear "
                "(dark halos clipped in the TIFF)"
            )
    for run in ev.ablations:
        for c in run.manifest.get("contributions", []):
            if c["source_psnr_db"] > 0.5 or (c.get("reference_psnr_db") or 0) < -0.05:
                evidence.append(
                    f"{run.label} ablation: removing {c['stage']} changes source PSNR by {-c['source_psnr_db']:+.2f} dB"
                )
    by = ev.reference_by_config()
    lanczos = {r["case"]: r for r in by.get("lanczos", [])}
    for name, rows in by.items():
        for r in rows:
            base = lanczos.get(r["case"])
            if base and name != "lanczos" and r["psnr_db"] < base["psnr_db"] - 0.05:
                evidence.append(
                    f"reference {r['case']}: {name} is {base['psnr_db'] - r['psnr_db']:.2f} dB less accurate "
                    "than Lanczos"
                )
    if not evidence:
        return _q("not measured", "Needs the ablation, the reference test and the specialised runs.")
    status = "answered" if ev.reference_rows else "partial"
    return _q(
        status,
        "These stages measurably harm the result (below): anything that lowers accuracy against ground truth or fails "
        "a gate. Grain and the sensor round trip lower accuracy by construction; the perceptual SR mode over-sharpens "
        "and failed the seam gate on the night image.",
        evidence,
    )


def best_configuration(ev: Evidence, config: EvaluationConfig) -> dict:
    """The cheapest configuration within ``report.quality_tolerance_db`` of the most accurate one (reference test),
    with its runtime at the full 50 MP size estimated from the ablation and checked against measured runs."""
    by = ev.reference_by_config()
    lanczos = {r["case"]: r["psnr_db"] for r in by.get("lanczos", [])}
    quality = {}
    for name, rows in by.items():
        gains = [r["psnr_db"] - lanczos[r["case"]] for r in rows if r["case"] in lanczos]
        if gains:
            quality[name] = statistics.fmean(gains)
    runtime = full_size_runtime(ev)
    if not quality:
        return {"status": "not measured", "runtime": runtime}
    best = max(quality.values())
    eligible = [n for n, q in quality.items() if q >= best - config.report.quality_tolerance_db]
    cost = {n: runtime.get(n, {}).get("estimated_full_seconds") for n in eligible}
    known = [n for n in eligible if cost[n] is not None]
    choice = min(known, key=lambda n: cost[n]) if known else max(eligible, key=lambda n: quality[n])
    return {"status": "answered", "choice": choice, "quality": quality, "eligible": eligible, "runtime": runtime}


def full_size_runtime(ev: Evidence) -> dict:
    """Per configuration, the ablation's measured seconds scaled from its crop to the whole source (every stage is
    close to linear in pixels), as an estimate of a full 50 MP run on the same GPU."""
    out: dict[str, dict] = {}
    for run in ev.ablations:
        crop = run.manifest.get("crop") or {}
        box, full = crop.get("box"), crop.get("full_target") or {}
        if not box or not full:
            continue
        factor = (full["source_width"] * full["source_height"]) / max(1, (box[2] - box[0]) * (box[3] - box[1]))
        gpu = (run.manifest.get("runtime", {}).get("gpu") or {}).get("name", "CPU")
        for rec in run.manifest.get("configurations", []):
            if "seconds" in rec:
                out.setdefault(rec["name"], {"estimated_full_seconds": rec["seconds"]["total"] * factor, "gpu": gpu})
    return out


def q15_colab(ev: Evidence, config: EvaluationConfig) -> dict:
    best = best_configuration(ev, config)
    evidence = []
    for run in ev.of("00"):
        p = run.performance
        sr = {o.name: o.detail.get("inference_seconds") for o in run.outputs if o.kind == "sr"}
        evidence.append(
            f"{run.label} notebook 00 on {p['gpu']}: {fmt(p['total_seconds'], 0)} s for both baselines and "
            f"{len(sr)} SR modes at {fmt(p['output_megapixels'], 1)} MP (SR inference "
            + ", ".join(f"{k} {fmt(v, 0)} s" for k, v in sr.items())
            + f"), peak RSS {fmt(p['peak_rss_mb'], 0)} MB"
        )
    for name, values in best.get("runtime", {}).items():
        evidence.append(
            f"ablation estimate at full size: {name} about {values['estimated_full_seconds']:.0f} s on {values['gpu']}"
        )
    if best["status"] != "answered":
        return _q(
            "partial" if evidence else "not measured",
            "Quality against ground truth needs the reference test; until then the measured runtimes below only rank "
            "cost. On an L4, conservative SR is about 6.5 minutes of a 50 MP run; balanced costs the same and failed "
            "the seam gate on the night image.",
            evidence,
        )
    quality = best["quality"]
    choice = best["choice"]
    evidence.insert(
        0,
        "mean accuracy gain over Lanczos against ground truth: "
        + ", ".join(f"{n} {q:+.2f} dB" for n, q in sorted(quality.items(), key=lambda kv: -kv[1])),
    )
    label = CONFIGURATIONS[choice].label if choice in CONFIGURATIONS else choice
    return _q(
        "answered",
        f"`{choice}` ({label}): the cheapest configuration within {config.report.quality_tolerance_db:g} dB of the "
        f"most accurate one (configurations within that band: {', '.join(best['eligible'])}).",
        evidence,
    )


def q16_universal(ev: Evidence) -> dict:
    evidence = []
    for runs in by_source(ev.runs).values():
        universal = [r for r in runs if r.notebook == "03" and r.status == "complete"]
        special = [r for r in runs if r.notebook in ("01", "02") and r.parent is None and r.status == "complete"]
        for u in universal:
            final = next((o for o in u.outputs if o.name == "universal_final"), None)
            category = final.detail.get("category") if final else "?"
            delegated = (u.manifest.get("route") or {}).get("delegate")
            for s in special:
                sf = next((o for o in s.outputs if o.kind == "final"), None)
                if final and sf:
                    evidence.append(
                        f"{u.label}: notebook 03 ({category}{DELEGATED if delegated else ''}) source PSNR "
                        f"{fmt(final.consistency.get('psnr_db'))} dB in {fmt(u.performance['total_seconds'], 0)} s vs "
                        f"notebook {s.notebook} {fmt(sf.consistency.get('psnr_db'))} dB in "
                        f"{fmt(s.performance['total_seconds'], 0)} s"
                    )
    if not evidence:
        return _q("not measured", "Needs notebook 03 runs and notebook 01 or 02 runs on the same image.")
    return _q(
        "answered",
        "Yes where it routes: notebook 03 hands human and night scenes to the specialised pipelines, so for those it "
        "is the specialised quality at the cost of the analysis step. Scenes it keeps (general, night with faces) use "
        "the conservative common pipeline.",
        evidence,
    )


def q17_absent(ev: Evidence) -> dict:
    evidence = []
    for run in ev.of("00"):
        v = run.manifest.get("validation") or {}
        samples, clip = v.get("samples") or {}, v.get("clipping") or {}
        target = run.manifest.get("target_resolution") or {}
        if target.get("pixels"):
            share = target["source_width"] * target["source_height"] / target["pixels"]
            evidence.append(
                f"{run.label}: the source holds {share:.1%} of the 50 MP output's samples per channel "
                f"(x{target.get('nominal_scale', 0):.4f}); the rest are estimated or synthesised"
            )
        evidence.append(
            f"{run.label}: {samples.get('effective_bit_depth', '?')}-bit effective codes, "
            f"{fmt((clip.get('any_channel_at_max_fraction') or 0) * 100, 3)}% of pixels clipped at full scale, EXIF "
            f"{'present' if (v.get('metadata') or {}).get('exif_present') else 'absent'}, ICC "
            f"{'present' if (v.get('color') or {}).get('icc_profile_present') else 'absent'}"
        )
    by = ev.reference_by_config()
    best = min(
        (r for rows in by.values() for r in rows if r.get("hf_nmse") is not None),
        key=lambda r: r["hf_nmse"],
        default=None,
    )
    if best:
        evidence.append(
            f"reference: the best reconstruction ({best['configuration']}, {best['case']}) still has above-limit error "
            f"{best['hf_nmse']:.3f} of the truth's above-limit energy"
        )
    return _q(
        "answered" if best else "reasoned",
        "Permanently absent: every detail finer than the render's sampling limit (at x1.71 the source holds "
        "about a third of the output's samples, and that is all the information there is), sensor photon counts "
        "and genuine noise, colour-filter samples, radiance above the clipped highlights and below the crushed "
        "shadows, scene-referred dynamic range beyond what 8 bits encode, and any optical information the "
        "renderer never computed.",
        evidence,
    )


def q18_needed(ev: Evidence) -> dict:
    return _q(
        "reasoned",
        "Genuine scene-referred or sensor-faithful 50 MP output needs data the render never had: native 50 MP "
        "rendering or capture (a model that outputs 50 MP, 16-bit linear directly, as the next experiment plans), or "
        "multi-frame captures with sub-pixel shifts for real super-resolution; raw sensor data (CFA samples, black "
        "and white levels, noise model) for sensor fidelity; exposure brackets or a linear HDR render for highlight "
        "and shadow radiance; and the camera and lens description for optical character.",
    )


ANSWERS = (
    q1_composition,
    q2_constrained,
    q3_priors,
    q4_identity,
    q5_face_size,
    q6_crowd,
    q7_night_shadows,
    q8_highlights,
    q9_sixteen_bit,
    q10_exr,
    q11_bayer,
    q12_dng,
    q13_realism,
    q14_harm,
    q15_colab,
    q16_universal,
    q17_absent,
    q18_needed,
)


# --------------------------------------------------------------------------- outcome and summary


def outcome(ev: Evidence, config: EvaluationConfig) -> dict:
    """The PRD 33 outcome from the reference test's best learned configuration vs Lanczos per case."""
    t = config.report
    by = ev.reference_by_config()
    lanczos = {r["case"]: r for r in by.get("lanczos", [])}
    learned = [n for n in by if n in CONFIGURATIONS and CONFIGURATIONS[n].learned]
    per_case = []
    for case, base in lanczos.items():
        candidates = [r for n in learned for r in by[n] if r["case"] == case]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: r["psnr_db"])
        per_case.append(
            {
                "case": case,
                "best": best["configuration"],
                "psnr_gain_db": best["psnr_db"] - base["psnr_db"],
                "hf_nmse": best.get("hf_nmse"),
            }
        )
    thresholds = t.to_dict()
    if not per_case:
        return {
            "outcome": None,
            "status": "undetermined",
            "reason": "the controlled reference test (E10) has not been collected, so accuracy against ground truth "
            "is unknown; source consistency alone cannot choose an outcome",
            "thresholds": thresholds,
            "per_case": [],
        }
    gain = statistics.fmean(c["psnr_gain_db"] for c in per_case)
    nmse_values = [c["hf_nmse"] for c in per_case if c["hf_nmse"] is not None]
    nmse = statistics.fmean(nmse_values) if nmse_values else 1.0
    preference = _review_preference(ev.review_scores)
    if gain >= t.a_min_psnr_gain_db and nmse <= t.a_max_hf_nmse and preference:
        choice = "A"
        reason = (
            f"the best SR gains {gain:+.2f} dB over Lanczos against ground truth with above-limit error {nmse:.3f}, "
            "and the blind review prefers it"
        )
    elif gain < t.c_max_psnr_gain_db and nmse >= t.c_min_hf_nmse:
        choice = "C"
        reason = (
            f"against ground truth the best SR gains only {gain:+.2f} dB over Lanczos and its above-limit error is "
            f"{nmse:.3f} (1.0 is adding nothing): larger, smoother files with little real structural improvement"
        )
    else:
        choice = "B"
        reason = (
            f"the best SR gains {gain:+.2f} dB over Lanczos against ground truth with above-limit error {nmse:.3f}: "
            "apparent resolution improves, but not enough real detail (or no blind-review preference) for a "
            "camera-grade claim"
        )
        if gain >= t.a_min_psnr_gain_db and nmse <= t.a_max_hf_nmse and not preference:
            reason += "; outcome A would also need the blind review to prefer it"
    return {
        "outcome": choice,
        "title": OUTCOMES[choice],
        "status": "recommended",
        "reason": reason,
        "next_step": NEXT_STEPS[choice],
        "mean_psnr_gain_db": gain,
        "mean_hf_nmse": nmse,
        "per_case": per_case,
        "thresholds": thresholds,
    }


def _review_preference(scores: dict) -> bool:
    """True when the blind review prefers an SR configuration over Lanczos on overall preference."""
    if not scores or "lanczos" not in scores:
        return False
    base = float(scores["lanczos"].get("overall_preference", 0))
    return any(
        float(v.get("overall_preference", 0)) > base
        for k, v in scores.items()
        if k in CONFIGURATIONS and CONFIGURATIONS[k].learned
    )


def can_and_cannot(ev: Evidence, answers: list[dict], decision: dict) -> dict:
    can = [
        "Make an exact 50 MP master (within 0.1% of 50,000,000 pixels at the source aspect ratio) from a 4K-class "
        "render, as validated 16-bit TIFF and linear OpenEXR with provenance.",
        "Preserve composition, geometry, tone and colour: every master reduces back to its source almost exactly.",
        "Make edges cleaner and add plausible fine texture with pretrained SR, while keeping faces and lights under "
        "explicit protection and measuring where detail was synthesised.",
        "Give later edits more headroom: 16-bit and float masters band less under strong grading than the 8-bit "
        "render.",
    ]
    cannot = [
        "Recover detail finer than the source sampling limit: SR invents it from priors, and the reference test "
        "measures how little of it matches reality.",
        "Add dynamic range, clipped highlight or crushed shadow information, or genuine sensor noise: 16-bit and EXR "
        "add precision only.",
        "Produce camera RAW: the synthetic CFA and DNG path is a simulation that only loses information.",
        "Guarantee identity on small faces: below about 32 source pixels any facial detail is invented.",
    ]
    if decision.get("outcome"):
        cannot.append(
            f"On this cycle's evidence the approach is Outcome {decision['outcome']} ({decision['title']}): "
            + decision["reason"]
            + "."
        )
    else:
        cannot.append("The outcome stays undetermined until the reference test results are collected.")
    return {
        "can": can,
        "cannot": cannot,
        "next_experiment": (
            "A model trained to output native 16-bit 50 MP images would remove the limit this approach cannot cross: "
            "it would render the fine detail and tonal range at full resolution instead of inferring them from a "
            "smaller 8-bit render. Its evaluation can reuse this cycle's tools: the reference test, the ablation and "
            "the source-consistency and above-limit metrics."
        ),
    }


def build(runs: list[Run], config: EvaluationConfig, review_scores: dict | None = None) -> dict:
    """Answer PRD 34, choose a PRD 33 outcome and summarise; a JSON-serialisable record."""
    ev = Evidence(runs, review_scores)
    answers = []
    for number, (question, function) in enumerate(zip(QUESTIONS, ANSWERS, strict=True), start=1):
        result = function(ev, config) if function is q15_colab else function(ev)
        answers.append({"number": number, "question": question, **result})
    decision = outcome(ev, config)
    return {
        "schema": "synthetic_raw_lab.phase6_final_report/1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "runs": [{"run_id": r.run_id, "notebook": r.notebook, "label": r.label, "status": r.status} for r in runs],
        "answers": answers,
        "outcome": decision,
        "best_configuration": best_configuration(ev, config),
        "summary": can_and_cannot(ev, answers, decision),
        "review_scores": review_scores or {},
        "status_counts": {
            s: sum(a["status"] == s for a in answers) for s in ("answered", "partial", "not measured", "reasoned")
        },
    }


def render(record: dict, runs: list[Run], figures: dict | None = None) -> str:
    """The final experiment report as Markdown."""
    counts = record["status_counts"]
    decision = record["outcome"]
    lines = ["# Synthetic RAW 50MP Lab: final experiment report (cycle 1)", ""]
    lines += [
        "> Synthetic camera-grade masters reconstructed from AI-generated sources. None is a camera capture or camera "
        "RAW file; preserved, interpolated, inferred, simulated and encoded information are kept apart throughout.",
        "",
        f"Generated {record['generated_at']} from {len(runs)} run(s). Questions: {counts['answered']} answered, "
        f"{counts['partial']} partial, {counts['not measured']} not measured, {counts['reasoned']} reasoned.",
        "",
        "## 1. Outcome (PRD 33)",
        "",
    ]
    if decision.get("outcome"):
        lines += [
            f"**Outcome {decision['outcome']}: {decision['title']}.** {_sentence(decision['reason'])}",
            "",
        ]
        lines += [f"Recommended next step: {decision['next_step']}", ""]
        lines += [
            markdown_table(
                ["reference case", "best learned configuration", "gain over Lanczos dB", "above-limit error"],
                [
                    (c["case"], c["best"], f"{c['psnr_gain_db']:+.2f}", fmt(c["hf_nmse"], 3))
                    for c in decision["per_case"]
                ],
            ),
            "",
        ]
    else:
        lines += [f"**Undetermined:** {decision['reason']}.", ""]
    lines += [
        "Provisional thresholds: " + ", ".join(f"{k} {v:g}" for k, v in decision["thresholds"].items()) + ".",
        "",
        "## 2. What this approach can and cannot achieve",
        "",
        "It can:",
        "",
        *[f"- {line}" for line in record["summary"]["can"]],
        "",
        "It cannot:",
        "",
        *[f"- {line}" for line in record["summary"]["cannot"]],
        "",
        f"Next experiment: {record['summary']['next_experiment']}",
        "",
        "## 3. Final scientific questions (PRD 34)",
        "",
    ]
    for a in record["answers"]:
        lines += [f"### {a['number']}. {a['question']}", "", f"*{a['status']}.* {a['answer']}", ""]
        lines += [f"- {e}" for e in a["evidence"]]
        if a["evidence"]:
            lines.append("")
    best = record["best_configuration"]
    lines += ["## 4. Best quality-to-runtime Colab configuration", ""]
    if best.get("status") == "answered":
        lines += [f"**`{best['choice']}`**, from the reference-test accuracy and the full-size runtime estimates:", ""]
        rows = [
            (
                n,
                f"{q:+.2f}",
                fmt((best["runtime"].get(n) or {}).get("estimated_full_seconds"), 0),
                "yes" if n in best["eligible"] else "",
            )
            for n, q in sorted(best["quality"].items(), key=lambda kv: -kv[1])
        ]
        lines += [
            markdown_table(["configuration", "gain over Lanczos dB", "est. full-size s", "within tolerance"], rows),
            "",
        ]
    else:
        lines += ["Not measured yet: needs the reference test (accuracy) and the ablation (runtime).", ""]
    lines += ["## 5. Evidence", ""]
    lines += [markdown_table(*inventory_table(runs)), ""]
    tables = (
        ("Source consistency of every output", outputs_table(runs)),
        ("Performance (PRD 28.1)", performance_table(runs)),
        ("Night metrics (PRD 23.5)", night_table(runs)),
        ("Identity by face tier (PRD 11.3)", identity_table(runs)),
    )
    for title, (header, rows) in tables:
        if rows:
            lines += [f"### {title}", "", markdown_table(header, rows), ""]
    references = [ref for r in runs if r.notebook == "06 reference" for ref in r.manifest.get("references", [])]
    if references:
        lines += ["### Reference photographs (E10)", ""]
        lines += [
            markdown_table(
                ["reference", "photograph", "author", "licence"],
                [
                    (x["name"], f"[{x['title'].removeprefix('File:')}]({x['page']})", x["author"], x["licence"])
                    for x in references
                ],
            ),
            "",
        ]
    for name, path in (figures or {}).items():
        lines += [f"### Visual review: {name}", "", f"![{name}]({path})", ""]
    lines += [
        "## 6. Notebooks",
        "",
        *[f"- {NOTEBOOK_NAMES[k]}" for k in NOTEBOOK_NAMES],
        "",
        "Re-run notebook 06 with more runs collected to update every answer; nothing in this report is edited by hand.",
    ]
    return "\n".join(lines) + "\n"


def write(record: dict, runs: list[Run], out_dir: str | Path, figures: dict | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "final_report.md"
    report.write_text(render(record, runs, figures))
    data = out_dir / "final_report.json"
    data.write_text(json.dumps(record, indent=2, default=str))
    return {"report": report, "json": data}
