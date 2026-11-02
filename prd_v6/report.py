"""report.html and summary.txt from the run folder's JSON (standard library only, like delivery)."""

from __future__ import annotations

import html
import json
from pathlib import Path

PLANES = ("R", "G", "B")
OUTPUTS = (("2k", "2K"), ("4k", "4K"), ("50mp", "50 MP"))


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _signed(value: float, places: int = 3) -> str:
    """Signed fixed-point text that never prints a negative zero."""
    return f"{round(float(value), places) + 0.0:+.{places}f}"


def _yes(flag) -> str:
    return "yes" if flag else "no"


def _pct(value) -> str:
    return "n/a" if value is None else f"{100 * float(value):.2f}%"


def _table(headers, rows) -> str:
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def _plane_rows(score: dict, keys=("A", "P2")) -> list[list[str]]:
    planes = (score or {}).get("planes", {})
    return [[p] + [_pct(planes.get(p, {}).get(k)) for k in keys] for p in PLANES]


def summary_text(run_dir: Path) -> str:
    verdict = _load(run_dir / "verdict.json") or {}
    lines = [f"run {run_dir.name}", f"status: {verdict.get('status', 'not decided yet')}", verdict.get("text", "")]
    for name, label in OUTPUTS:
        out = (verdict.get("outputs") or {}).get(name)
        if out:
            planes = "  ".join(
                f"{p}: A {_pct(out['planes'][p]['A'])} P2 {_pct(out['planes'][p]['P2'])}" for p in PLANES
            )
            exposure = f"  k {_signed(out['k_ev'])} EV" if isinstance(out.get("k_ev"), (int, float)) else ""
            lines.append(f"{label}: {'PASS' if out.get('passes') else 'MISSED'}  {planes}{exposure}")
    for reason in verdict.get("invalid_reasons", []):
        lines.append(f"invalid: {reason}")
    for warning in verdict.get("warnings", []):
        lines.append(f"warning: {warning}")
    for error in verdict.get("completed_with_errors", []):
        lines.append(f"error: {error}")
    state = _load(run_dir / "state.json") or {}
    for step in state.get("steps", []):
        lines.append(f"step {step['step']} {step['name']}: {step.get('status')}")
    return "\n".join(lines) + "\n"


def write(run_dir) -> Path:
    run_dir = Path(run_dir)
    (run_dir / "summary.txt").write_text(summary_text(run_dir))
    verdict = _load(run_dir / "verdict.json") or {}
    parts = [
        "<!doctype html><meta charset='utf-8'><title>PRD v6 run</title>",
        "<style>body{font:14px system-ui,sans-serif;margin:24px;max-width:1100px}"
        "table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #bbb;padding:3px 8px;text-align:right}"
        "th{background:#eee}h1{font-size:22px}img{max-width:100%}</style>",
        f"<h1>PRD v6 run {html.escape(run_dir.name)}: {html.escape(str(verdict.get('status', 'not decided')))}</h1>",
        f"<p><b>{html.escape(verdict.get('text', ''))}</b></p>",
        f"<p>{html.escape(verdict.get('in_sample', ''))}</p>",
    ]
    for key, title in (("invalid_reasons", "Invalid"), ("warnings", "Warnings"), ("completed_with_errors", "Errors")):
        if verdict.get(key):
            items = "".join(f"<li>{html.escape(str(item))}</li>" for item in verdict[key])
            parts.append(f"<h2>{title}</h2><ul>{items}</ul>")
    for name, label in OUTPUTS:
        metrics = _load(run_dir / "benchmark" / name / "metrics.json")
        if not metrics:
            continue
        gated = metrics["gated"]
        parts.append(f"<h2>{label} ({html.escape(metrics['layout'])}, {metrics['size'][1]}x{metrics['size'][0]})</h2>")
        parts.append(
            f"<p>k = {gated['k']:.4f} ({_signed(gated['k_ev'])} EV). Gate: {'PASS' if gated['passes'] else 'MISSED'}</p>"
        )
        rows = [
            [
                p,
                _pct(gated["planes"][p]["A"]),
                _pct(gated["planes"][p]["P2"]),
                _pct(gated["planes"][p]["A_unfitted"]),
                _pct(gated["planes"][p]["A_l1"]),
                _pct(gated["planes"][p]["diagnostic_plane_fitted"]["A"]),
            ]
            for p in PLANES
        ]
        parts.append(_table(["plane", "A (P1)", "P2", "A unfitted", "A L1", "A plane-fitted (diagnostic)"], rows))
        ladder = metrics.get("ladder", {})
        parts.append("<h3>Error budget ladder (diagnostic)</h3>")
        parts.append(
            _table(
                ["rung"] + [f"A {p}" for p in PLANES],
                [[rung] + [_pct(s["planes"][p]["A"]) for p in PLANES] for rung, s in ladder.items()],
            )
        )
        steps = metrics.get("ladder_steps", {})
        if steps:
            names = list(next(iter(steps.values())).keys())
            parts.append(
                _table(
                    ["plane"] + [f"{n} (points)" for n in names],
                    [[p] + [f"{100 * steps[p][n]:.2f}" for n in names] for p in PLANES],
                )
            )
        if metrics.get("ablation"):
            parts.append("<h3>Ablation (50 MP clean mosaic)</h3>")
            parts.append(
                _table(
                    ["stage"] + [f"delta A {p} (points)" for p in PLANES] + ["meets rule"],
                    [
                        [row] + [_signed(100 * v["delta_A"][p]) for p in PLANES] + [_yes(v["meets_rule"])]
                        for row, v in metrics["ablation"].items()
                    ],
                )
            )
        reported = metrics.get("reported", {})
        if reported:
            parts.append("<h3>Reported only</h3>")
            parts.append(
                _table(
                    ["variant"] + [f"A {p}" for p in PLANES],
                    [[k] + [_pct(v["planes"][p]["A"]) for p in PLANES] for k, v in reported.items()],
                )
            )
        sanity = metrics.get("sanity", {})
        if sanity:
            parts.append("<h3>Sanity checks</h3>")
            hard = sanity["hard"]
            sens = sanity["sensitivity"]
            rows = [
                ["self (hard)", _yes(hard["self"]["passed"]), _pct(hard["self"]["min_A"])],
                [
                    "noise floor (hard)",
                    _yes(hard["noise_floor"]["passed"]),
                    _pct(min(v["A"] for v in hard["noise_floor"]["planes"].values())),
                ],
            ]
            rows += [[f"{k} (sensitivity)", _yes(v["passed"]), _pct(v["min_A"])] for k, v in sens.items()]
            parts.append(_table(["check", "passed", "min A"], rows))
        integrity = metrics.get("integrity", {})
        parts.append(
            f"<p>Integrity: {'passed' if integrity.get('passed') else 'FAILED'} "
            f"{html.escape('; '.join(integrity.get('problems', [])))}</p>"
        )
        maps = [m for m in metrics.get("error_maps", []) if m.endswith("_small.png")]
        for image in maps:
            parts.append(
                f"<figure><img src='benchmark/{name}/{image}'><figcaption>{image}: signed relative error, "
                "clipped to +-20%, blue low, red high, grey masked</figcaption></figure>"
            )
    timings = verdict.get("timings_s")
    if timings:
        parts.append("<h2>Stage times</h2>" + _table(["stage", "seconds"], list(timings.items())))
    secondary = _load(run_dir / "secondary" / "secondary.json")
    if secondary:
        parts.append(
            "<h2>Secondary reports</h2><pre>" + html.escape(json.dumps(secondary, indent=2)[:20000]) + "</pre>"
        )
    target = run_dir / "report.html"
    target.write_text("\n".join(parts))
    return target
