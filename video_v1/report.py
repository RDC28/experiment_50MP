"""report.html and summary.txt from the run's JSON records (standard library only).

Written after every driver step and at the end, so a failed or interrupted run still explains itself.
"""

from __future__ import annotations

import datetime as _dt
import html
from pathlib import Path

from prd_v6.delivery import _read_json
from video_v1.delivery import NO_ACCURACY_CLAIM, load_state, run_id

STYLE = """
:root { --bg: #fff; --fg: #1b1f24; --muted: #5b6470; --line: #d8dde3;
  --ok: #1a7f37; --bad: #c62828; --warn: #9a6700; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #111418; --fg: #e6e9ed; --muted: #9aa4af; --line: #2c333b;
    --ok: #4ac26b; --bad: #ff6b6b; --warn: #d4a72c; }
}
body { background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif; margin: 0 auto; max-width: 1100px;
  padding: 24px 16px; }
h1 { font-size: 22px; margin: 0 0 8px; }
h2 { font-size: 17px; margin: 28px 0 8px; border-bottom: 1px solid var(--line); }
h3 { font-size: 15px; margin: 18px 0 6px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0; } td, th { border-bottom: 1px solid var(--line);
  padding: 4px 8px; text-align: left; vertical-align: top; } th { color: var(--muted); font-weight: 600; }
.status { font-weight: 700; } .COMPLETE, .passed { color: var(--ok); } .INVALID, .FAILED, .failed { color: var(--bad); }
.FLAGGED { color: var(--warn); } .muted { color: var(--muted); } img, video { max-width: 100%; height: auto; }
code { font-size: 13px; overflow-wrap: anywhere; } table.kv td:first-child { width: 30%; }
video { width: 100%; background: #000; } .thumbs { display: grid; gap: 12px; margin: 12px 0;
  grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); }
figure { margin: 0; min-width: 0; } figcaption { color: var(--muted); font-size: 13px; margin-top: 4px; }
"""


def _e(value) -> str:
    return html.escape("" if value is None else str(value))


def _table(rows, header=None) -> str:
    out = ["<table>" if header else "<table class='kv'>"]
    if header:
        out.append("<tr>" + "".join(f"<th>{_e(h)}</th>" for h in header) + "</tr>")
    for row in rows:
        out.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def _pct(value) -> str:
    return "-" if value is None else f"{100 * value:.3f}%"


def _seconds(value) -> str:
    return "-" if value is None else f"{value:.2f} s"


def _duration(value) -> str:
    """Seconds, minutes or hours, whichever reads best."""
    if value is None:
        return "-"
    if value < 90:
        return f"{value:.1f} s"
    if value < 90 * 60:
        return f"{value / 60:.1f} min"
    return f"{value / 3600:.1f} h"


def _parse(stamp):
    try:
        return _dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None


def wall_time_s(state: dict) -> float | None:
    starts = [_parse(r.get("started_at")) for r in state.get("steps", [])]
    ends = [_parse(r.get("finished_at")) for r in state.get("steps", [])]
    starts, ends = [s for s in starts if s], [e for e in ends if e]
    if not starts or not ends:
        return None
    return (max(ends) - min(starts)).total_seconds()


def gather(run_dir) -> dict:
    run_dir = Path(run_dir)
    return {
        "run_id": run_id(run_dir),
        "verdict": _read_json(run_dir / "verdict.json"),
        "state": load_state(run_dir),
        "source": _read_json(run_dir / "source.json"),
        "calibration": _read_json(run_dir / "calibration.json"),
        "convert": _read_json(run_dir / "convert.json"),
        "check": _read_json(run_dir / "check.json"),
        "environment": _read_json(run_dir / "environment.json"),
    }


def _source_rows(src: dict) -> list:
    signal = src.get("signal") or {}
    assumed = ", ".join(signal.get("assumed") or []) or "none"
    rotated = f", rotated {src['rotation_ccw']} deg" if src.get("rotation_ccw") else ""
    return [
        ("File", f"<code>{_e(Path(src['path']).name)}</code> ({_e(src.get('origin', {}).get('how'))})"),
        ("Container / codec", _e(f"{src.get('container')} / {src.get('codec')} {src.get('profile') or ''}")),
        ("Pixel format", _e(f"{src.get('pix_fmt')}, decoded as {src.get('layout', {}).get('pix_fmt')}")),
        ("Size", _e(f"{src['size'][1]}x{src['size'][0]}" + rotated)),
        ("Frame rate", _e(f"{src['frame_rate']} ({src.get('frame_rate_float', 0):.3f} fps, "
                          f"from {src.get('frame_rate_source')})")),
        ("Frames", _e(f"{src['frames']} ({src.get('duration_s') or 0:.3f} s)")),
        ("Colour", _e(
            f"matrix {signal.get('matrix')}, primaries {signal.get('primaries')}, transfer {signal.get('transfer')}, "
            f"{'full' if signal.get('full_range') else 'limited'} range, {signal.get('bits')}-bit, "
            f"chroma siting {signal.get('chroma_location')}; assumed (untagged): {assumed}")),
        ("Audio", _e(src.get("audio"))),
        ("Warnings", _e("; ".join(src.get("probe_warnings") or []) or "none")),
    ]  # fmt: skip


def _stability_section(check: dict, run_dir: Path) -> list[str]:
    out = ["<h2>Stability (flicker) check</h2>"]
    thresholds = check.get("thresholds", {})
    out.append(
        f"<p>Per frame: developed/source ratio of mean linear luma and of the R/G and B/G colour ratios. Passes when "
        f"no ratio changes by more than {_pct(thresholds.get('step_max'))} between consecutive frames and none spans "
        f"more than {_pct(thresholds.get('span_max'))} of its median over the clip. Frames with source luma below "
        f"{thresholds.get('dark_luma')} are not scored ({check.get('frames_dark', 0)} here).</p>"
    )
    status = "passed" if check.get("passed") else "FAILED"
    out.append(
        f"<p class='status {status}'>Stability {status}"
        + (f": {_e(check['reason'])}" if check.get("reason") else "")
        + "</p>"
    )
    rows = []
    for metric, record in (check.get("metrics") or {}).items():
        rows.append((
            _e(metric.replace("_", "/")), _e(f"{record['median']:.4f}" if record.get("median") is not None else "-"),
            _pct(record.get("span")), _pct(record.get("step_max")), _e(record.get("worst_step_frame")),
            f"<span class='{'passed' if record.get('passed') else 'failed'}'>"
            f"{'yes' if record.get('passed') else 'no'}</span>",
        ))  # fmt: skip
    out.append(_table(rows, ["ratio", "median", "span", "largest step", "at frame", "passed"]))
    if check.get("source_level_step_max") is not None:
        out.append(
            f"<p class='muted'>For context, the source's own mean luma changes by up to "
            f"{_pct(check['source_level_step_max'])} between frames "
            f"(median {_pct(check.get('source_level_step_median'))}).</p>"
        )
    if check.get("plot"):
        out.append(f"<img src='{_e(check['plot'])}' alt='stability plot'>")
    if check.get("preview"):
        out.append("<h2>Preview: source (left) and developed from the DNGs (right)</h2>")
        out.append(f"<video controls loop muted playsinline preload='metadata' src='{_e(check['preview'])}'></video>")
        out.append(f"<p class='muted'>{_e(check.get('develop'))}. Encoder: {_e(check.get('preview_encoder'))}.</p>")
    thumbs = check.get("thumbnails") or []
    if thumbs:
        out.append("<h3>First, middle and last frame (source left, developed right)</h3><div class='thumbs'>")
        for thumb in thumbs:
            caption = f"frame {int(Path(thumb).stem.split('_')[-1])}"
            out.append(
                f"<figure><img src='{_e(thumb)}' alt='{_e(caption)}'><figcaption>{_e(caption)}</figcaption></figure>"
            )
        out.append("</div>")
    return out


def _per_frame_and_clip(seconds_per_frame: float, hours_for_clip: float) -> str:
    return _e(f"{_duration(seconds_per_frame)}; whole clip {_duration(hours_for_clip * 3600)}")


def _estimate_rows(est: dict) -> list:
    tile = f"{est['hat_l_seconds_per_tile_a100']:.1f}, measured in v6 run {est['hat_l_measured_in']}"
    return [
        ("Target", _e(est["target"])),
        ("HAT-L tiles per frame", _e(est["sr_tiles_per_frame"])),
        ("HAT-L seconds per tile (A100, x8 ensemble)", _e(tile)),
        ("Other stages per frame at 4K", _seconds(est["other_stages_seconds_per_frame"])),
        ("Per frame, x8 ensemble (v6 MAX)",
         _per_frame_and_clip(est["seconds_per_frame_x8_ensemble"], est["hours_for_clip_x8_ensemble"])),
        ("Per frame, no ensemble",
         _per_frame_and_clip(est["seconds_per_frame_no_ensemble"], est["hours_for_clip_no_ensemble"])),
        ("DNG size per frame", _e(f"{est['dng_bytes_per_frame'] / 1e6:.1f} MB")),
        ("Note", _e(est["note"])),
    ]  # fmt: skip


def render_html(data: dict, run_dir: Path) -> str:
    verdict, src, cal, conv, check = (
        data["verdict"],
        data["source"],
        data["calibration"],
        data["convert"],
        data["check"],
    )
    status = (verdict or {}).get("status", "RUNNING")
    text = (verdict or {}).get("text") or f"In progress. {NO_ACCURACY_CLAIM}"
    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Video v1 run</title><style>{STYLE}</style></head><body>",
        f"<h1>Video to synthetic RAW, run {_e(data['run_id'])}: "
        f"<span class='status {_e(status)}'>{_e(status)}</span></h1>",
        f"<p>{_e(text)}</p>",
    ]
    if src:
        out += ["<h2>Source clip</h2>", _table(_source_rows(src))]
    if cal:
        out += [
            "<h2>Clip calibration (fixed once, used for every frame)</h2>",
            _table(
                [
                    ("Illuminant", _e(f"{cal['method']}: {cal['cct_k']:.0f} K")),
                    ("As-shot neutral", _e(", ".join(f"{v:.4f}" for v in cal["as_shot_neutral"]))),
                    ("Exposure", _e(cal["exposure"])),
                    ("Per-frame estimation", "none"),
                ]
            ),  # fmt: skip
        ]
    if conv:
        spf = conv["seconds_per_frame"]
        stages = ", ".join(f"{k} {v:.3f} s" for k, v in conv["stage_seconds_mean"].items())
        out += [
            "<h2>DNG sequence</h2>",
            _table(
                [
                    (
                        "Frames",
                        _e(f"{conv['frames_converted']} of {conv['frames_source']} (planned {conv['frames_planned']})"),
                    ),
                    (
                        "Size",
                        _e(
                            f"{conv['size'][1]}x{conv['size'][0]} RGGB, 16-bit, uncompressed DNG 1.4"
                            + (
                                f" (cropped to even from {conv['cropped_from'][1]}x{conv['cropped_from'][0]})"
                                if conv.get("cropped_from")
                                else ""
                            )
                        ),
                    ),
                    (
                        "Folder and manifest",
                        f"<code>{_e(conv['sequence_dir'])}</code>, <code>{_e(conv['manifest'])}</code>",
                    ),
                    (
                        "Validation",
                        _e(
                            f"LibRaw {conv['libraw_passed']}/{conv['frames_converted']}, "
                            f"dng_validate {conv['dng_validate_passed']}/{conv['frames_converted']}"
                        ),
                    ),
                    ("Total size", _e(f"{conv['bytes'] / 1e6:.1f} MB")),
                ]
            ),  # fmt: skip
            "<h2>Timings</h2>",
            _table(
                [
                    (
                        "Seconds per frame",
                        _e(f"mean {spf['mean']:.2f}, median {spf['median']:.2f}, max {spf['max']:.2f}"),
                    ),
                    ("Stage means per frame", _e(stages)),
                    ("Conversion wall time", _e(_duration(conv["wall_s"]))),
                    ("Develop and check wall time", _e(_duration(check["wall_s"] if check else None))),
                    ("Whole run wall time", _e(_duration(wall_time_s(data["state"])))),
                ]
            ),  # fmt: skip
            "<p class='muted'>Per-frame timings: <code>timings.csv</code>.</p>",
        ]
        if conv.get("estimate_4k"):
            out += ["<h2>Estimated cost of a 4K version (not run)</h2>", _table(_estimate_rows(conv["estimate_4k"]))]
    if check:
        out += _stability_section(check, run_dir)
    out += ["<h2>No accuracy claim</h2>", f"<p>{_e(NO_ACCURACY_CLAIM)}</p>"]
    rows = [
        (_e(r["step"]), _e(r["name"]), _e(r.get("status")), _e(r.get("error", "")))
        for r in data["state"].get("steps", [])
    ]
    out += ["<h2>Steps</h2>", _table(rows, ["step", "name", "status", "error"]), "</body></html>"]
    return "\n".join(out)


def render_text(data: dict) -> str:
    verdict, conv, check = data["verdict"], data["convert"], data["check"]
    lines = [f"VIDEO v1 RUN {data['run_id']}: {(verdict or {}).get('status', 'RUNNING')}"]
    lines.append((verdict or {}).get("text") or NO_ACCURACY_CLAIM)
    if data["source"]:
        src = data["source"]
        lines.append(
            f"source: {Path(src['path']).name} {src['codec']} {src['size'][1]}x{src['size'][0]} "
            f"{src['frame_rate']} fps, "
            f"{src['frames']} frames; {src.get('audio')}"
        )
    if conv:
        lines.append(
            f"convert: {conv['frames_converted']} frames, {conv['seconds_per_frame']['mean']:.2f} s per frame, "
            f"{_duration(conv['wall_s'])} in all; LibRaw {conv['libraw_passed']}, "
            f"dng_validate {conv['dng_validate_passed']} passed"
        )
        est = conv.get("estimate_4k")
        if est:
            lines.append(
                f"4K estimate (A100): {_duration(est['seconds_per_frame_x8_ensemble'])} per frame with the x8 "
                f"ensemble ({_duration(est['hours_for_clip_x8_ensemble'] * 3600)} for the clip), "
                f"{_duration(est['seconds_per_frame_no_ensemble'])} without "
                f"({_duration(est['hours_for_clip_no_ensemble'] * 3600)})"
            )
    if check:
        for metric, record in (check.get("metrics") or {}).items():
            lines.append(
                f"stability {metric}: span {_pct(record.get('span'))}, largest step {_pct(record.get('step_max'))}, "
                f"{'passed' if record.get('passed') else 'FAILED'}"
            )
    return "\n".join(lines) + "\n"


def write(run_dir) -> Path:
    run_dir = Path(run_dir)
    data = gather(run_dir)
    (run_dir / "summary.txt").write_text(render_text(data))
    target = run_dir / "report.html"
    target.write_text(render_html(data, run_dir))
    return target
