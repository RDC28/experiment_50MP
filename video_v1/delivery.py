"""Run state, verdict, split archives, Drive mirror and downloads for the video notebook (README section 5).

Standard library only (with `prd_v6.delivery`, which is too), so the notebook's `finally` block and
`download_results` work after a failed install and after a kernel crash: a new cell can run

    import sys; sys.path.insert(0, "/content/experiment_50MP")
    from video_v1.delivery import download_results; download_results("<run_dir>")
"""

from __future__ import annotations

import os
import shutil
import traceback as _traceback
import zipfile
from pathlib import Path

from prd_v6.delivery import MULTIPLE_DOWNLOADS_WARNING, _read_json, _sha256, _write_json, log, now

STEPS = {1: "Input", 2: "Setup", 3: "Preflight", 4: "Convert", 5: "Develop and check"}
GATING_STEPS = (1, 2, 3, 4)
CLONE_DIR = "/content/experiment_50MP"
DRIVE_ROOT = Path("/content/drive/MyDrive/exp50mp_video_v1")
SUMMARY_SUFFIXES = {".json", ".html", ".txt", ".log", ".md", ".csv"}
SUMMARY_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SUMMARY_IMAGE_MAX_BYTES = 5 * 1024 * 1024
SUMMARY_VIDEO_SUFFIXES = {".mp4"}
STORED_SUFFIXES = {".dng", ".zip", ".jpg", ".jpeg", ".png", ".mp4"}
EXCLUDED_DIRS = {"archives"}
INPUTS_DIR = "inputs"  # the source clip stays on the runtime; only its small text records are packaged
DNG_DIR = "dng"
PART_MAX_BYTES = 500 * 1000 * 1000
NO_ACCURACY_CLAIM = (
    "No accuracy claim: there is no real RAW of this clip to score against, and the v6 benchmark result "
    "(each colour plane within the 95% target against one real Sony A1 II RAW) does not transfer to video frames."
)


def run_id(run_dir) -> str:
    return Path(run_dir).name


def recovery_line(run_dir, clone_dir: str = CLONE_DIR) -> str:
    return (
        f'import sys; sys.path.insert(0, "{clone_dir}"); '
        f'from video_v1.delivery import download_results; download_results("{run_dir}")'
    )


# ---------------------------------------------------------------- run state


def load_state(run_dir) -> dict:
    return _read_json(Path(run_dir) / "state.json", {"steps": []})


def _step_record(state: dict, step: int) -> dict:
    for record in state["steps"]:
        if record["step"] == step:
            return record
    record = {"step": step, "name": STEPS.get(step, f"step {step}")}
    state["steps"].append(record)
    return record


def mark_step_started(run_dir, step: int) -> None:
    state = load_state(run_dir)
    state["step_started"] = step
    _step_record(state, step).update(status="started", started_at=now())
    _write_json(Path(run_dir) / "state.json", state)
    log(run_dir, f"step {step} started: {STEPS.get(step)}")


def mark_step_finished(run_dir, step: int, error: BaseException | None = None) -> None:
    state = load_state(run_dir)
    record = _step_record(state, step)
    record["finished_at"] = now()
    if error is None:
        record["status"] = "finished"
    else:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = "".join(_traceback.format_exception(type(error), error, error.__traceback__))
    _write_json(Path(run_dir) / "state.json", state)
    log(run_dir, f"step {step} {record['status']}: {STEPS.get(step)}" + (f" ({record['error']})" if error else ""))


def failed_steps(state: dict) -> list[dict]:
    return [r for r in state["steps"] if r.get("status") == "failed"]


def finished_steps(state: dict) -> set[int]:
    return {r["step"] for r in state["steps"] if r.get("status") == "finished"}


def unfinished_steps(state: dict) -> list[int]:
    return sorted(r["step"] for r in state["steps"] if r.get("status") == "started")


# ---------------------------------------------------------------- verdict


def _flags(convert: dict, check: dict | None, state: dict) -> list[str]:
    """Why a run that converted every frame is FLAGGED rather than COMPLETE."""
    flags = []
    if convert.get("validation_failures"):
        flags.append(
            f"{convert['validation_failures']} of {convert['frames_converted']} DNGs failed LibRaw or dng_validate "
            f"(frames {convert.get('failed_frames')})"
        )
    if convert.get("frame_count_mismatch"):
        flags.append(convert["frame_count_mismatch"])
    step5 = next((r for r in state["steps"] if r["step"] == 5), {})
    if step5.get("status") == "failed":
        flags.append(f"step 5 (Develop and check) raised {step5.get('error')}")
    if check is None:
        if step5.get("status") != "failed":
            flags.append("the stability check did not run")
    elif not check.get("passed"):
        failing = [m for m, r in (check.get("metrics") or {}).items() if not r.get("passed")]
        flags.append(
            "stability check failed"
            + (f" on {', '.join(failing)}" if failing else "")
            + (f" ({check['reason']})" if check.get("reason") else "")
        )
    return flags


def finalize_verdict(run_dir, exception_info: dict | None = None, *, process_lost: bool = False) -> dict:
    """COMPLETE, FLAGGED (it says what) or INVALID (a gating step failed; it says why)."""
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    convert = _read_json(run_dir / "convert.json")
    check = _read_json(run_dir / "check.json")
    errors = [f"step {r['step']} ({r['name']}) raised {r.get('error', 'an exception')}" for r in failed_steps(state)]
    if exception_info and not any(f"step {exception_info.get('step')} " in e for e in errors):
        errors.append(f"step {exception_info.get('step')} raised {exception_info.get('error')}")
    lost_step = None
    if process_lost:
        lost_step = (unfinished_steps(state) or [state.get("step_started", 0)])[-1]
        errors.append(f"process lost during step {lost_step}")
    gating_failure = (
        any(r["step"] in GATING_STEPS for r in failed_steps(state))
        or (exception_info is not None and exception_info.get("step") in GATING_STEPS)
        or (lost_step in GATING_STEPS)
    )
    if gating_failure or convert is None or 4 not in finished_steps(state):
        reason = errors[0] if errors else "the conversion did not run to completion"
        status, reasons = "INVALID", [reason]
        text = f"INVALID: {reason}"
    else:
        reasons = _flags(convert, check, state)
        if process_lost and lost_step not in GATING_STEPS:
            reasons.append(errors[-1])
        status = "FLAGGED" if reasons else "COMPLETE"
        frames = f"{convert['frames_converted']} of {convert['frames_source']} frames"
        what = f"{frames} converted to 16-bit RGGB DNGs at {convert['size'][1]}x{convert['size'][0]}"
        text = f"{status}: {what}" + (f"; {'; '.join(reasons)}." if reasons else ", all validated, stability passed.")
    verdict = {
        "status": status,
        "text": f"{text} {NO_ACCURACY_CLAIM}",
        "reasons": reasons,
        "no_accuracy_claim": NO_ACCURACY_CLAIM,
        "completed_with_errors": errors,
        "convert": {k: v for k, v in (convert or {}).items() if k != "estimate_4k"},
        "stability_passed": None if check is None else bool(check.get("passed")),
        "steps": state["steps"],
        "final": True,
        "finalized_at": now(),
    }
    _write_json(run_dir / "verdict.json", verdict)
    return verdict


# ---------------------------------------------------------------- archives


def _files(root: Path, base: Path):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(base)
        if not path.is_file() or rel.parts[0] in EXCLUDED_DIRS or path.name.endswith((".tmp", ".part")):
            continue
        if rel.parts[0] == INPUTS_DIR and path.suffix.lower() not in SUMMARY_SUFFIXES:
            continue
        yield path, rel


def _in_summary(path: Path, rel: Path) -> bool:
    suffix = path.suffix.lower()
    if rel.parts[0] == DNG_DIR:
        return suffix == ".json"  # the sequence manifest, not the frames
    if suffix in SUMMARY_SUFFIXES or suffix in SUMMARY_VIDEO_SUFFIXES:
        return True
    return suffix in SUMMARY_IMAGE_SUFFIXES and path.stat().st_size <= SUMMARY_IMAGE_MAX_BYTES


def _zip(target: Path, entries) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", allowZip64=True) as archive:
        for path, rel in entries:
            method = zipfile.ZIP_STORED if path.suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
            archive.write(path, str(rel), compress_type=method)
    os.replace(tmp, target)
    return target


def write_manifest(run_dir) -> Path:
    run_dir = Path(run_dir)
    files = [
        {"path": str(rel), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path, rel in _files(run_dir, run_dir)
        if rel.name != "manifest.json"
    ]
    target = run_dir / "manifest.json"
    _write_json(target, {"run_id": run_id(run_dir), "written_at": now(), "files": files})
    return target


def summary_path(run_dir) -> Path:
    return Path(run_dir) / "archives" / f"{run_id(run_dir)}_01_summary.zip"


def build_summary(run_dir) -> Path:
    run_dir = Path(run_dir)
    target = summary_path(run_dir)
    for old in target.parent.glob(f"{run_id(run_dir)}_01_summary*.zip"):
        old.unlink()
    return _zip(target, [(p, r) for p, r in _files(run_dir, run_dir) if _in_summary(p, r)])


def split_parts(entries: list[tuple[Path, Path]], limit: int) -> list[list[tuple[Path, Path]]]:
    """Frames in order, greedily packed into parts of at most `limit` bytes (a larger file gets its own part)."""
    parts: list[list[tuple[Path, Path]]] = []
    size = 0
    for path, rel in entries:
        length = path.stat().st_size
        if not parts or (size + length > limit and parts[-1]):
            parts.append([])
            size = 0
        parts[-1].append((path, rel))
        size += length
    return parts


def build_dng_archives(run_dir) -> list[Path]:
    """The DNG sequence in parts of at most PART_MAX_BYTES, each with the sequence manifest; _partial
    when step 4 did not finish."""
    run_dir = Path(run_dir)
    rid = run_id(run_dir)
    for old in (run_dir / "archives").glob(f"{rid}_*_dng_part*.zip"):
        old.unlink()
    folder = run_dir / DNG_DIR
    if not folder.is_dir():
        return []
    entries = list(_files(folder, run_dir))
    frames = [(p, r) for p, r in entries if p.suffix.lower() == ".dng"]
    manifests = [(p, r) for p, r in entries if p.suffix.lower() == ".json"]
    if not frames:
        return []
    partial = "" if 4 in finished_steps(load_state(run_dir)) else "_partial"
    parts = split_parts(frames, PART_MAX_BYTES)
    built = []
    for number, part in enumerate(parts, start=1):
        target = run_dir / "archives" / f"{rid}_{number + 1:02d}_dng_part{number}of{len(parts)}{partial}.zip"
        built.append(_zip(target, manifests + part))
    return built


def ordered_archives(run_dir) -> list[Path]:
    return sorted((Path(run_dir) / "archives").glob(f"{run_id(run_dir)}_*.zip"))


def mirror_to_drive(run_dir, paths) -> list[str]:
    """Copy files to MyDrive/exp50mp_video_v1/<run_id>/ when Drive is mounted; never raises."""
    if not DRIVE_ROOT.parent.is_dir():
        return []
    copied = []
    target = DRIVE_ROOT / run_id(run_dir)
    for path in paths:
        try:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / Path(path).name)
            copied.append(str(target / Path(path).name))
        except OSError as error:
            log(run_dir, f"Drive copy failed for {Path(path).name}: {error}")
    return copied


def load_settings(run_dir) -> dict:
    return _read_json(Path(run_dir) / "config.json", {}).get("settings", {})


def _render_report(run_dir) -> None:
    try:
        from video_v1 import report
    except Exception:  # noqa: BLE001 - the report is optional; the archives are not
        return
    try:
        report.write(run_dir)
    except Exception as error:  # noqa: BLE001
        log(run_dir, f"report failed: {type(error).__name__}: {error}")


def checkpoint(run_dir, settings: dict | None = None) -> None:
    """After each driver step: rebuild the summary archive and mirror it to Drive when enabled."""
    settings = settings if settings is not None else load_settings(run_dir)
    try:
        _render_report(run_dir)
        summary = build_summary(run_dir)
        if settings.get("DRIVE_MIRROR"):
            mirror_to_drive(run_dir, [summary])
    except Exception as error:  # noqa: BLE001 - a checkpoint must never break the run
        log(run_dir, f"checkpoint failed: {type(error).__name__}: {error}")


# ---------------------------------------------------------------- downloads and banner


def _download(paths: list[Path], run_dir) -> list[dict]:
    try:
        from google.colab import files  # type: ignore[import-not-found]
    except ImportError:
        files = None
    print(MULTIPLE_DOWNLOADS_WARNING, flush=True)
    records = []
    for path in paths:
        record = {"archive": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        print(f"download {path.name}  {record['bytes'] / 1e6:.1f} MB  sha256 {record['sha256']}", flush=True)
        if files is None:
            record["status"] = "not in Colab: use the file on disk"
        else:
            try:
                files.download(str(path))
                record["status"] = "triggered"
            except Exception as error:  # noqa: BLE001 - a blocked download must not stop the others
                record["status"] = f"failed: {type(error).__name__}: {error}"
        records.append(record)
    _write_json(Path(run_dir) / "downloads.json", records)
    return records


def banner(run_dir, verdict: dict, archives: list[Path], settings: dict) -> str:
    rule = "=" * 78
    lines = [rule, f"VIDEO v1 RUN {run_id(run_dir)}: {verdict.get('status')}", verdict.get("text", "")]
    convert = verdict.get("convert") or {}
    if convert.get("seconds_per_frame"):
        lines.append(
            f"  {convert['seconds_per_frame']['mean']:.2f} s per frame, {convert['wall_s'] / 60:.1f} min for "
            f"{convert['frames_converted']} frames; report.html in the summary archive has the computed 4K estimate"
        )
    lines.append("Archives (download order):")
    lines += [f"  {a.name}  {a.stat().st_size / 1e6:.1f} MB" for a in archives if a.exists()]
    lines += [
        f"Run folder: {run_dir}",
        MULTIPLE_DOWNLOADS_WARNING,
        "Recovery, in a new cell: " + recovery_line(run_dir),
        "Or download the archives from the Files pane (folder above, archives/).",
        f"Drive mirror: {DRIVE_ROOT / run_id(run_dir)}. Files on the runtime are lost when Colab recycles it."
        if settings.get("DRIVE_MIRROR")
        else "Drive mirror is off: files on the runtime are lost when Colab recycles it.",
        rule,
    ]
    return "\n".join(lines)


def _package(run_dir, settings: dict, verdict: dict, *, download: bool) -> list[Path]:
    run_dir = Path(run_dir)
    _render_report(run_dir)
    write_manifest(run_dir)
    data = build_dng_archives(run_dir)
    _write_json(
        run_dir / "archives.json",
        [{"archive": summary_path(run_dir).name, "sha256": "self"}]
        + [{"archive": a.name, "bytes": a.stat().st_size, "sha256": _sha256(a)} for a in data],
    )
    archives = [build_summary(run_dir)] + data
    if settings.get("DRIVE_MIRROR"):
        mirror_to_drive(run_dir, archives)
    if download:
        _download(archives, run_dir)
    print(banner(run_dir, verdict, archives, settings), flush=True)
    return archives


def finalize_run(run_dir, settings: dict | None = None, exception_info: dict | None = None, *, download=True):
    """The driver cell's `finally` block: final verdict, manifest, archives, downloads, banner."""
    run_dir = Path(run_dir)
    settings = settings if settings is not None else load_settings(run_dir)
    verdict = finalize_verdict(run_dir, exception_info)
    archives = _package(run_dir, settings, verdict, download=download)
    _write_json(run_dir / "run_status.json", {"finalized": True, "at": now(), "status": verdict["status"]})
    return verdict, archives


def download_results(run_dir, download: bool = True):
    """Recovery after a blocked download, a closed tab or a kernel crash.

    Rebuilds the summary and every DNG part without recomputing results, and downloads. If
    the `finally` block never ran, the process was lost: the verdict records the step it was lost in.
    """
    run_dir = Path(run_dir)
    settings = load_settings(run_dir)
    finalized = (_read_json(run_dir / "run_status.json", {}) or {}).get("finalized", False)
    if finalized:
        verdict = _read_json(run_dir / "verdict.json", {})
    else:
        verdict = finalize_verdict(run_dir, process_lost=True)
        _write_json(
            run_dir / "run_status.json",
            {"finalized": True, "at": now(), "status": verdict["status"], "by": "download_results"},
        )
    return verdict, _package(run_dir, settings, verdict, download=download)
