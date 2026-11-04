"""Run state, verdict finalization, archives, Drive mirror and downloads (sections 6.6, 9.2, 10).

Standard library only, so the notebook's `finally` block and `download_results` work after a failed
install and after a kernel crash: a new cell can run

    import sys; sys.path.insert(0, "/content/experiment_50MP")
    from prd_v6.delivery import download_results; download_results("<run_dir>")
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import shutil
import traceback as _traceback
import zipfile
from pathlib import Path

STEPS = {
    1: "Drive mount",
    2: "Setup",
    3: "Preflight",
    4: "Benchmark",
    5: "Production",
    6: "Sweeps and secondary reports",
}
GATING_STEPS = (1, 2, 3, 4)
CLONE_DIR = "/content/experiment_50MP"
DRIVE_ROOT = Path("/content/drive/MyDrive/exp50mp_v6")
SUMMARY_SUFFIXES = {".json", ".html", ".txt", ".log", ".md", ".csv"}
SUMMARY_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SUMMARY_IMAGE_MAX_BYTES = 5 * 1024 * 1024
EXCLUDED_DIRS = {"archives"}
INPUTS_DIR = "inputs"  # downloaded ARW and checkpoint: only its small text records are packaged
STORED_SUFFIXES = {".dng", ".tif", ".tiff", ".zip", ".jpg", ".jpeg", ".png", ".npy", ".arw", ".pth"}
# (order, archive name, folder inside the run, producing step)
DATA_ARCHIVES = (
    (2, "benchmark_2k", "benchmark/2k", 4),
    (3, "benchmark_4k", "benchmark/4k", 4),
    (4, "benchmark_50mp", "benchmark/50mp", 4),
    (5, "production_4k_source", "production/4k_source", 5),
    (6, "production_2k_source", "production/2k_source", 5),
)
MULTIPLE_DOWNLOADS_WARNING = (
    "Chrome may ask 'This site is trying to download multiple files'. Allow it for colab.research.google.com; "
    "if it was blocked only the summary archive arrives (recovery below)."
)


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def _write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.replace(tmp, path)


def run_id(run_dir) -> str:
    return Path(run_dir).name


def recovery_line(run_dir, clone_dir: str = CLONE_DIR) -> str:
    return (
        f'import sys; sys.path.insert(0, "{clone_dir}"); '
        f'from prd_v6.delivery import download_results; download_results("{run_dir}")'
    )


def log(run_dir, message: str) -> None:
    line = f"{now()} {message}"
    print(line, flush=True)
    try:
        with open(Path(run_dir) / "run.log", "a") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


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
    record = _step_record(state, step)
    record.update(status="started", started_at=now())
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


def unfinished_steps(state: dict) -> list[int]:
    return sorted(r["step"] for r in state["steps"] if r.get("status") == "started")


def failed_steps(state: dict) -> list[dict]:
    return [r for r in state["steps"] if r.get("status") == "failed"]


# ---------------------------------------------------------------- verdict


def write_illuminant(run_dir, illuminant: dict) -> None:
    """The benchmark's white-balance source, recorded before any step so even an INVALID verdict states it."""
    _write_json(Path(run_dir) / "illuminant.json", illuminant)


def finalize_verdict(run_dir, exception_info: dict | None = None, *, process_lost: bool = False) -> dict:
    """Write the final verdict (section 6.6) from the step-4 verdict and the run state.

    An exception or a process loss in steps 1-4 makes the run INVALID. In steps 5 and 6 it keeps the
    benchmark's PASS or MISSED, marked "completed with errors".
    """
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    verdict = _read_json(run_dir / "verdict.json")
    errors: list[str] = []
    for record in failed_steps(state):
        errors.append(f"step {record['step']} ({record['name']}) raised {record.get('error', 'an exception')}")
    if exception_info and not any(f"step {exception_info.get('step')} " in e for e in errors):
        errors.append(f"step {exception_info.get('step')} raised {exception_info.get('error')}")
    if process_lost:
        lost = unfinished_steps(state) or [state.get("step_started", 0)]
        errors.append(f"process lost during step {lost[-1]}")
    gating_failure = any(r["step"] in GATING_STEPS for r in failed_steps(state)) or (
        exception_info is not None and exception_info.get("step") in GATING_STEPS
    )
    if verdict is None or gating_failure or verdict.get("status") not in {"PASS", "MISSED", "INVALID"}:
        reason = errors[0] if errors else "the benchmark did not run to completion"
        verdict = {
            "status": "INVALID",
            "text": f"INVALID: {reason}",
            "invalid_reasons": [reason],
            "warnings": [],
            "outputs": (verdict or {}).get("outputs", {}),
            "accuracy_percent": None,
        }
        recorded = (verdict or {}).get("illuminant") or _read_json(run_dir / "illuminant.json")
        if recorded:
            verdict["illuminant"] = recorded
            verdict["text"] += f" Illuminant: {recorded['label']}."
    elif errors:
        verdict["text"] = f"{verdict['text']} Completed with errors: {'; '.join(errors)}."
    verdict["completed_with_errors"] = errors
    verdict["steps"] = state["steps"]
    verdict["final"] = True
    verdict["finalized_at"] = now()
    _write_json(run_dir / "verdict.json", verdict)
    return verdict


# ---------------------------------------------------------------- archives


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


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
    if suffix in SUMMARY_SUFFIXES:
        return True
    if suffix in SUMMARY_IMAGE_SUFFIXES:
        if path.name.startswith("error_map_") and not path.stem.endswith("_small"):
            return False
        return path.stat().st_size <= SUMMARY_IMAGE_MAX_BYTES
    return False


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


def build_archives(run_dir, settings: dict, *, only_missing: bool = False) -> list[Path]:
    """Data archives 2-7 (section 10.2), labelled _partial when their producing step did not finish."""
    run_dir = Path(run_dir)
    state = load_state(run_dir)
    finished = {r["step"] for r in state["steps"] if r.get("status") == "finished"}
    rid = run_id(run_dir)
    built = []
    for order, name, folder, step in DATA_ARCHIVES:
        source = run_dir / folder
        if not source.is_dir() or not any(source.rglob("*")):
            continue
        partial = "" if step in finished else "_partial"
        target = run_dir / "archives" / f"{rid}_{order:02d}_{name}{partial}.zip"
        existing = list(target.parent.glob(f"{rid}_{order:02d}_{name}*.zip"))
        if only_missing and existing:
            built.extend(existing)
            continue
        for old in existing:
            old.unlink()
        built.append(_zip(target, list(_files(source, run_dir))))
    if settings.get("DOWNLOAD_ALL"):
        partial = "" if all(r.get("status") == "finished" for r in state["steps"]) else "_partial"
        target = run_dir / "archives" / f"{rid}_07_all{partial}.zip"
        existing = list(target.parent.glob(f"{rid}_07_all*.zip"))
        if only_missing and existing:
            built.extend(existing)
        else:
            for old in existing:
                old.unlink()
            built.append(_zip(target, list(_files(run_dir, run_dir))))
    return built


def ordered_archives(run_dir) -> list[Path]:
    return sorted((Path(run_dir) / "archives").glob(f"{run_id(run_dir)}_*.zip"))


def mirror_to_drive(run_dir, paths) -> list[str]:
    """Copy files to MyDrive/exp50mp_v6/<run_id>/ when Drive is mounted; never raises."""
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


def load_settings(run_dir) -> dict:
    return _read_json(Path(run_dir) / "config.json", {}).get("settings", {})


def _render_report(run_dir) -> None:
    try:
        from prd_v6 import report
    except Exception:  # noqa: BLE001 - the report is optional; the archives are not
        return
    try:
        report.write(run_dir)
    except Exception as error:  # noqa: BLE001
        log(run_dir, f"report failed: {type(error).__name__}: {error}")


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
    from prd_v6 import report

    rule = "=" * 78
    lines = [rule, f"PRD v6 RUN {run_id(run_dir)}: {verdict.get('status')}", verdict.get("text", "")]
    for name, out in (verdict.get("outputs") or {}).items():
        planes = ", ".join(
            f"{p} A {100 * v['A']:.2f}% P2 {100 * v['P2']:.2f}%" for p, v in (out.get("planes") or {}).items()
        )
        lines.append(f"  {out.get('label', name)}: {'PASS' if out.get('passes') else 'MISSED'}  {planes}")
    illuminant = verdict.get("illuminant")
    if illuminant:
        lines.append(f"  illuminant: {illuminant['label']}" + report.estimate_note(illuminant))
    for warning in verdict.get("warnings", [])[:10]:
        lines.append(f"  warning: {warning}")
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


def _package(run_dir, settings: dict, verdict: dict, *, download: bool, only_missing: bool) -> list[Path]:
    run_dir = Path(run_dir)
    _render_report(run_dir)
    write_manifest(run_dir)
    data = sorted(build_archives(run_dir, settings, only_missing=only_missing))
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
    archives = _package(run_dir, settings, verdict, download=download, only_missing=False)
    _write_json(run_dir / "run_status.json", {"finalized": True, "at": now(), "status": verdict["status"]})
    return verdict, archives


def download_results(run_dir, download: bool = True):
    """Recovery after a blocked download, a closed tab or a kernel crash (sections 9.2, 10.3).

    Rebuilds the summary, builds any missing archives without recomputing results, and downloads. If
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
    return verdict, _package(run_dir, settings, verdict, download=download, only_missing=True)
