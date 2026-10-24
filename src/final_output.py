"""The final 50 MP image of a run (PRD sections 10.2, 11.2, 27.3 and 32.3).

Every reconstruction run ends with one final master. Notebooks 01 (human and crowd) and 02 (night scenery)
build that one master themselves and hand it to ``publish``. Notebook 00 chooses it from the outputs it
produced by a documented rule (``select``):

- ``final_output`` names an output (``lanczos``, ``bicubic`` or ``sr_<mode>``): that output, when the run
  produced it. When it did not (for example super-resolution was skipped because the target is not an
  enlargement), the automatic rule decides and the manifest says why.
- ``auto``: a preference order, then the first output that finished with every export valid and, for
  super-resolution, passed the seam gates.

  - Faces first: when the category is ``human`` or the primary face is at least
    ``subjects.TIER1_MIN_FACE_PX`` source pixels on its shortest side (PRD 11.2 tier 1, provisional), the order
    is ``sr_conservative``, ``lanczos``, ``bicubic``. PRD 10.2 prefers the conservative mode for faces and
    keeps generative texture away from identity-critical regions, so the balanced mode is not a fallback here.
  - Every other scene: ``sr_balanced`` (PRD 10.2: default for general scenes), ``sr_conservative``,
    ``lanczos``, ``bicubic``.
  - ``sr_aggressive`` is experimental and never chosen automatically.

The final master is the selected output's own 16-bit TIFF and EXR (byte for byte, hard-linked where the
file system allows, copied otherwise), saved again under ``final/`` with ``final_`` in front of the name and
verified by SHA-256, so its provenance and checks are those of the output it came from.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Callable
from pathlib import Path

AUTO = "auto"
FACE_ORDER = ("sr_conservative", "lanczos", "bicubic")
GENERAL_ORDER = ("sr_balanced", "sr_conservative", "lanczos", "bicubic")
FINAL_DIR = "final"
FINAL_PREFIX = "final_"
KINDS = ("tiff16", "exr")
DRIVE_ROOT = "/content/drive"


def outputs(manifest: dict) -> dict[str, dict]:
    """Every reconstruction of a run by output name: the baselines, then ``sr_<mode>``."""
    modes = (manifest.get("super_resolution") or {}).get("modes", {})
    return {**manifest.get("baselines", {}), **{f"sr_{mode}": summary for mode, summary in modes.items()}}


def preference(category: str, subject: dict | None) -> tuple[tuple[str, ...], str]:
    """The automatic rule's preference order and the reason for it."""
    subject = subject or {}
    face = subject.get("primary_face")
    if category == "human":
        return FACE_ORDER, "the input category is human, and PRD 10.2 prefers the conservative mode for faces"
    if face and subject.get("high_information_face"):
        return FACE_ORDER, (
            f"the primary face is {face['width']}x{face['height']} source px, at least the provisional "
            f"{subject['tier1_min_face_px']} px of a high-information face (PRD 11.2 tier 1), and PRD 10.2 prefers "
            "the conservative mode for faces"
        )
    found = f"the largest face is {face['width']}x{face['height']} source px" if face else "no face was detected"
    return GENERAL_ORDER, f"a general scene ({found}), for which PRD 10.2 makes the balanced mode the default"


def problems(name: str, summary: dict | None) -> list[str]:
    """Why an output cannot be the final master; empty when it can."""
    if summary is None:
        return ["not produced in this run"]
    exports = summary.get("exports", {})
    issues = []
    if not exports:
        issues.append("no master file was exported")
    issues += [f"its {kind} export failed validation" for kind, e in exports.items() if not e.get("valid")]
    if name.startswith("sr_"):
        if "seams" not in summary:
            issues.append("it did not finish")
        elif not summary["seams"].get("passed"):
            issues.append("it failed a seam gate")
    return issues


def select(manifest: dict, requested: str = AUTO) -> dict:
    """Apply the rule to a run's manifest; returns the decision record (``method`` is None if nothing fits)."""
    available = outputs(manifest)
    category = manifest.get("config", {}).get("input_category", "")
    order, basis = preference(category, manifest.get("subject"))
    decision = {"requested": requested, "rule": "auto", "basis": basis, "order": list(order), "skipped": []}
    notes = []
    if requested != AUTO:
        issues = problems(requested, available.get(requested))
        if requested in available and available[requested].get("exports"):
            reason = f"requested with final_output = {requested}"
            if issues:
                reason += f" (warning: {'; '.join(issues)})"
            return {**decision, "rule": "override", "method": requested, "reason": reason, "notes": issues}
        notes.append(f"final_output = {requested} could not be used ({'; '.join(issues)}); the automatic rule decided")
    for name in order:
        issues = problems(name, available.get(name))
        if not issues:
            reason = f"first eligible output in the order {', '.join(order)}, because {basis}"
            if decision["skipped"]:
                passed = "; ".join(f"{s['output']}: {s['why']}" for s in decision["skipped"])
                reason += f"; passed over {passed}"
            return {**decision, "method": name, "reason": reason, "notes": notes}
        decision["skipped"].append({"output": name, "why": "; ".join(issues)})
    return {**decision, "method": None, "reason": "no output finished with valid exports", "notes": notes}


def write_final(decision: dict, manifest: dict, masters_dir: Path, figures_dir: Path | None = None) -> dict:
    """Save the selected output's masters as the final image; returns the decision with ``files`` added."""
    method = decision.get("method")
    if method is None:
        return dict(decision)
    preview = Path(figures_dir) / f"{method}_preview.jpg" if figures_dir is not None else None
    return publish(decision, outputs(manifest)[method].get("exports", {}), masters_dir, preview)


def publish(record: dict, exports: dict, masters_dir: Path, preview: Path | None = None) -> dict:
    """Save validated masters (``exports`` by kind, as the exporters record them) as the run's final image.

    Each TIFF and EXR (and its sidecar) is hard-linked, or copied, into ``final/`` with ``final_`` in front of
    its name and verified by SHA-256. ``preview``, when it exists, is copied next to it as
    ``final_preview.jpg``. Returns ``record`` with ``files`` (and ``preview``) added, the shape ``deliver``
    reads. Specialised pipelines that build their one final master themselves call this directly. Without a
    TIFF or EXR there is no final image: ``method`` becomes None and ``reason`` says so.
    """
    if not any(exports.get(kind) for kind in KINDS):
        return {**record, "method": None, "reason": "no master file was exported", "files": {}}
    record = dict(record)
    final_dir = Path(masters_dir) / FINAL_DIR
    final_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for kind in KINDS:
        export = exports.get(kind)
        if not export:
            continue
        original = Path(export["path"])
        name = original.name if original.name.startswith(FINAL_PREFIX) else f"{FINAL_PREFIX}{original.name}"
        target = final_dir / name
        how = _link_or_copy(original, target)
        sidecar = original.with_name(original.name + ".json")
        if sidecar.exists():
            _link_or_copy(sidecar, target.with_name(target.name + ".json"))
        digest = file_sha256(target)
        if digest != export["file_sha256"]:
            raise RuntimeError(f"{target} does not match {original.name} (SHA-256 {digest})")
        files[kind] = {
            "path": str(target),
            "from": original.name,
            "how": how,
            "size_bytes": target.stat().st_size,
            "file_sha256": digest,
            "verified": True,
            "width": export["width"],
            "height": export["height"],
        }
    record["files"] = files
    if preview is not None and Path(preview).exists():
        shutil.copyfile(preview, Path(preview).with_name("final_preview.jpg"))
        record["preview"] = "figures/final_preview.jpg"
    return record


def storage_bytes(*folders: Path) -> int:
    """Bytes used by every file under ``folders``, counting hard links (the final image) once."""
    sizes = {}
    for folder in folders:
        for path in Path(folder).rglob("*"):
            if path.is_file():
                info = path.stat()
                sizes[(info.st_dev, info.st_ino)] = info.st_size
    return sum(sizes.values())


def deliver(
    manifest: dict,
    *,
    download: bool = True,
    drive_dir: str | Path | None = None,
    drive_root: str | Path = DRIVE_ROOT,
    log: Callable[[str], None] = print,
) -> dict:
    """Hand the final image to the person running the notebook.

    Copies the final folder (TIFF, EXR and their sidecars) into ``drive_dir/<run id>`` when Google Drive is
    mounted at ``drive_root`` and the masters are not on Drive already, then, in Colab and with ``download``,
    downloads the final TIFF (the EXR when no TIFF was exported) through the browser. Returns what was done.
    """
    final = manifest.get("final_output") or {}
    files = final.get("files") or {}
    done: dict = {"downloaded": None, "copied_to": None}
    if not files:
        log(f"no final image, nothing downloaded: {final.get('reason', 'the run did not finish')}")
        return done
    image = Path(files["tiff16" if "tiff16" in files else "exr"]["path"])
    if drive_dir:
        drive, root = Path(drive_dir), Path(drive_root)
        source = Path(next(iter(files.values()))["path"]).parent
        if not root.is_dir():
            log(f"Drive copy skipped: Google Drive is not mounted at {root}")
        elif _within(source, root):
            log(f"final image already on Drive: {source}")
            done["copied_to"] = str(source)
        else:
            target = drive / manifest.get("run_id", "run")
            target.mkdir(parents=True, exist_ok=True)
            for path in sorted(source.iterdir()):
                shutil.copy2(path, target / path.name)
            done["copied_to"] = str(target)
            log(f"final image copied to Drive: {target}")
    if download:
        try:
            from google.colab import files as colab_files
        except ImportError:
            log(f"not in Colab: the final image is {image}")
        else:
            if "tiff16" not in files:
                log("no final TIFF was exported, so the final EXR is downloaded instead")
            log(f"downloading {image.name} ({image.stat().st_size / 1e6:,.0f} MB)")
            colab_files.download(str(image))
            done["downloaded"] = str(image)
    return done


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _link_or_copy(source: Path, target: Path) -> str:
    target.unlink(missing_ok=True)
    try:
        os.link(source, target)
        return "hard link"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _within(path: Path, folder: Path) -> bool:
    try:
        path.resolve().relative_to(folder.resolve())
        return True
    except ValueError:
        return False
