"""Driver step 2 after the clone: pinned install without a runtime restart, then dng_validate (section 9.2).

Standard library only: it runs before the package's own dependencies are installed.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import subprocess
import sys
from pathlib import Path

PINNED = ("rawpy==0.27.1", "spandrel==0.4.2")
DNG_VALIDATE_BUILD = "inputs/dng_validate_build.json"


class RestartRequired(RuntimeError):
    pass


def loaded_distributions() -> dict[str, str]:
    """Versions of the installed distributions whose modules this kernel has already imported."""
    loaded = {name.split(".")[0] for name in list(sys.modules)}
    versions = {}
    for module, dists in metadata.packages_distributions().items():
        if module in loaded:
            for dist in dists:
                try:
                    versions[dist] = metadata.version(dist)
                except metadata.PackageNotFoundError:
                    pass
    return versions


def changed(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    return {name: [old, after[name]] for name, old in before.items() if name in after and after[name] != old}


def install(clone_dir, log=print) -> dict:
    """pip install the clone with the sr extra and the pinned readers; refuse a change that needs a restart."""
    before = loaded_distributions()
    command = [sys.executable, "-m", "pip", "install", "-q", f"{clone_dir}[sr]", *PINNED]
    log("pip install " + " ".join(command[4:]))
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"pip install failed (exit {done.returncode}): {(done.stderr or done.stdout)[-3000:]}")
    importlib.invalidate_caches()
    after = {name: metadata.version(name) for name in before if _installed(name)}
    moved = changed(before, after)
    if moved:
        raise RestartRequired(
            "these already-imported packages changed version and would need a runtime restart, which breaks "
            f"Run all: {moved}. Use a fresh runtime (Runtime > Disconnect and delete runtime), then Run all again."
        )
    return {"command": command[4:], "already_loaded_unchanged": sorted(before)}


def _installed(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def build_dng_validate(run_dir, clone_dir, log=print) -> dict:
    """Build Adobe's validator; a failure is recorded here and stops the run at preflight."""
    if str(clone_dir) not in sys.path:
        sys.path.insert(0, str(clone_dir))
    from prd_v5.dng_validate import build_dng_validate as build

    run_dir = Path(run_dir)
    log("building dng_validate (Adobe DNG SDK)")
    # Outside the run folder, so the SDK sources are never packaged; shared by later runs on the runtime.
    workdir = run_dir.parent / "dng_validate_build"
    result = build(workdir, script=Path(clone_dir) / "prd_v5" / "tools" / "build_dng_validate.sh")
    target = run_dir / DNG_VALIDATE_BUILD
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    log(f"dng_validate: {result['status']}" + (f" ({result.get('reason')})" if result["status"] != "built" else ""))
    return result


def run(run_dir, clone_dir, log=print) -> dict:
    record = {"install": install(clone_dir, log), "dng_validate": build_dng_validate(run_dir, clone_dir, log)}
    (Path(run_dir) / "inputs" / "setup.json").write_text(json.dumps(record, indent=2))
    return record
