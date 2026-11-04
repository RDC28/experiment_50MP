"""Driver steps 3-5 for the notebook's single driver cell.

Steps 3 and 4 gate: an exception is recorded in the run state and re-raised, so the notebook's
`finally` block finalizes an INVALID run. Step 5 is guarded: an exception is recorded with its
traceback and the run is FLAGGED. The summary archive is rebuilt after every step.
"""

from __future__ import annotations

from pathlib import Path

from video_v1 import delivery


def _step(run_dir: Path, step: int, settings: dict, task, *, gating: bool):
    delivery.mark_step_started(run_dir, step)
    try:
        result = task()
    except BaseException as error:
        delivery.mark_step_finished(run_dir, step, error)
        delivery.checkpoint(run_dir, settings)
        if gating or not isinstance(error, Exception):
            raise
        return None
    delivery.mark_step_finished(run_dir, step)
    delivery.checkpoint(run_dir, settings)
    return result


def run(run_dir, settings: dict, log=None) -> None:
    from video_v1 import check, convert, preflight

    run_dir = Path(run_dir)
    log = log or (lambda message: delivery.log(run_dir, message))
    inputs = _step(run_dir, 3, settings, lambda: preflight.run(run_dir, settings, log), gating=True)
    _step(run_dir, 4, settings, lambda: convert.run(run_dir, inputs, settings, log), gating=True)
    _step(run_dir, 5, settings, lambda: check.run(run_dir, inputs, settings, log), gating=False)
