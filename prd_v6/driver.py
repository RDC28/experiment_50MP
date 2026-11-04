"""Driver steps 3-6 for the notebook's single driver cell (section 9.2).

Steps 3 and 4 gate: an exception is recorded in the run state and re-raised, so the notebook's
`finally` block finalizes an INVALID run. Steps 5 and 6 are each guarded: an exception is recorded
with its traceback and the next step still runs. The summary archive is rebuilt after every step.
"""

from __future__ import annotations

from pathlib import Path

from prd_v6 import colour, delivery


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
    from prd_v6 import benchmark, pipeline, preflight, production, sweeps

    run_dir = Path(run_dir)
    log = log or (lambda message: delivery.log(run_dir, message))
    if settings.get("PRESET", "max") != "max":
        raise ValueError(f"PRESET {settings.get('PRESET')!r}: v6 defines only the 'max' preset (section 9.3)")
    # Parse both declarations before any download: a bad value fails at the start, not after step 3.
    benchmark_cct = colour.parse_declared_cct(settings.get("ILLUMINANT_CCT_K"))
    colour.parse_declared_cct(settings.get("SOURCE_ILLUMINANT_CCT_K"), "SOURCE_ILLUMINANT_CCT_K")
    delivery.write_illuminant(run_dir, colour.resolve_illuminant(pipeline.camera_matrix(), benchmark_cct))
    inputs = _step(run_dir, 3, settings, lambda: preflight.run(run_dir, settings, log), gating=True)
    config = benchmark.Config(
        run_dir=run_dir,
        arw=inputs["arw"],
        model=inputs["model"],
        preset=pipeline.MAX,
        illuminant_cct_k=benchmark_cct,
        dng_validate_binary=inputs["dng_validate"],
        log=log,
    )
    _step(run_dir, 4, settings, lambda: benchmark.run(config), gating=True)
    if settings.get("SOURCE_4K_URL") or settings.get("SOURCE_2K_URL"):
        _step(
            run_dir,
            5,
            settings,
            lambda: production.run(run_dir, settings, inputs["model"], inputs["source_errors"], log=log),
            gating=False,
        )
    else:
        log("step 5 skipped: no SOURCE_4K_URL or SOURCE_2K_URL, so the run is benchmark-only")
    _step(run_dir, 6, settings, lambda: sweeps.run(run_dir, inputs["arw"], log), gating=False)
