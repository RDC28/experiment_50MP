"""Tiny end-to-end benchmark: a 270 x 180 stand-in reference, bicubic in place of HAT-L (local, seconds)."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pytest

from prd_v6 import benchmark, constants, delivery, dng, pipeline, report


def cubic_model(tile: np.ndarray) -> np.ndarray:
    """Stand-in for HAT-L: a x4 upscaler that is not the Lanczos baseline the no-op check compares with."""
    import cv2

    h, w = tile.shape[:2]
    return np.clip(cv2.resize(tile, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC), 0, 1)


@pytest.fixture(scope="module")
def run(tiny_raw, tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("run")
    preset = replace(pipeline.MAX, tile=32, overlap=8)
    cfg = benchmark.Config(
        run_dir=run_dir,
        arw=tiny_raw,
        model=cubic_model,
        preset=preset,
        require_dng_validate=False,
        expect_camera_size=False,
        log=lambda message: None,
    )
    return run_dir, benchmark.run(cfg)


def test_verdict_is_decided_and_finite(run):
    run_dir, verdict = run
    assert verdict["status"] in {"PASS", "MISSED"}, verdict["text"]
    assert verdict["finite"]
    assert set(verdict["outputs"]) == set(constants.OUTPUT_ORDER)
    on_disk = json.loads((run_dir / "verdict.json").read_text())
    assert on_disk["status"] == verdict["status"]


def test_each_output_is_written_at_its_size(run):
    run_dir, _ = run
    sizes = {"2k": (40, 60), "4k": (80, 120), "50mp": (180, 270)}
    for name, hw in sizes.items():
        folder = run_dir / "benchmark" / name
        samples = dng.read_samples(folder / f"exp50v6_{name}.dng")
        assert samples.dtype == np.uint16
        assert samples.shape[:2] == hw
        assert samples.ndim == (2 if name == "50mp" else 3)
        assert dng.tiff16_check(folder / f"exp50v6_{name}.tif", hw)["status"] == "passed"
        metrics = json.loads((folder / "metrics.json").read_text())
        assert metrics["integrity"]["libraw"]["dng"]["status"] == "passed"
        for plane in ("R", "G", "B"):
            assert math.isfinite(metrics["gated"]["planes"][plane]["A"])


def test_binned_outputs_score_high_on_a_smooth_scene(run):
    _, verdict = run
    for name in ("2k", "4k"):
        assert verdict["outputs"][name]["min_A"] > 0.9, verdict["outputs"][name]


def test_ladder_steps_sum_to_the_error(run):
    run_dir, _ = run
    metrics = json.loads((run_dir / "benchmark" / "50mp" / "metrics.json").read_text())
    for plane, steps in metrics["ladder_steps"].items():
        a0 = metrics["ladder"]["L0"]["planes"][plane]["A"]
        assert sum(steps.values()) == pytest.approx(1 - a0, abs=1e-6)


def test_report_and_summary_archive(run):
    run_dir, verdict = run
    (run_dir / "config.json").write_text(json.dumps({"settings": {}}))
    report.write(run_dir)
    assert verdict["status"] in (run_dir / "summary.txt").read_text()
    assert (run_dir / "report.html").stat().st_size > 1000
    summary = delivery.build_summary(run_dir)
    assert summary.exists()
