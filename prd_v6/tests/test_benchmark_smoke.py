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


def test_report_prints_no_negative_zero_raw_dicts_or_python_booleans(run, tmp_path):
    import shutil

    run_dir = tmp_path / run[0].name
    shutil.copytree(run[0], run_dir)
    for name in constants.OUTPUT_ORDER:
        path = run_dir / "benchmark" / name / "metrics.json"
        metrics = json.loads(path.read_text())
        metrics["gated"]["k_ev"] = -1e-5
        for row in metrics.get("ablation", {}).values():
            row["delta_A"] = {p: -1e-7 for p in "RGB"}
        path.write_text(json.dumps(metrics))
    verdict = json.loads((run_dir / "verdict.json").read_text())
    for out in verdict["outputs"].values():
        out["k_ev"] = -1e-5
    (run_dir / "verdict.json").write_text(json.dumps(verdict))
    report.write(run_dir)
    page = (run_dir / "report.html").read_text()
    summary = (run_dir / "summary.txt").read_text()
    assert "(+0.000 EV)" in page and "k +0.000 EV" in summary
    assert "-0.000" not in page and "-0.000" not in summary
    assert "<td>True</td>" not in page and "<td>False</td>" not in page
    assert "<td>yes</td>" in page
    ablation = [w for w in run[1]["warnings"] if w.startswith("ablation:")]
    for warning in ablation:
        assert "{" not in warning and warning.endswith(" points)"), warning


def _gated(planes: dict) -> dict:
    from prd_v6 import metrics

    return metrics.finish({"k": 1.0, "k_ev": 0.0, "planes": {p: dict(v) for p, v in planes.items()}})


def _results(**planes_by_output) -> dict:
    steps = {"noise": 0.01, "illuminant": 0.02, "resolution": 0.005, "remaining": 0.001}
    return {
        name: {"gated": _gated(planes_by_output[name]), "ladder_steps": {p: dict(steps) for p in "RGB"}}
        for name in constants.OUTPUT_ORDER
    }


PASSING = {p: {"A": 0.97, "P2": 0.98} for p in "RGB"}


def test_missed_on_p2_only_names_the_failing_plane_and_condition():
    p2_miss = {"R": {"A": 0.959, "P2": 0.9487}, "G": {"A": 0.962, "P2": 0.9442}, "B": {"A": 0.965, "P2": 0.9491}}
    verdict = benchmark.build_verdict(_results(**{"2k": PASSING, "4k": PASSING, "50mp": p2_miss}), [], [])
    assert verdict["status"] == "MISSED"
    assert verdict["text"].startswith("MISSED: 50 MP G P2 = 94.42% (< 95%), A = 95.9%."), verdict["text"]
    assert "50 MP MISSED G P2 94.42%" in verdict["text"]
    assert "Largest contributor (50 MP, G)" in verdict["text"]
    assert verdict["outputs"]["50mp"]["failing"][0] == {"plane": "G", "condition": "P2", "value": 0.9442}


def test_missed_on_a_names_the_failing_plane_and_pass_text_is_unchanged():
    a_miss = {"R": {"A": 0.912, "P2": 0.96}, "G": {"A": 0.97, "P2": 0.97}, "B": {"A": 0.96, "P2": 0.96}}
    missed = benchmark.build_verdict(_results(**{"2k": PASSING, "4k": a_miss, "50mp": PASSING}), [], [])
    assert missed["text"].startswith("MISSED: 4K R A = 91.20% (< 95%), A = 91.2%."), missed["text"]
    assert "2K PASS A 97.0%" in missed["text"]
    passed = benchmark.build_verdict(_results(**{"2k": PASSING, "4k": PASSING, "50mp": PASSING}), [], [])
    assert passed["text"].startswith("PASS: A = 97.0% (2K, R)."), passed["text"]
    assert passed["outputs"]["2k"]["failing"] == []
