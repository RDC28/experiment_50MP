"""The final experiment report: PRD 34 answers, the PRD 33 outcome rules and the quality-to-runtime choice."""

from __future__ import annotations

from pathlib import Path

import pytest

from synthetic_raw_lab import ablation, final_report, reference_test
from synthetic_raw_lab import evaluation_config as ec
from synthetic_raw_lab import run_collection as rc

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ec.load_evaluation_config(ROOT / "configs" / "evaluation_default.yaml")
STATUSES = {"answered", "partial", "not measured", "reasoned"}


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    return rc.collect([ROOT / "evidence" / "colab_runs"], tmp_path_factory.mktemp("work"), CONFIG.aggregation)


def reference_run(gains: dict[str, tuple[float, float]], seconds: dict[str, float] | None = None) -> rc.Run:
    """A reference-test run whose configurations score ``gains[name] = (psnr gain over Lanczos, hf_nmse)``."""
    records = []
    for name, (gain, nmse) in {"lanczos": (0.0, 0.98), **gains}.items():
        records.append(
            {
                **ablation.CONFIGURATIONS[name].to_dict(),
                "seconds": {"upscale": 0, "processing": 0, "total": (seconds or {}).get(name, 1.0)},
                "measurements": {},
                "reference": {
                    "psnr_db": 30 + gain,
                    "ssim": 0.9,
                    "lowband_psnr_db": 40.0,
                    "hf_nmse": nmse,
                    "shadow": {},
                    "faces": [],
                },
                "vs_lanczos": {"reference_psnr_db": gain},
            }
        )
    manifest = {
        "schema": reference_test.SCHEMA,
        "run_id": "20261101T000000Z_reference_test",
        "status": "complete",
        "cases": [
            {
                "key": "portrait_holi_area_x1.7134",
                "reference": "portrait_holi",
                "degradation": {"model": "area"},
                "scale": 1.7134,
                "configurations": records,
            }
        ],
        "references": [],
    }
    return rc.normalise(manifest, Path("."), CONFIG.aggregation)


def test_every_question_is_answered_with_a_status_and_no_em_dash(evidence):
    record = final_report.build(evidence, CONFIG)
    assert [a["number"] for a in record["answers"]] == list(range(1, 19))
    assert [a["question"] for a in record["answers"]] == list(final_report.QUESTIONS)
    assert all(a["status"] in STATUSES and a["answer"] for a in record["answers"])
    text = final_report.render(record, evidence)
    assert "—" not in text and "## 3. Final scientific questions (PRD 34)" in text
    assert sum(record["status_counts"].values()) == 18


def test_notebook_00_evidence_answers_what_it_can(evidence):
    answers = {a["number"]: a for a in final_report.build(evidence, CONFIG)["answers"]}
    assert answers[1]["status"] == "answered" and answers[1]["answer"].startswith("Yes")
    assert answers[3]["status"] == "partial" and any("15.5x" in e for e in answers[3]["evidence"])
    assert answers[10]["status"] == "answered"
    assert any("FAILED its seam gate" in e for e in answers[14]["evidence"])
    assert answers[4]["status"] == "not measured" and "notebook 01" in answers[4]["answer"]
    assert answers[18]["status"] == "reasoned"


def test_outcome_is_undetermined_without_the_reference_test(evidence):
    decision = final_report.build(evidence, CONFIG)["outcome"]
    assert decision["outcome"] is None and "reference test" in decision["reason"]


@pytest.mark.parametrize(
    ("gains", "scores", "expected"),
    [
        ({"sr_conservative": (0.1, 0.99), "sr_balanced": (-0.4, 1.3)}, None, "C"),
        ({"sr_conservative": (0.6, 0.9)}, None, "B"),
        ({"sr_conservative": (1.5, 0.5)}, None, "B"),  # A also needs the blind review
        (
            {"sr_conservative": (1.5, 0.5)},
            {"lanczos": {"overall_preference": 2}, "sr_conservative": {"overall_preference": 4}},
            "A",
        ),
    ],
)
def test_outcome_rules(gains, scores, expected):
    record = final_report.build([reference_run(gains)], CONFIG, scores)
    assert record["outcome"]["outcome"] == expected
    assert record["outcome"]["next_step"] == final_report.NEXT_STEPS[expected]


def test_best_configuration_is_the_cheapest_within_tolerance():
    run = reference_run({"sr_conservative": (0.50, 0.9), "specialized": (0.45, 0.9), "sr_balanced": (0.2, 1.2)})
    ablation_manifest = {
        "schema": ablation.SCHEMA,
        "run_id": "20261101T000000Z_x_ablation",
        "status": "complete",
        "crop": {"box": [0, 0, 100, 100], "full_target": {"source_width": 200, "source_height": 100}},
        "runtime": {"gpu": {"name": "NVIDIA L4"}},
        "configurations": [
            {"name": "sr_conservative", "seconds": {"total": 10.0}},
            {"name": "specialized", "seconds": {"total": 12.0}},
        ],
    }
    ablation_run = rc.normalise(ablation_manifest, Path("."), CONFIG.aggregation)
    tolerant = ec.load_evaluation_config(
        ROOT / "configs" / "evaluation_default.yaml", report={"quality_tolerance_db": 0.1}
    )
    best = final_report.best_configuration(final_report.Evidence([run, ablation_run]), tolerant)
    assert best["choice"] == "sr_conservative" and set(best["eligible"]) == {"sr_conservative", "specialized"}
    assert best["runtime"]["sr_conservative"]["estimated_full_seconds"] == pytest.approx(20.0)


def test_write_produces_markdown_and_json(tmp_path, evidence):
    record = final_report.build(evidence, CONFIG)
    paths = final_report.write(record, evidence, tmp_path, {"portrait": "sheet.png"})
    assert paths["report"].read_text().startswith("# Synthetic RAW 50MP Lab: final experiment report")
    assert "![portrait](sheet.png)" in paths["report"].read_text()
    assert '"schema": "synthetic_raw_lab.phase6_final_report/1"' in paths["json"].read_text()
