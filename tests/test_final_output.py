import json
import sys
import types
from pathlib import Path

import pytest

from synthetic_raw_lab import final_output


def summary(valid=True, seams=True, sr=False):
    record = {"exports": {"tiff16": {"valid": valid}, "exr": {"valid": True}}}
    if sr:
        record["seams"] = {"passed": seams}
    return record


def manifest(category="universal", face=None, modes=("conservative", "balanced"), **mode_overrides):
    subject = {
        "primary_face": face,
        "high_information_face": bool(face) and min(face["width"], face["height"]) >= 96,
        "tier1_min_face_px": 96,
    }
    return {
        "config": {"input_category": category},
        "subject": subject,
        "baselines": {"lanczos": summary(), "bicubic": summary()},
        "super_resolution": {"modes": {m: mode_overrides.get(m, summary(sr=True)) for m in modes}},
    }


PORTRAIT_FACE = {"x": 2900, "y": 400, "width": 700, "height": 700}


def test_a_high_information_face_selects_the_conservative_mode():
    decision = final_output.select(manifest(face=PORTRAIT_FACE))
    assert decision["method"] == "sr_conservative" and decision["rule"] == "auto"
    assert "700x700 source px" in decision["reason"] and "PRD 11.2 tier 1" in decision["reason"]


def test_the_human_category_selects_the_conservative_mode_without_a_detection():
    assert final_output.select(manifest(category="human"))["method"] == "sr_conservative"


def test_a_general_scene_selects_the_balanced_mode():
    decision = final_output.select(manifest())
    assert decision["method"] == "sr_balanced" and "no face was detected" in decision["reason"]
    small_face = {"x": 0, "y": 0, "width": 40, "height": 50}
    assert final_output.select(manifest(face=small_face))["method"] == "sr_balanced"


def test_faces_never_fall_back_to_the_perceptual_network():
    decision = final_output.select(manifest(face=PORTRAIT_FACE, modes=("balanced",)))
    assert decision["method"] == "lanczos"
    assert decision["skipped"] == [{"output": "sr_conservative", "why": "not produced in this run"}]


def test_outputs_failing_validation_or_seams_are_passed_over():
    m = manifest(balanced=summary(sr=True, seams=False), conservative=summary(sr=True, valid=False))
    decision = final_output.select(m)
    assert decision["method"] == "lanczos"
    assert decision["reason"].endswith(
        "; passed over sr_balanced: it failed a seam gate; sr_conservative: its tiff16 export failed validation"
    )
    assert [s["why"] for s in decision["skipped"]] == [
        "it failed a seam gate",
        "its tiff16 export failed validation",
    ]


def test_the_aggressive_mode_is_never_chosen_automatically():
    assert final_output.select(manifest(modes=("aggressive",)))["method"] == "lanczos"


def test_without_super_resolution_lanczos_then_bicubic():
    m = manifest(modes=())
    assert final_output.select(m)["method"] == "lanczos"
    m["baselines"] = {"bicubic": summary()}
    assert final_output.select(m)["method"] == "bicubic"
    m["baselines"] = {"bicubic": summary(valid=False)}
    decision = final_output.select(m)
    assert decision["method"] is None and decision["reason"] == "no output finished with valid exports"


def test_an_override_wins_and_warns_about_problems():
    decision = final_output.select(manifest(face=PORTRAIT_FACE, modes=("aggressive",)), "sr_aggressive")
    assert (decision["method"], decision["rule"]) == ("sr_aggressive", "override")
    m = manifest(balanced=summary(sr=True, seams=False))
    decision = final_output.select(m, "sr_balanced")
    assert decision["method"] == "sr_balanced" and "warning: it failed a seam gate" in decision["reason"]


def test_an_override_that_was_not_produced_falls_back_to_the_rule():
    m = manifest(modes=())
    m["super_resolution"] = {"skipped": "not an enlargement", "modes": {}}
    decision = final_output.select(m, "sr_conservative")
    assert decision["method"] == "lanczos" and decision["rule"] == "auto"
    assert decision["notes"] == [
        "final_output = sr_conservative could not be used (not produced in this run); the automatic rule decided"
    ]


def written_run(tmp_path):
    masters = tmp_path / "outputs" / "run1"
    figures = tmp_path / "reports" / "run1" / "figures"
    masters.mkdir(parents=True)
    figures.mkdir(parents=True)
    exports = {}
    for kind, name in (("tiff16", "scene_sr_conservative_10x6_srgb16.tif"), ("exr", "scene_sr_conservative.exr")):
        path = masters / name
        path.write_bytes(name.encode() * 100)
        (masters / f"{name}.json").write_text(json.dumps({"provenance": kind}))
        exports[kind] = {
            "path": str(path),
            "file_sha256": final_output.file_sha256(path),
            "valid": True,
            "width": 10,
            "height": 6,
        }
    (figures / "sr_conservative_preview.jpg").write_bytes(b"jpeg")
    m = manifest(face=PORTRAIT_FACE, modes=("conservative",))
    m["run_id"] = "run1"
    m["super_resolution"]["modes"]["conservative"] = {"exports": exports, "seams": {"passed": True}}
    return m, masters, figures


def test_the_final_image_is_the_selected_masters_saved_again_and_verified(tmp_path):
    m, masters, figures = written_run(tmp_path)
    record = final_output.write_final(final_output.select(m), m, masters, figures)
    tiff = Path(record["files"]["tiff16"]["path"])
    assert tiff == masters / "final" / "final_scene_sr_conservative_10x6_srgb16.tif"
    assert tiff.read_bytes() == (masters / "scene_sr_conservative_10x6_srgb16.tif").read_bytes()
    assert record["files"]["tiff16"]["verified"] and record["files"]["tiff16"]["how"] in {"hard link", "copy"}
    assert json.loads(tiff.with_name(tiff.name + ".json").read_text()) == {"provenance": "tiff16"}
    assert Path(record["files"]["exr"]["path"]).name == "final_scene_sr_conservative.exr"
    assert record["preview"] == "figures/final_preview.jpg" and (figures / "final_preview.jpg").read_bytes() == b"jpeg"
    # a second run of the same stage replaces the files instead of failing
    assert final_output.write_final(final_output.select(m), m, masters, figures)["files"]["tiff16"]["verified"]


def test_a_final_image_that_does_not_match_its_master_is_rejected(tmp_path):
    m, masters, figures = written_run(tmp_path)
    m["super_resolution"]["modes"]["conservative"]["exports"]["exr"]["file_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="does not match"):
        final_output.write_final(final_output.select(m), m, masters, figures)


def test_hard_links_fall_back_to_copies(tmp_path, monkeypatch):
    def refuse(*args):
        raise OSError("hard links are not supported on this file system")

    monkeypatch.setattr(final_output.os, "link", refuse)
    m, masters, figures = written_run(tmp_path)
    record = final_output.write_final(final_output.select(m), m, masters, figures)
    assert {f["how"] for f in record["files"].values()} == {"copy"}


@pytest.fixture
def colab(monkeypatch):
    downloads = []
    files = types.SimpleNamespace(download=downloads.append)
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    return downloads


def delivered_run(tmp_path):
    m, masters, figures = written_run(tmp_path)
    m["final_output"] = final_output.write_final(final_output.select(m), m, masters, figures)
    return m


def test_delivery_downloads_the_tiff_and_copies_the_final_folder_to_drive(tmp_path, colab):
    m = delivered_run(tmp_path)
    drive = tmp_path / "drive"
    drive.mkdir()
    done = final_output.deliver(m, drive_dir=drive / "MyDrive" / "final", drive_root=drive, log=lambda _: None)
    tiff = m["final_output"]["files"]["tiff16"]["path"]
    assert colab == [tiff] and done["downloaded"] == tiff
    copied = drive / "MyDrive" / "final" / "run1"
    assert done["copied_to"] == str(copied)
    assert sorted(p.name for p in copied.iterdir()) == sorted(p.name for p in Path(tiff).parent.iterdir())


def test_delivery_skips_drive_when_unmounted_or_already_there(tmp_path, colab):
    m = delivered_run(tmp_path)
    logs = []
    done = final_output.deliver(
        m, download=False, drive_dir=tmp_path / "nowhere" / "final", drive_root=tmp_path / "nowhere", log=logs.append
    )
    assert done == {"downloaded": None, "copied_to": None} and "not mounted" in logs[0] and colab == []
    done = final_output.deliver(m, download=False, drive_dir=tmp_path / "x", drive_root=tmp_path, log=logs.append)
    assert done["copied_to"] == str(Path(m["final_output"]["files"]["tiff16"]["path"]).parent)
    assert "already on Drive" in logs[-1]


def test_delivery_outside_colab_prints_the_path(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    m = delivered_run(tmp_path)
    logs = []
    assert final_output.deliver(m, log=logs.append)["downloaded"] is None
    assert logs == [f"not in Colab: the final image is {m['final_output']['files']['tiff16']['path']}"]


def test_delivery_without_a_final_image_says_why(tmp_path):
    logs = []
    final_output.deliver({"final_output": {"method": None, "reason": "no output finished"}}, log=logs.append)
    assert logs == ["no final image: no output finished"]


def test_a_general_scene_whose_balanced_mode_fails_a_seam_gate_ends_with_the_conservative_mode():
    """The second Colab run: a night forest scene where the balanced mode failed the blend-error gate."""
    decision = final_output.select(manifest(balanced=summary(sr=True, seams=False)))
    assert decision["method"] == "sr_conservative"
    assert "passed over sr_balanced: it failed a seam gate" in decision["reason"]
