"""Tiny executable catalogue selection and strict failure checks."""

import pytest

from prd_v5.hypothesis import CameraHypothesis
from prd_v5.panel import resolve_camera_panel


def test_panel_default_and_fabricate_keep_shared_hypothesis():
    base = CameraHypothesis()
    for mode in ("AUTO", "FABRICATE"):
        camera, records = resolve_camera_panel(base, [{"name": "ISO", "mode": "AUTO", "value": None}], {"ISO": mode})
        assert camera == base
        assert next(r for r in records if r.name == "iso").mode == mode


@pytest.mark.parametrize(
    "name,value,mode,error",
    [
        ("ISO", None, "ESTIMATE", "Unresolved"),
        ("ISO", None, "USER", "Unresolved"),
        ("ISO", 400, "AUTO", "requires USER"),
        ("ISO", -1, "USER", "ISO"),
        ("GPS", [1, 2], "USER", "Unsupported"),
        ("PRNU", None, "FABRICATE", "Unsupported"),
    ],
)
def test_panel_rejects_unresolved_invalid_and_unsupported_edits(name, value, mode, error):
    default = "USER" if name == "GPS" else "AUTO"
    with pytest.raises(ValueError, match=error):
        resolve_camera_panel(CameraHypothesis(), [{"name": name, "mode": default, "value": value}], {name: mode})


@pytest.mark.parametrize("neutral", [[1, 0, 1], [1, 2, 1], [1, 1], [float("nan"), 1, 1]])
def test_panel_rejects_invalid_neutral(neutral):
    from prd_v5.analytic import AnalyticISP
    from prd_v5.panel import resolve_analytic_panel

    with pytest.raises(ValueError, match="AsShotNeutral"):
        resolve_analytic_panel(
            CameraHypothesis(),
            AnalyticISP(),
            [{"name": "AsShotNeutral", "mode": "AUTO", "value": neutral}],
            {"AsShotNeutral": "USER"},
        )


@pytest.mark.parametrize("mode", ["USER", "ESTIMATE"])
def test_panel_neutral_requires_supplied_evidence(mode):
    from prd_v5.analytic import AnalyticISP
    from prd_v5.panel import resolve_analytic_panel

    with pytest.raises(ValueError, match="Unresolved ISP"):
        resolve_analytic_panel(
            CameraHypothesis(),
            AnalyticISP(),
            [{"name": "AsShotNeutral", "mode": "AUTO", "value": None}],
            {"AsShotNeutral": mode},
        )


def test_panel_neutral_changes_paired_raw_and_preserves_metadata(tmp_path):
    import json

    import numpy as np
    import rawpy
    from PIL import Image

    from prd_v5.analytic import AnalyticISP
    from prd_v5.config import AnalyticRunConfig
    from prd_v5.paired import run_analytic_pair
    from prd_v5.panel import resolve_analytic_panel

    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(source)
    base = CameraHypothesis(shot_noise=False, read_noise=False)
    camera, isp, camera_records, isp_records = resolve_analytic_panel(
        base,
        AnalyticISP(),
        [{"name": "AsShotNeutral", "mode": "AUTO", "value": [0.5, 1, 0.25]}],
        {"AsShotNeutral": "USER"},
    )
    assert camera == base and isp.white_balance == (2, 1, 4)
    config = AnalyticRunConfig(camera=camera, isp=isp, target_megapixels=0.001024, assume_unprofiled_srgb=True)
    run_analytic_pair(
        source,
        source,
        tmp_path / "selected",
        config,
        camera_selections=camera_records,
        isp_selections=isp_records,
    )
    AnalyticRunConfig(camera=base, target_megapixels=0.001024, assume_unprofiled_srgb=True).run(
        source, tmp_path / "baseline"
    )
    with rawpy.imread(str(tmp_path / "baseline/capture.dng")) as raw:
        baseline = raw.raw_image.copy().astype(float) - base.black_level
    for label in ("2k", "4k"):
        manifest = json.loads((tmp_path / "selected" / label / "manifest.json").read_text())
        assert manifest["registry"]["isp.white_balance"]["provenance"] == "USER_SET"
        assert manifest["registry"]["isp.white_balance"]["mode"] == "USER"
        with rawpy.imread(str(tmp_path / "selected" / label / "capture.dng")) as raw:
            signal = raw.raw_image.copy().astype(float) - base.black_level
            # RGGB: reciprocal neutral halves red and quarters blue electron signal.
            np.testing.assert_allclose(signal[0::2, 0::2], baseline[0::2, 0::2] / 2, atol=1)
            np.testing.assert_allclose(signal[1::2, 1::2], baseline[1::2, 1::2] / 4, atol=1)
            np.testing.assert_array_equal(signal[0::2, 1::2], baseline[0::2, 1::2])
            np.testing.assert_allclose(raw.camera_whitebalance[:3], [2, 1, 4])


@pytest.mark.parametrize("mode", ["AUTO", "FABRICATE"])
def test_panel_neutral_default_keeps_explicit_isp(mode):
    from prd_v5.analytic import AnalyticISP
    from prd_v5.panel import resolve_analytic_panel

    base = AnalyticISP(white_balance=(2, 1.5, 4))
    _, isp, _, records = resolve_analytic_panel(
        CameraHypothesis(),
        base,
        [{"name": "AsShotNeutral", "mode": "AUTO", "value": None}],
        {"AsShotNeutral": mode},
    )
    assert isp == base
    assert next(r for r in records if r.name == "white_balance").mode == mode


def test_panel_neutral_rejects_value_without_user_mode():
    from prd_v5.analytic import AnalyticISP
    from prd_v5.panel import resolve_analytic_panel

    with pytest.raises(ValueError, match="requires USER"):
        resolve_analytic_panel(
            CameraHypothesis(),
            AnalyticISP(),
            [{"name": "AsShotNeutral", "mode": "AUTO", "value": [1, 1, 1]}],
            {"AsShotNeutral": "AUTO"},
        )


def test_sr_panel_verified_selection_and_fail_closed(tmp_path):
    import hashlib

    from prd_v5.panel import resolve_sr_panel

    path = tmp_path / "weights.pth"
    path.write_bytes(b"model double, never loaded")
    checkpoint = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "backend": "HAT",
        "repository_revision": "a" * 40,
        "source": "user test fixture",
        "licence": "test only",
        "licence_accepted": True,
    }
    spec, state = resolve_sr_panel("HAT", checkpoint, allow_baseline=False)
    assert spec.path == str(path)
    assert state["planned"] == "HAT" and state["effective"] is None
    assert state["checkpoint"]["integrity"] == "verified"
    with pytest.raises(ValueError, match="differs"):
        resolve_sr_panel("REAL_ESRGAN", checkpoint, allow_baseline=True)
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolve_sr_panel("HAT", checkpoint, allow_baseline=True)


def test_sr_panel_missing_weights_require_explicit_baseline():
    from prd_v5.panel import resolve_sr_panel

    with pytest.raises(ValueError, match="opt-in"):
        resolve_sr_panel("HAT", None, allow_baseline=False)
    spec, state = resolve_sr_panel("HAT", None, allow_baseline=True)
    assert spec is None
    assert state["requested"] == "HAT" and state["planned"] == "BILINEAR_BASELINE"
    assert state["effective"] is None and state["skip_reason"]
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_sr_panel("unknown", None, allow_baseline=True)
    with pytest.raises(ValueError, match="cannot accept"):
        resolve_sr_panel("BILINEAR_BASELINE", {}, allow_baseline=True)
