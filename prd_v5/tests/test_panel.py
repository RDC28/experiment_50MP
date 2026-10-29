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
