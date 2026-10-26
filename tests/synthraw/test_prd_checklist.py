"""Guard docs/prd_v3_checklist.md against drifting from the PRD_v3 checklist and toggle table."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATES = ("Tested locally", "Pending Colab", "Gated stub", "Not implemented")


def _prd_items() -> tuple[list[str], list[str]]:
    prd = (ROOT / "PRD_v3.md").read_text(encoding="utf-8")
    checklist = prd.split("final Claude Code delivery checklist is:", 1)[1]
    items = re.findall(r"^- \[ \] (.+)$", checklist.split("\n\n", 2)[1], flags=re.MULTILINE)
    table = prd.split("acceptance contract for key toggles is:", 1)[1].split("\n\n", 2)[1]
    toggles = re.findall(r"^\| (`[^|]+`) \|", table, flags=re.MULTILINE)
    return items, toggles


def _doc_rows() -> dict[str, str]:
    doc = (ROOT / "docs/prd_v3_checklist.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| ([^|]+) \| ([^|]+) \|", doc, flags=re.MULTILINE)
    return {name.strip(): state.strip() for name, state in rows}


def test_every_checklist_item_and_toggle_has_a_state():
    items, toggles = _prd_items()
    assert len(items) == 45 and len(toggles) == 17
    rows = _doc_rows()
    for name in items + toggles:
        assert name in rows, name
        assert rows[name].startswith(STATES), (name, rows[name])
    assert "\u2014" not in (ROOT / "docs/prd_v3_checklist.md").read_text(encoding="utf-8")


def test_experimental_items_are_gated_stubs():
    rows = _doc_rows()
    for name in ("Optional SUPIR with licence warning", "Optional ExpandDiff backend", "Optional RAW-Flow/SpiralDiff"):
        assert rows[name] == "Gated stub"
    assert rows["`GENERATIVE_DETAIL_STRENGTH`"] == "Not implemented"
