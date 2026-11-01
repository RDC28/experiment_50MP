"""Real tiny DNG fixtures with independently corrupted registry and file tags."""

from copy import deepcopy

import numpy as np
import pytest
import tifffile

from prd_v5.analytic import AnalyticISP
from prd_v5.coherence import metadata_coherence
from prd_v5.dng import write_dng
from prd_v5.geometry import GeometryPlanner
from prd_v5.hypothesis import CameraHypothesis


@pytest.fixture
def exported(tmp_path):
    path = tmp_path / "capture.dng"
    geometry = GeometryPlanner().plan(32, 24, target_megapixels=0.000768)
    tags = write_dng(
        path,
        np.full((24, 32), 600, dtype=np.uint16),
        CameraHypothesis(),
        geometry,
        AnalyticISP(white_balance=(2, 1, 1.5)),
    )["tags"]
    registry = {f"dng.{name}": {"value": record["value"]} for name, record in tags.items()}
    registry["AsShotNeutral"] = deepcopy(registry["dng.AsShotNeutral"])
    registry["ISO"] = deepcopy(registry["dng.PhotographicSensitivity"])
    registry["ColorMatrix1/2"] = {"value": [tags[f"ColorMatrix{i}"]["value"] for i in (1, 2)]}
    registry["Full-well capacity"] = {"value": 50000}
    return path, registry, tags


def test_serialized_rationals_and_explicit_unmapped_coverage(exported, monkeypatch):
    path, registry, tags = exported

    def no_raster(*args, **kwargs):
        pytest.fail("Metadata validation must not load a raster")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", no_raster)
    result = metadata_coherence(path, registry, tags)
    assert result["status"] == "passed"
    assert result["unmapped_registry_fields"] == ["Full-well capacity"]
    assert "AsShotNeutral" in result["checked"]
    assert "ColorMatrix1/2:ColorMatrix2" in result["checked"]


@pytest.mark.parametrize("key", ["dng.AsShotNeutral", "AsShotNeutral", "ISO", "ColorMatrix1/2"])
def test_registry_disagreement_is_visible(exported, key):
    path, registry, tags = exported
    registry[key]["value"] = [9, 8, 7]
    result = metadata_coherence(path, registry, tags)
    assert result["status"] == "failed"
    assert key in result["mismatches"]


def test_written_neutral_corruption_is_detected(exported):
    path, registry, tags = exported
    with tifffile.TiffFile(path, mode="r+") as container:
        container.pages[0].tags["AsShotNeutral"].overwrite((1, 1, 1, 1, 1, 1))
    result = metadata_coherence(path, registry, tags)
    assert result["status"] == "failed"
    assert {"manifest.AsShotNeutral", "dng.AsShotNeutral", "AsShotNeutral"} <= set(result["mismatches"])


def test_missing_registry_tag_cannot_pass(exported):
    path, registry, tags = exported
    del registry["dng.WhiteLevel"]
    result = metadata_coherence(path, registry, tags)
    assert "dng.WhiteLevel" in result["mismatches"]


def test_missing_file_tag_cannot_pass(exported):
    path, registry, tags = exported
    with tifffile.TiffFile(path, mode="r+") as container:
        tag = container.pages[0].tags["AsShotNeutral"]
        tag.code = 65000
        # overwrite changes the entry's tag code as well as its value.
        tag.overwrite((1, 1, 1, 1, 1, 1))
    result = metadata_coherence(path, registry, tags)
    assert result["status"] == "failed"
    assert "manifest.AsShotNeutral" in result["mismatches"]


def test_omitted_optional_tag_must_really_be_absent(exported):
    path, registry, tags = exported
    del tags["NoiseProfile"]
    registry["dng.NoiseProfile"]["value"] = None
    result = metadata_coherence(path, registry, tags)
    assert "dng.NoiseProfile" in result["mismatches"]


def test_absent_optional_tag_with_null_registry_is_valid(exported):
    path, registry, tags = exported
    registry["dng.FocalLength"] = {"value": None}
    result = metadata_coherence(path, registry, tags)
    assert result["status"] == "passed"
    assert "dng.FocalLength" in result["checked"]
