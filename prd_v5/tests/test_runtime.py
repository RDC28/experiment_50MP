"""Runtime evidence follows loaded codecs and stays JSON portable."""

import json

import rawpy

from prd_v5.runtime import runtime_record


def test_native_versions_and_missing_optional_codec(monkeypatch):
    monkeypatch.setattr("prd_v5.runtime.features.version", lambda name: None)
    record = runtime_record()
    assert record["libraries"]["littlecms"] is None
    assert record["libraries"]["libraw"] == ".".join(map(str, rawpy.libraw_version))
    assert record["libraries"]["rawpy"] == rawpy.__version__
    assert "cross-runtime bitwise reproducibility not validated" in record["reproducibility"]
    assert json.loads(json.dumps(record, allow_nan=False)) == record
    assert set(record) == {"python", "platform", "libraries", "reproducibility"}
    assert set(record["platform"]) == {"system", "machine"}
    assert set(record["python"]) == {"implementation", "version"}


def test_runtime_records_are_detached():
    first = runtime_record()
    first["libraries"]["numpy"] = "altered"
    assert runtime_record()["libraries"]["numpy"] != "altered"


def test_installed_wheel_revision_is_explicit_and_constrained(monkeypatch):
    from prd_v5.runtime import software_identity

    revision = "a" * 40
    monkeypatch.setenv("SYNTHRAW_SOURCE_REVISION", revision)
    assert software_identity().endswith("git:" + revision)
    monkeypatch.setenv("SYNTHRAW_SOURCE_REVISION", "not-a-public-commit-identifier")
    assert "not-a-public-commit-identifier" not in software_identity()
