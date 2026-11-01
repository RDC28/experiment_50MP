import subprocess
from types import SimpleNamespace

import pytest

from prd_v5 import dng_validate as dv


@pytest.fixture
def tool(tmp_path):
    binary = tmp_path / "dng_validate"
    binary.write_bytes(b"#!/bin/sh\n")
    binary.chmod(0o755)
    return binary


def fake_run(returncode=0, stdout="", stderr=""):
    return lambda *a, **k: SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_missing_tool_is_not_measured(tmp_path, monkeypatch):
    monkeypatch.delenv(dv.ENV_VAR, raising=False)
    monkeypatch.delenv(dv.SKIP_ENV_VAR, raising=False)
    monkeypatch.setattr(dv.shutil, "which", lambda name: None)
    record = dv.run_dng_validate(tmp_path / "x.dng")
    assert record["status"] == "not measured" and dv.ENV_VAR in record["reason"]


def test_missing_tool_names_the_recorded_build_failure(tmp_path, monkeypatch):
    monkeypatch.delenv(dv.ENV_VAR, raising=False)
    monkeypatch.setenv(dv.SKIP_ENV_VAR, "dng_validate build failed (exit 2): SDK SHA-256 mismatch")
    monkeypatch.setattr(dv.shutil, "which", lambda name: None)
    record = dv.run_dng_validate(tmp_path / "x.dng")
    assert record == {"status": "not measured", "reason": "dng_validate build failed (exit 2): SDK SHA-256 mismatch"}


def test_clean_run_passes_and_records_tool_hash(tmp_path, tool, monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stdout="Validation OK\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    record = dv.run_dng_validate(tmp_path / "x.dng", tool)
    assert record["status"] == "passed" and len(record["tool_sha256"]) == 64
    assert calls == [[str(tool), "-v", str(tmp_path / "x.dng")]]


def test_error_output_or_exit_code_fails(tmp_path, tool, monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_run(0, "*** Error: bad IFD\n"))
    assert dv.run_dng_validate("x.dng", tool)["status"] == "failed"
    monkeypatch.setattr(subprocess, "run", fake_run(1, "ok"))
    assert dv.run_dng_validate("x.dng", tool)["status"] == "failed"


def test_env_variable_resolves_tool(tool, monkeypatch):
    monkeypatch.setenv(dv.ENV_VAR, str(tool))
    assert dv.find_dng_validate() == tool


def test_timeout_is_not_measured(tmp_path, tool, monkeypatch):
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("dng_validate", 1)

    monkeypatch.setattr(subprocess, "run", boom)
    assert dv.run_dng_validate("x.dng", tool, timeout_s=1)["status"] == "not measured"


def test_build_failure_is_visible_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", fake_run(2, "", "SDK SHA-256 mismatch"))
    record = dv.build_dng_validate(tmp_path)
    assert record["status"] == "not measured" and "mismatch" in record["reason"]


def test_build_success_returns_binary(tmp_path, monkeypatch):
    def ok(*a, **k):
        (tmp_path / "dng_validate").write_bytes(b"x")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", ok)
    record = dv.build_dng_validate(tmp_path)
    assert record["status"] == "built" and record["sdk_sha256"] == dv.SDK_SHA256
