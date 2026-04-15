"""Tests for the `agentis` CLI (scaffolder + doctor)."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from agentis.cli import main


# ── `agentis new` ──────────────────────────────────────────


def test_new_creates_expected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = main(["new", "demo"])
    assert exit_code == 0

    project = tmp_path / "demo"
    for expected in ("main.py", ".env.example", "pyproject.toml", "README.md", ".gitignore"):
        assert (project / expected).exists(), f"missing {expected}"

    main_src = (project / "main.py").read_text()
    ast.parse(main_src)
    assert "AnthropicProvider.from_env()" in main_src

    env_example = (project / ".env.example").read_text()
    assert "ANTHROPIC_API_KEY" in env_example

    out = capsys.readouterr().out
    assert "Created" in out
    assert "cd demo" in out


def test_new_openai_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = main(["new", "demo_oai", "--provider", "openai"])
    assert exit_code == 0

    main_src = (tmp_path / "demo_oai" / "main.py").read_text()
    assert "OpenAIProvider.from_env()" in main_src

    env_example = (tmp_path / "demo_oai" / ".env.example").read_text()
    assert "OPENAI_API_KEY" in env_example


def test_new_rejects_existing_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "demo").mkdir()

    exit_code = main(["new", "demo"])
    assert exit_code == 1

    err = capsys.readouterr().err
    assert "already exists" in err


def test_new_rejects_invalid_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    exit_code = main(["new", "bad/name"])
    assert exit_code == 2
    assert "alphanumeric" in capsys.readouterr().err


# ── `agentis doctor` ──────────────────────────────────────


def test_doctor_reports_missing_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = main(["doctor"])
    out = capsys.readouterr().out

    assert "ANTHROPIC_API_KEY" in out
    assert "OPENAI_API_KEY" in out
    # With a provider SDK installed but no keys, doctor should fail.
    assert exit_code == 1
    assert "FAIL" in out


def test_doctor_passes_with_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = main(["doctor"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "PASS" in out
    # Never leak the actual key value.
    assert "sk-test" not in out


# ── version ──────────────────────────────────────────────


def test_version_flag_exits_cleanly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "agentis" in out
