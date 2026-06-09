"""Tests for the ``asb`` passthrough run command."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import agent_sandbox.cli as cli
from agent_sandbox.cli import app

runner = CliRunner()


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Replace ``run_sandboxed_binary`` with a capturing stub that exits 0."""
    record: dict[str, object] = {}

    def fake_run_sandboxed_binary(
        binary: str,
        profile: str,
        extra_args: list[str],
        env: dict[str, str] | None = None,
        extra_allow_read: tuple[str, ...] = (),
    ) -> None:
        record.update(
            binary=binary,
            profile=profile,
            args=list(extra_args),
            env=env,
            extra_allow_read=tuple(extra_allow_read),
        )
        raise typer.Exit(code=0)

    monkeypatch.setattr(cli, "run_sandboxed_binary", fake_run_sandboxed_binary)
    return record


# --- run passthrough -------------------------------------------------------


def test_run_passes_binary_args_and_profile(captured: dict[str, object]) -> None:
    result = runner.invoke(app, ["-p", "open", "--", "claude", "--resume"])

    assert result.exit_code == 0
    assert captured["binary"] == "claude"
    assert captured["args"] == ["--resume"]
    assert captured["profile"] == "open"
    assert captured["extra_allow_read"] == ()


def test_run_defaults_profile_to_git(captured: dict[str, object]) -> None:
    result = runner.invoke(app, ["--", "pi"])

    assert result.exit_code == 0
    assert captured["binary"] == "pi"
    assert captured["args"] == []
    assert captured["profile"] == "git"


def test_run_secrets_populates_extra_allow_read(captured: dict[str, object], tmp_path: Path) -> None:
    secrets = tmp_path / ".env"
    secrets.write_text("TOKEN=abc\n")

    result = runner.invoke(app, ["--secrets", str(secrets), "--", "codex", "--yolo"])

    assert result.exit_code == 0
    assert captured["binary"] == "codex"
    assert captured["args"] == ["--yolo"]
    assert captured["extra_allow_read"] == (str(secrets.resolve()),)


def test_run_secrets_missing_file_errors(captured: dict[str, object], tmp_path: Path) -> None:
    missing = tmp_path / "nope.env"

    result = runner.invoke(app, ["--secrets", str(missing), "--", "pi"])

    assert result.exit_code != 0
    assert captured == {}


def test_run_scrubs_llm_auth_vars(captured: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    # Force sandbox_run_env to leak an LLM auth var so the cli scrub is exercised.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setattr(
        cli,
        "sandbox_run_env",
        lambda available, include_srt_debug=True: {
            "PATH": os.environ.get("PATH", ""),
            "ANTHROPIC_API_KEY": "secret",
            "OPENAI_API_KEY": "leak",
        },
    )

    result = runner.invoke(app, ["--", "pi"])

    assert result.exit_code == 0
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert "ANTHROPIC_API_KEY" not in child_env
    assert "OPENAI_API_KEY" not in child_env
    assert child_env.get("PATH")


def test_run_no_command_exits_2(captured: dict[str, object]) -> None:
    result = runner.invoke(app, ["--"])

    assert result.exit_code == 2
    assert captured == {}


def test_run_no_command_at_all_exits_2(captured: dict[str, object]) -> None:
    result = runner.invoke(app, ["-p", "git"])

    assert result.exit_code == 2
    assert captured == {}


# --- version subcommand still works ---------------------------------------


def test_version_subcommand_still_works() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0


def test_help_advertises_passthrough_usage() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "-- <command...>" in result.output
