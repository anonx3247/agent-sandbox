"""Tests for the ``asb`` passthrough run command."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import agent_sandbox.cli as cli
from agent_sandbox.aws import AwsRuntimeCreds
from agent_sandbox.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip(text: str) -> str:
    """Strip ANSI styling and remove all whitespace.

    Typer renders ``--help`` through Rich, which boxes output and wraps it to
    the terminal width — splitting the usage synopsis across lines (or even
    mid-token at tiny widths) and weaving in ANSI escapes. Removing whitespace
    entirely rejoins any wrapped token, so synopsis assertions hold at any CI
    terminal width regardless of whether ``COLUMNS`` is honored.
    """
    return "".join(_ANSI_RE.sub("", text).split())


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


# --- ASB_* sandbox-identity env vars --------------------------------------


def test_run_emits_sandbox_identity_vars(captured: dict[str, object]) -> None:
    result = runner.invoke(app, ["-p", "git", "--", "true"])

    assert result.exit_code == 0
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env["ASB_SANDBOX"] == "1"
    assert child_env["ASB_PROFILE"] == "git"
    assert isinstance(json.loads(child_env["ASB_PROFILE_JSON"]), dict)
    # Compact JSON: no whitespace after separators.
    assert " " not in child_env["ASB_PROFILE_JSON"]


def test_run_secrets_sets_asb_secrets_file(captured: dict[str, object], tmp_path: Path) -> None:
    secrets = tmp_path / ".env"
    secrets.write_text("TOKEN=abc\n")

    result = runner.invoke(app, ["--secrets", str(secrets), "--", "pi"])

    assert result.exit_code == 0
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env["ASB_SECRETS_FILE"] == str(secrets.resolve())


def test_run_without_secrets_omits_asb_secrets_file(captured: dict[str, object]) -> None:
    result = runner.invoke(app, ["--", "pi"])

    assert result.exit_code == 0
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert "ASB_SECRETS_FILE" not in child_env


def test_run_aws_profile_sets_asb_aws_profile(captured: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "mint_profile_creds", _fake_creds)

    result = runner.invoke(app, ["--aws-profile", "dev", "--", "pi"])

    assert result.exit_code == 0
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env["ASB_AWS_PROFILE"] == "dev"


def test_run_without_aws_profile_omits_asb_aws_profile(captured: dict[str, object]) -> None:
    result = runner.invoke(app, ["--", "pi"])

    assert result.exit_code == 0
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert "ASB_AWS_PROFILE" not in child_env


def test_run_custom_profile_path_marks_profile_custom(
    captured: dict[str, object], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A ``-p <path>`` profile is not a built-in name, so it classifies as
    # "custom". resolve_profile only understands base names, so stub it here to
    # isolate the classification logic under test.
    monkeypatch.setattr(cli, "resolve_profile", lambda profile, extra_allow_read: {"custom": True})
    custom = tmp_path / "my-profile.json"
    custom.write_text("{}\n")

    result = runner.invoke(app, ["-p", str(custom), "--", "pi"])

    assert result.exit_code == 0
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env["ASB_PROFILE"] == "custom"


# --- --aws-profile injection ----------------------------------------------


def _fake_creds(*_args: object, **_kwargs: object) -> AwsRuntimeCreds:
    return AwsRuntimeCreds(access_key_id="AKIA", secret_access_key="secret", session_token="token")


def test_run_aws_profile_overlays_aws_env(captured: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "mint_profile_creds", _fake_creds)

    result = runner.invoke(app, ["--aws-profile", "dev", "--", "pi"])

    assert result.exit_code == 0
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env["AWS_ACCESS_KEY_ID"] == "AKIA"
    assert child_env["AWS_SECRET_ACCESS_KEY"] == "secret"
    assert child_env["AWS_SESSION_TOKEN"] == "token"
    assert child_env["AWS_DEFAULT_REGION"] == child_env["AWS_REGION"] == "us-west-2"


def test_run_aws_profile_honours_region(captured: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "mint_profile_creds", _fake_creds)

    result = runner.invoke(app, ["--aws-profile", "dev", "--aws-region", "eu-central-1", "--", "pi"])

    assert result.exit_code == 0
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env["AWS_DEFAULT_REGION"] == child_env["AWS_REGION"] == "eu-central-1"


def test_run_without_aws_profile_omits_aws_env(captured: dict[str, object]) -> None:
    result = runner.invoke(app, ["--", "pi"])

    assert result.exit_code == 0
    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert not any(key.startswith("AWS_") for key in child_env)


# --- version subcommand still works ---------------------------------------


def test_version_subcommand_still_works() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0


def test_help_advertises_passthrough_usage() -> None:
    # The passthrough synopsis lives on plain (non-boxed) help lines, so removing
    # all whitespace rejoins any Rich wrapping into a stable ``--<command...>``
    # token — no dependency on COLUMNS being honored by the CI terminal.
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--<command...>" in _strip(result.output)
