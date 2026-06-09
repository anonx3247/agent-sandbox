"""Tests for the ``asb install`` bootstrap command.

All external commands (npm, cargo, git, sxd, sx) are mocked — these tests never
touch the network or a real toolchain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent_sandbox import install as install_mod
from agent_sandbox.cli import app
from agent_sandbox.install import (
    _DEFAULT_SECURITY_PROFILE,
    _SRT_PACKAGE_SPEC,
    _SX_BINARIES,
    _SX_REPO_SPEC,
    setup_security_profile,
    setup_srt,
    setup_sx,
)

runner = CliRunner()


@pytest.fixture
def record_run(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Patch ``_run`` to record argv instead of executing anything."""
    calls: list[list[str]] = []
    monkeypatch.setattr(install_mod, "_run", lambda cmd: calls.append(cmd))
    return calls


def _only_has(monkeypatch: pytest.MonkeyPatch, present: set[str]) -> None:
    """Patch ``_has`` so only commands in *present* resolve on PATH."""
    monkeypatch.setattr(install_mod, "_has", lambda cmd: cmd in present)


# --- setup_srt -------------------------------------------------------------


def test_setup_srt_skips_without_npm(monkeypatch: pytest.MonkeyPatch, record_run: list[list[str]]) -> None:
    _only_has(monkeypatch, present=set())

    setup_srt()

    assert record_run == []


def test_setup_srt_runs_npm_install(monkeypatch: pytest.MonkeyPatch, record_run: list[list[str]]) -> None:
    _only_has(monkeypatch, present={"npm", "rg", "bwrap", "socat"})
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Linux")

    setup_srt()

    assert record_run == [["npm", "install", "-g", _SRT_PACKAGE_SPEC]]


def test_setup_srt_warns_linux_runtime_deps(monkeypatch: pytest.MonkeyPatch, record_run: list[list[str]]) -> None:
    _only_has(monkeypatch, present={"npm"})
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Linux")
    warnings: list[str] = []
    monkeypatch.setattr(install_mod.typer, "secho", lambda msg, **kw: warnings.append(msg))

    setup_srt()

    assert record_run == [["npm", "install", "-g", _SRT_PACKAGE_SPEC]]
    assert any("bubblewrap" in w for w in warnings)


def test_setup_srt_warns_darwin_ripgrep(monkeypatch: pytest.MonkeyPatch, record_run: list[list[str]]) -> None:
    _only_has(monkeypatch, present={"npm"})
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Darwin")
    warnings: list[str] = []
    monkeypatch.setattr(install_mod.typer, "secho", lambda msg, **kw: warnings.append(msg))

    setup_srt()

    assert any("ripgrep" in w for w in warnings)


def test_setup_srt_warns_on_install_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    _only_has(monkeypatch, present={"npm"})

    def _boom(cmd: list[str]) -> None:
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(install_mod, "_run", _boom)

    # Should not raise — failure is caught, warned, and returns.
    setup_srt()


# --- setup_sx --------------------------------------------------------------


def test_setup_sx_skips_without_cargo(monkeypatch: pytest.MonkeyPatch, record_run: list[list[str]]) -> None:
    _only_has(monkeypatch, present=set())

    setup_sx()

    assert record_run == []


def test_setup_sx_runs_cargo_install_darwin(monkeypatch: pytest.MonkeyPatch, record_run: list[list[str]]) -> None:
    _only_has(monkeypatch, present={"cargo"})
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Darwin")

    setup_sx()

    assert record_run == [
        ["cargo", "install", "--git", _SX_REPO_SPEC, *_SX_BINARIES, "--force"],
        ["sxd", "install"],
        ["sx", "skill", "install"],
    ]


def test_setup_sx_skips_autostart_on_linux(monkeypatch: pytest.MonkeyPatch, record_run: list[list[str]]) -> None:
    _only_has(monkeypatch, present={"cargo"})
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Linux")

    setup_sx()

    # No `sxd install` on Linux — auto-start is macOS-only.
    assert record_run == [
        ["cargo", "install", "--git", _SX_REPO_SPEC, *_SX_BINARIES, "--force"],
        ["sx", "skill", "install"],
    ]


def test_setup_sx_continues_when_skill_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    _only_has(monkeypatch, present={"cargo"})
    monkeypatch.setattr(install_mod.platform, "system", lambda: "Linux")

    def _fail_skill(cmd: list[str]) -> None:
        if cmd[:2] == ["sx", "skill"]:
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(install_mod, "_run", _fail_skill)

    # Skill failure is caught and does not propagate.
    setup_sx()


# --- setup_security_profile ------------------------------------------------


def test_setup_security_profile_writes_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(install_mod, "_find_main_repo_root", lambda: tmp_path)

    setup_security_profile()

    profile_path = tmp_path / "security_profile.json"
    assert profile_path.exists()
    assert json.loads(profile_path.read_text()) == _DEFAULT_SECURITY_PROFILE
    assert profile_path.read_text().endswith("\n")


def test_setup_security_profile_noop_when_exists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(install_mod, "_find_main_repo_root", lambda: tmp_path)
    profile_path = tmp_path / "security_profile.json"
    custom = '{"custom": true}\n'
    profile_path.write_text(custom)

    setup_security_profile()

    assert profile_path.read_text() == custom


def test_setup_security_profile_warns_outside_git_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(install_mod, "_find_main_repo_root", lambda: None)
    warnings: list[str] = []
    monkeypatch.setattr(install_mod.typer, "secho", lambda msg, **kw: warnings.append(msg))

    setup_security_profile()

    assert any("git repository" in w for w in warnings)


# --- CLI -------------------------------------------------------------------


def test_install_help_lists_command() -> None:
    result = runner.invoke(app, ["install", "--help"])

    assert result.exit_code == 0
    assert "force" in result.stdout.lower()


def test_install_listed_in_root_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "install" in result.stdout
    assert "run" in result.stdout


def test_run_is_still_default_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare ``-- <command>`` invocation still routes to ``run``, not ``install``."""
    captured: dict[str, object] = {}

    def _fake_run_sandboxed(binary: str, profile: str, args: list[str], **kw: object) -> None:
        captured["binary"] = binary
        captured["args"] = args

    monkeypatch.setattr("agent_sandbox.cli.run_sandboxed_binary", _fake_run_sandboxed)
    monkeypatch.setattr("agent_sandbox.cli.sandbox_run_env", lambda *a, **k: {})
    monkeypatch.setattr("agent_sandbox.cli.is_sandbox_available", lambda: True)

    result = runner.invoke(app, ["--", "echo", "hi"])

    assert result.exit_code == 0
    assert captured == {"binary": "echo", "args": ["hi"]}
