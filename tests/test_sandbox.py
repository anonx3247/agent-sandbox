"""Unit tests for the standalone srt sandbox engine.

These tests exercise the pure profile-resolution logic and never require the
``srt`` binary on PATH — they call the resolution helpers directly or stub
``shutil.which`` where a sandbox-availability branch is involved.

Ported/adapted from Isara's ``cave`` suite
(``tests/small/test_sandbox_exec.py`` and
``tests/small/test_sandbox_profiles_integration.py``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agent_sandbox import sandbox


def _use_tmp_as_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point ``Path.home()`` and cwd at *tmp_path* so override walks stay sandboxed."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Profile catalogue
# ---------------------------------------------------------------------------


def test_list_profiles_returns_sorted_builtin_names() -> None:
    assert sandbox.list_profiles() == ["git", "locked", "open", "sealed"]


def test_mmo_profile_is_not_present() -> None:
    """The isara-specific ``mmo`` profile was intentionally dropped in the port."""
    assert "mmo" not in sandbox._BASE_PROFILES
    assert not hasattr(sandbox, "_MMO_PROFILE")


# ---------------------------------------------------------------------------
# resolve_profile — sentinel substitution
# ---------------------------------------------------------------------------


def test_resolve_git_substitutes_root_sentinels_to_real_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """resolve_profile('git') replaces the git root sentinels with real paths and
    leaves no sentinels behind anywhere in the filesystem section."""
    (tmp_path / ".git").mkdir()
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(nested)

    result = sandbox.resolve_profile("git")
    allow_write = result["filesystem"]["allowWrite"]

    assert sandbox._GIT_WORKTREE_ROOT_SENTINEL not in allow_write
    assert sandbox._GIT_REPO_ROOT_SENTINEL not in allow_write
    assert str(tmp_path.resolve()) in allow_write

    # No remaining sentinels of any kind across the whole filesystem section.
    sentinels = (
        sandbox._GIT_WORKTREE_ROOT_SENTINEL,
        sandbox._GIT_REPO_ROOT_SENTINEL,
        sandbox._JJ_WORKSPACE_ROOT_SENTINEL,
        sandbox._JJ_REPO_ROOT_SENTINEL,
        sandbox._JJ_GIT_BACKEND_SENTINEL,
        sandbox._CWD_SENTINEL,
    )
    for key in ("allowRead", "allowWrite", "denyRead", "denyWrite"):
        for entry in result["filesystem"].get(key, []):
            for sentinel in sentinels:
                assert sentinel not in entry, f"{sentinel!r} survived in {key}: {entry!r}"


def test_resolve_git_falls_back_to_cwd_without_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.chdir(isolated)
    monkeypatch.setattr(sandbox, "_find_git_root", lambda start=None: None)
    monkeypatch.setattr(sandbox, "_find_git_repo_root", lambda: None)

    result = sandbox.resolve_profile("git")
    assert str(isolated.resolve()) in result["filesystem"]["allowWrite"]


def test_resolve_locked_substitutes_cwd_sentinel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    result = sandbox.resolve_profile("locked")
    assert sandbox._CWD_SENTINEL not in result["filesystem"]["allowRead"]
    assert str(tmp_path.resolve()) in result["filesystem"]["allowRead"]


def test_resolve_drops_jj_sentinels_outside_jj_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sandbox, "_run_jj_path_command", lambda args, cwd=None: None)

    allow_write = sandbox.resolve_profile("git")["filesystem"]["allowWrite"]
    for sentinel in (
        sandbox._JJ_WORKSPACE_ROOT_SENTINEL,
        sandbox._JJ_REPO_ROOT_SENTINEL,
        sandbox._JJ_GIT_BACKEND_SENTINEL,
    ):
        assert all(sentinel not in entry for entry in allow_write)


def test_resolve_none_defaults_to_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    assert sandbox.resolve_profile(None) == sandbox.resolve_profile("git")


def test_resolve_expands_tilde_in_deny_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    result = sandbox.resolve_profile("locked")
    for entry in result["filesystem"]["denyRead"]:
        assert not entry.startswith("~"), f"unexpanded tilde in {entry!r}"


def test_resolve_unknown_name_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(sandbox.SandboxProfileNotFoundError, match=r"Unknown sandbox profile 'strict'"):
        sandbox.resolve_profile("strict")
    # Available names are listed so callers can self-correct.
    try:
        sandbox.resolve_profile("nope")
    except sandbox.SandboxProfileNotFoundError as exc:
        for name in ("git", "locked", "open", "sealed"):
            assert name in str(exc)


# ---------------------------------------------------------------------------
# .env.example carve-out (extra allowRead surviving a denyRead)
# ---------------------------------------------------------------------------


def test_dotenv_example_allow_survives_dotenv_deny(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The ``**/.env.example`` allowRead carve-out is forced into allowRead and
    survives resolution even though ``**/.env`` / ``**/.env.*`` stay denied.

    ``_apply_deny_wins`` is exact-match, so the carve-out never collides with
    the broad dotenv deny globs and re-opens ``.env.example`` specifically.
    """
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    fs = sandbox.resolve_profile("git")["filesystem"]
    assert "**/.env.example" in fs["allowRead"]
    assert "**/.env" in fs["denyRead"]
    assert "**/.env.*" in fs["denyRead"]


# ---------------------------------------------------------------------------
# Override walk-up — $HOME boundary
# ---------------------------------------------------------------------------


def test_find_override_file_at_repo_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_tmp_as_home(monkeypatch, tmp_path)
    override = tmp_path / "security_profile.json"
    override.write_text("{}")
    monkeypatch.setattr(sandbox, "_find_git_repo_root", lambda: tmp_path)
    result = sandbox._find_override_file()
    assert result is not None
    assert result.resolve() == override.resolve()


def test_find_override_file_none_when_absent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_tmp_as_home(monkeypatch, tmp_path)
    isolated = tmp_path / "no-override-here"
    isolated.mkdir()
    assert sandbox._find_override_file(start=isolated) is None


def test_find_override_file_stops_at_home_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The walk never crosses ``$HOME`` — a profile planted above home is ignored."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    # Plant a profile ABOVE home — the attack case.
    (tmp_path / "security_profile.json").write_text('{"network": {"allowedDomains": ["attacker.example.com"]}}')

    nested = fake_home / "project" / "src"
    nested.mkdir(parents=True)
    monkeypatch.setattr(sandbox, "_find_git_repo_root", lambda: nested)

    assert sandbox._find_override_file() is None


def test_find_override_file_includes_home_itself(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    home_override = fake_home / "security_profile.json"
    home_override.write_text("{}")
    nested = fake_home / "project" / "src"
    nested.mkdir(parents=True)
    monkeypatch.setattr(sandbox, "_find_git_repo_root", lambda: nested)
    result = sandbox._find_override_file()
    assert result is not None
    assert result.resolve() == home_override.resolve()


# ---------------------------------------------------------------------------
# Override schema validation
# ---------------------------------------------------------------------------


def test_validate_override_rejects_unknown_section() -> None:
    with pytest.raises(ValueError, match="Unknown override section"):
        sandbox._validate_override_schema({"netwrok": {"allowedDomains": ["x"]}})


def test_validate_override_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown override key"):
        sandbox._validate_override_schema({"network": {"allowdDomains": ["x"]}})


def test_validate_override_rejects_wrong_type_for_list_key() -> None:
    with pytest.raises(ValueError, match="must be list"):
        sandbox._validate_override_schema({"network": {"allowedDomains": "example.com"}})


def test_validate_override_rejects_wrong_type_for_bool_key() -> None:
    with pytest.raises(ValueError, match="must be bool"):
        sandbox._validate_override_schema({"network": {"allowAllDomains": 1}})


def test_validate_override_rejects_non_string_list_entry() -> None:
    with pytest.raises(ValueError, match=r"\[0\] must be str"):
        sandbox._validate_override_schema({"filesystem": {"allowWrite": [123]}})


# ---------------------------------------------------------------------------
# Merge + deny-wins
# ---------------------------------------------------------------------------


def test_merge_profile_unions_list_fields() -> None:
    merged = sandbox._merge_profile(
        {"network": {"allowedDomains": ["a.com"]}},
        {"network": {"allowedDomains": ["b.com"]}},
    )
    assert merged["network"]["allowedDomains"] == ["a.com", "b.com"]


def test_deny_wins_strips_allow_entry_that_is_also_denied() -> None:
    """An allow entry that also appears in the matching deny list is dropped."""
    settings = {
        "filesystem": {
            "allowRead": ["/data", "/secret"],
            "denyRead": ["/secret"],
            "allowWrite": ["/work", "/locked"],
            "denyWrite": ["/locked"],
        },
        "network": {"allowedDomains": ["ok.com", "bad.com"], "deniedDomains": ["bad.com"]},
    }
    result = sandbox._apply_deny_wins(settings)
    assert result["filesystem"]["allowRead"] == ["/data"]
    assert result["filesystem"]["allowWrite"] == ["/work"]
    assert result["network"]["allowedDomains"] == ["ok.com"]
    # Deny lists are preserved untouched.
    assert result["filesystem"]["denyRead"] == ["/secret"]


def test_deny_wins_normalises_trailing_slash() -> None:
    settings = {"filesystem": {"allowWrite": ["/a/"], "denyWrite": ["/a"]}}
    result = sandbox._apply_deny_wins(settings)
    assert result["filesystem"]["allowWrite"] == []


def test_resolve_override_deny_wins_over_base_allow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An override deny strips a base allow during full resolution."""
    _use_tmp_as_home(monkeypatch, tmp_path)
    (tmp_path / "security_profile.json").write_text(
        json.dumps({"network": {"deniedDomains": ["evil.com"], "allowedDomains": ["evil.com"]}})
    )
    monkeypatch.setattr(sandbox, "_find_git_repo_root", lambda: tmp_path)
    result = sandbox.resolve_profile("git")
    assert "evil.com" not in result["network"]["allowedDomains"]
    assert "evil.com" in result["network"]["deniedDomains"]


def test_resolve_override_malformed_json_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _use_tmp_as_home(monkeypatch, tmp_path)
    (tmp_path / "security_profile.json").write_text("{not valid json")
    monkeypatch.setattr(sandbox, "_find_git_repo_root", lambda: tmp_path)
    with pytest.raises(json.JSONDecodeError):
        sandbox.resolve_profile("git")


# ---------------------------------------------------------------------------
# Env scrubbing
# ---------------------------------------------------------------------------


def test_sandbox_run_env_scrubs_non_allowlisted_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("HOME", "/home/alice")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "SHOULD_NOT_LEAK")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA_SECRET")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_SHOULD_NOT_LEAK")
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")

    env = sandbox.sandbox_run_env(available=False)

    assert env["PATH"] == "/usr/bin:/bin"
    assert env["HOME"] == "/home/alice"
    assert env["LC_ALL"] == "en_US.UTF-8"  # LC_ prefix allowlisted
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "AWS_ACCESS_KEY_ID" not in env
    assert "GITHUB_TOKEN" not in env
    assert "SHOULD_NOT_LEAK" not in env.values()
    assert "ghp_SHOULD_NOT_LEAK" not in env.values()


def test_sandbox_run_env_forwards_anthropic_and_openai_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    env = sandbox.sandbox_run_env(available=False)
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-test"
    assert env["OPENAI_API_KEY"] == "sk-openai-test"


def test_sandbox_run_env_sets_srt_debug_only_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    assert sandbox.sandbox_run_env(available=True)["SRT_DEBUG"] == "1"
    assert "SRT_DEBUG" not in sandbox.sandbox_run_env(available=False)
    assert "SRT_DEBUG" not in sandbox.sandbox_run_env(available=True, include_srt_debug=False)


# ---------------------------------------------------------------------------
# wrap_command / is_sandbox_available
# ---------------------------------------------------------------------------


def test_is_sandbox_available_reflects_which(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/srt")
    assert sandbox.is_sandbox_available() is True
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert sandbox.is_sandbox_available() is False


def test_wrap_command_prefixes_srt_when_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/srt")
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    wrapped = sandbox.wrap_command(["git", "log"], str(settings))
    assert wrapped[:2] == ["srt", "--settings"]
    assert wrapped[-3:] == ["--", "git", "log"]


def test_wrap_command_passthrough_when_srt_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(shutil, "which", lambda _: None)
    settings = tmp_path / "settings.json"
    settings.write_text("{}")
    cmd = ["git", "log"]
    wrapped = sandbox.wrap_command(cmd, str(settings))
    assert wrapped == cmd
    assert wrapped is not cmd  # never mutates the caller's list


def test_wrap_command_raises_on_unresolved_profile_variable() -> None:
    with pytest.raises(sandbox.SandboxVariableError):
        sandbox.wrap_command(["echo", "hi"], "$DEFINITELY_UNSET_VAR/settings.json")
