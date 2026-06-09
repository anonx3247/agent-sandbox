"""Bootstrap everything ``asb`` needs: the ``srt`` sandbox runtime, the
``sx``/``sxd`` secret-broker, and a default ``security_profile.json``.

``asb install`` is a one-shot, idempotent setup command. It installs:

1. The ``srt`` sandbox-runtime from the Isara-Laboratories fork (via
   ``npm install -g`` from its GitHub release tarball) that backs every
   ``asb run`` invocation.
2. The ``sx``/``sxd`` secret-broker binaries (via ``cargo install --git``), the
   ``sxd`` login auto-start agent (macOS only), and the ``sx`` agent skill.
3. A default ``security_profile.json`` at the main repo root, written only when
   absent so user customisations are never clobbered.

Each step is warn-and-continue: a missing toolchain (npm, cargo) skips that
step with a manual-install hint rather than aborting the whole install, so the
command is safe to re-run after installing the missing prerequisite.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import typer

# Install srt from the Isara-Laboratories fork's GitHub release tarball rather
# than the public npm registry. v0.0.49 is the only fork tag that publishes an
# installable .tgz release asset (the package has no build/`prepare` step, so
# installing from a git ref is unreliable). The tarball still exposes the same
# `@anthropic-ai/sandbox-runtime` package and `srt` binary.
_SRT_PACKAGE_SPEC = "https://github.com/Isara-Laboratories/sandbox-runtime/releases/download/v0.0.49/anthropic-ai-sandbox-runtime-0.0.49.tgz"

# Git source for the `sx`/`sxd` secret-broker binaries, installed via
# `cargo install --git`. `sx` lets agents in short-lived sandboxes use host
# secrets on demand against a long-lived `sxd` daemon that lives outside the
# sandbox.
_SX_REPO_SPEC = "https://github.com/anonx3247/secrets"
_SX_BINARIES = ("sx", "sxd")

# Minimal default override: allow writes to uv's cache so `uv sync` succeeds
# under the `git` sandbox profile. Everything else is left to the base profile.
_DEFAULT_SECURITY_PROFILE = {"filesystem": {"allowWrite": ["~/.cache/uv"]}}


def _has(cmd: str) -> bool:
    """Return whether *cmd* resolves on PATH."""
    return shutil.which(cmd) is not None


def _run(cmd: list[str]) -> None:
    """Run *cmd*, raising ``CalledProcessError`` on a non-zero exit."""
    subprocess.run(cmd, check=True)


def _run_optional(cmd: list[str], *, ok: str, warn: str, hint: str) -> None:
    """Run *cmd*, warning and continuing (never raising) on failure."""
    try:
        _run(cmd)
    except subprocess.CalledProcessError as e:
        typer.secho(f"{warn}: {e}", fg=typer.colors.YELLOW, err=True)
        typer.echo(hint, err=True)
    else:
        typer.echo(ok)


def setup_srt() -> None:
    """Install the ``srt`` sandbox-runtime from the Isara-Laboratories fork.

    Runs ``npm install -g <release-tarball-url>`` against the fork's GitHub
    release tarball, which installs the same ``srt`` binary as the upstream
    ``@anthropic-ai/sandbox-runtime`` package.

    Skips with a warning if npm isn't available. On Linux, also warns about the
    bubblewrap / socat / ripgrep runtime dependencies that ``srt`` needs at
    invocation time; on macOS, warns about ripgrep.
    """
    typer.secho("Installing srt sandbox-runtime...", fg=typer.colors.CYAN)

    if not _has("npm"):
        typer.secho("npm not found, skipping srt installation", fg=typer.colors.YELLOW, err=True)
        typer.echo("To install srt, first install Node.js (includes npm) and re-run: asb install", err=True)
        return

    install_cmd = ["npm", "install", "-g", _SRT_PACKAGE_SPEC]
    try:
        _run(install_cmd)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Could not install srt: {e}", fg=typer.colors.YELLOW, err=True)
        typer.echo(f"You can install it manually with: {' '.join(install_cmd)}", err=True)
        return
    typer.echo("srt installed successfully")

    if platform.system() == "Linux" and not all(_has(b) for b in ("bwrap", "socat", "rg")):
        typer.secho(
            "srt on Linux needs bubblewrap, socat, and ripgrep at runtime. "
            "Install them via your package manager (e.g. `apt install bubblewrap socat ripgrep`).",
            fg=typer.colors.YELLOW,
            err=True,
        )
    elif platform.system() == "Darwin" and not _has("rg"):
        typer.secho(
            "srt on macOS needs ripgrep at runtime. Install it with `brew install ripgrep`.",
            fg=typer.colors.YELLOW,
            err=True,
        )


def setup_sx() -> None:
    """Install the ``sx``/``sxd`` secret-broker binaries, agent skill, and daemon.

    ``sx`` gives agents in short-lived sandboxes conditioned access to host
    secrets: a long-lived ``sxd`` daemon outside the sandbox holds the values and
    injects them into commands run via ``sx run``, gated by the user — the agent
    never reads the secrets directly. We install the binaries from the secrets
    repo via ``cargo install --git``, register ``sxd`` as a login auto-start
    agent (macOS only; the daemon must present TouchID prompts, so it is a
    per-user LaunchAgent), and install the usage skill for Claude Code / Codex / Pi.

    Skips with a warning when ``cargo`` is absent. The daemon and skill steps
    warn-and-continue: they rely on the secrets repo's ``sxd install`` /
    ``sx skill`` subcommands.
    """
    typer.secho("Installing sx/sxd secret-broker binaries...", fg=typer.colors.CYAN)

    if not _has("cargo"):
        typer.secho("cargo not found, skipping sx installation", fg=typer.colors.YELLOW, err=True)
        typer.echo("To install sx, first install Rust (https://rustup.rs) and re-run: asb install", err=True)
        return

    install_cmd = ["cargo", "install", "--git", _SX_REPO_SPEC, *_SX_BINARIES, "--force"]
    try:
        _run(install_cmd)
    except subprocess.CalledProcessError as e:
        typer.secho(f"Could not install sx/sxd: {e}", fg=typer.colors.YELLOW, err=True)
        typer.echo(f"You can install them manually with: {' '.join(install_cmd)}", err=True)
        return
    typer.echo("sx/sxd installed successfully")

    if platform.system() == "Darwin":
        _run_optional(
            ["sxd", "install"],
            ok="sxd registered as a login auto-start agent",
            warn="Could not register sxd auto-start agent",
            hint="You can register it manually with: sxd install",
        )
    else:
        typer.echo("sxd auto-start is macOS-only; start `sxd` manually on this platform")

    _run_optional(
        ["sx", "skill", "install"],
        ok="sx agent skill installed for Claude Code, Codex, and Pi",
        warn="Could not install sx agent skill",
        hint="You can install it manually with: sx skill install",
    )


def _find_main_repo_root() -> Path | None:
    """Return the main repository root, even when invoked from a linked worktree.

    The sandbox override walk-up begins at the main repo root (the parent of
    ``git rev-parse --git-common-dir``), so writing ``security_profile.json``
    there means every worktree picks it up. Returns None when cwd is not inside
    a git repo or the git call fails.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    return Path(raw).resolve().parent


def setup_security_profile() -> None:
    """Write a default ``security_profile.json`` at the main repo root if missing.

    The file acts as the sandbox override for any srt-wrapped command whose cwd
    is inside this repo. We never overwrite an existing file — users customise
    the profile locally, so clobbering it would destroy their edits.
    """
    typer.secho("Checking default security_profile.json...", fg=typer.colors.CYAN)

    repo_root = _find_main_repo_root()
    if repo_root is None:
        typer.secho(
            "Not inside a git repository, skipping security_profile.json setup",
            fg=typer.colors.YELLOW,
            err=True,
        )
        return

    profile_path = repo_root / "security_profile.json"
    if profile_path.exists():
        typer.echo(f"security_profile.json already exists at {profile_path}")
        return

    try:
        profile_path.write_text(json.dumps(_DEFAULT_SECURITY_PROFILE, indent=2) + "\n")
    except OSError as e:
        typer.secho(f"Could not create security_profile.json at {profile_path}: {e}", fg=typer.colors.YELLOW, err=True)
        typer.echo(f"You can create it manually with the contents: {json.dumps(_DEFAULT_SECURITY_PROFILE)}", err=True)
        return
    typer.echo(f"Created default security_profile.json at {profile_path}")


def install(
    force: bool = typer.Option(False, "--force", "-f", help="Force installation even if targets exist"),
) -> None:
    """Bootstrap the srt runtime, sx/sxd secret-broker, and a default security profile.

    Runs :func:`setup_srt`, :func:`setup_sx`, and :func:`setup_security_profile`
    in sequence. Each step warns and continues on a missing toolchain or
    failure, so the command is idempotent and safe to re-run.
    """
    if os.name != "posix":
        typer.secho("asb install is only supported on Unix-like systems.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    typer.secho("Starting asb install...", fg=typer.colors.GREEN, bold=True)
    setup_srt()
    setup_sx()
    setup_security_profile()
    typer.secho("asb install complete.", fg=typer.colors.GREEN, bold=True)
