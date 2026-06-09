"""Console entry point for the ``asb`` command.

``asb`` runs an arbitrary command inside an srt sandbox profile::

    asb [--profile/-p PROFILE] [--secrets FILE] -- <command...>

Everything after ``--`` is the command line executed inside the sandbox; the
first token is the binary and the rest are its arguments. A ``version``
subcommand prints the package version.
"""

from __future__ import annotations

from pathlib import Path

import typer
from typer.core import TyperGroup

from agent_sandbox import __version__
from agent_sandbox.caller_cwd import restore_caller_cwd
from agent_sandbox.passthrough import run_sandboxed_binary
from agent_sandbox.sandbox import is_sandbox_available, sandbox_run_env

# LLM auth vars the sandbox allowlist would otherwise forward from the parent
# shell. ``asb`` is a generic wrapper with no credential story of its own, so
# scrub them before launching the child rather than leaking the user's shell
# creds into an arbitrary sandboxed command.
_LLM_AUTH_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
)

_DEFAULT_COMMAND = "run"


class DefaultCommandGroup(TyperGroup):
    """Typer group whose default subcommand is :data:`_DEFAULT_COMMAND`.

    Tokens that are not a known subcommand (the flags and ``-- <command...>``
    of a passthrough invocation) are routed to ``run`` so ``asb -p git -- pi``
    works without naming the command, while ``asb version`` still resolves the
    explicit subcommand.
    """

    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:
        if (
            args
            and args[0] not in self.commands
            and args[0] not in self.get_help_option_names(ctx)
        ):
            args = [_DEFAULT_COMMAND, *args]
        return super().parse_args(ctx, args)

    def collect_usage_pieces(self, ctx: typer.Context) -> list[str]:
        return ["[OPTIONS]", "--", "<command...>"]


app = typer.Typer(
    cls=DefaultCommandGroup,
    no_args_is_help=True,
    add_completion=False,
    help=(
        "agent-sandbox: run any command inside an srt sandbox.\n\n"
        "Usage: asb [--profile/-p PROFILE] [--secrets FILE] -- <command...>"
    ),
)


@app.command()
def version() -> None:
    """Print the agent-sandbox version."""
    typer.echo(__version__)


@app.command(context_settings={"allow_interspersed_args": False})
def run(
    profile: str = typer.Option(
        "git",
        "--profile",
        "-p",
        help="Sandbox profile: git (default), open, sealed, locked, or path to settings JSON.",
    ),
    secrets: Path | None = typer.Option(
        None,
        "--secrets",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="File to open for READ inside the sandbox (values are NOT injected into env).",
    ),
    command: list[str] = typer.Argument(
        None,
        metavar="-- <command...>",
        help="Command to run inside the sandbox, e.g. `-- claude --resume`.",
    ),
) -> None:
    """Run a command inside the srt sandbox: ``asb [flags] -- <command...>``."""
    # An outer shell wrapper may have `cd`'d elsewhere so `uv run` resolves the
    # workspace; restore the user's real cwd so the child inherits it.
    restore_caller_cwd()

    if not command:
        typer.echo(
            "error: no command given. Usage: asb [flags] -- <command...>", err=True
        )
        raise typer.Exit(code=2)

    # include_srt_debug=False: interactive TTY runs inherit stderr, so srt's
    # debug chatter would otherwise garble fullscreen child apps.
    child_env = dict(sandbox_run_env(is_sandbox_available(), include_srt_debug=False))
    for leaked in _LLM_AUTH_ENV_VARS:
        child_env.pop(leaked, None)

    extra_allow_read = (str(secrets),) if secrets else ()
    binary, *args = command
    run_sandboxed_binary(
        binary,
        profile,
        args,
        env=child_env,
        extra_allow_read=extra_allow_read,
    )
