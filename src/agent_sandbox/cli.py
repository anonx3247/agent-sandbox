"""Console entry point for the ``asb`` command.

``asb`` runs an arbitrary command inside an srt sandbox profile::

    asb [--profile/-p PROFILE] [--secrets FILE] -- <command...>

Everything after ``--`` is the command line executed inside the sandbox; the
first token is the binary and the rest are its arguments. A ``version``
subcommand prints the package version.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from typer.core import TyperGroup

from agent_sandbox import __version__
from agent_sandbox.aws import build_aws_env, mint_profile_creds
from agent_sandbox.install import install
from agent_sandbox.passthrough import run_sandboxed_binary
from agent_sandbox.sandbox import (
    is_sandbox_available,
    list_profiles,
    resolve_profile,
    sandbox_run_env,
)

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

    def parse_args(self, ctx, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] not in self.get_help_option_names(ctx):
            args = [_DEFAULT_COMMAND, *args]
        return super().parse_args(ctx, args)

    def collect_usage_pieces(self, ctx) -> list[str]:
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


app.command(name="install")(install)


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
    aws_profile: str | None = typer.Option(
        None,
        "--aws-profile",
        help="Mint short-lived STS creds from this SSO profile and overlay them as AWS_* env vars.",
    ),
    aws_region: str = typer.Option(
        "us-west-2",
        "--aws-region",
        help="AWS region for the injected AWS_REGION/AWS_DEFAULT_REGION env vars.",
    ),
    command: list[str] = typer.Argument(
        None,
        metavar="-- <command...>",
        help="Command to run inside the sandbox, e.g. `-- claude --resume`.",
    ),
) -> None:
    """Run a command inside the srt sandbox: ``asb [flags] -- <command...>``."""
    if not command:
        typer.echo("error: no command given. Usage: asb [flags] -- <command...>", err=True)
        raise typer.Exit(code=2)

    # include_srt_debug=False: interactive TTY runs inherit stderr, so srt's
    # debug chatter would otherwise garble fullscreen child apps.
    child_env = dict(sandbox_run_env(is_sandbox_available(), include_srt_debug=False))
    for leaked in _LLM_AUTH_ENV_VARS:
        child_env.pop(leaked, None)

    # Opt back into AWS access by minting fresh STS creds and overlaying them as
    # env vars. The sandbox still denies ~/.aws, so the child can never refresh.
    if aws_profile:
        child_env.update(build_aws_env(mint_profile_creds(aws_profile), region=aws_region))

    extra_allow_read = (str(secrets),) if secrets else ()

    # Emit sandbox-identity vars so tools running inside the sandbox can detect
    # they're sandboxed, which profile is active, and what it grants. child_env
    # is passed explicitly to the sandbox, so the allowlist (which only filters
    # inheritance from os.environ) does not strip these.
    child_env["ASB_SANDBOX"] = "1"
    child_env["ASB_PROFILE"] = profile if profile in list_profiles() else "custom"
    child_env["ASB_PROFILE_JSON"] = json.dumps(
        resolve_profile(profile, extra_allow_read), separators=(",", ":")
    )
    if secrets:
        child_env["ASB_SECRETS_FILE"] = str(secrets)
    if aws_profile:
        child_env["ASB_AWS_PROFILE"] = aws_profile

    binary, *args = command
    run_sandboxed_binary(
        binary,
        profile,
        args,
        env=child_env,
        extra_allow_read=extra_allow_read,
    )
