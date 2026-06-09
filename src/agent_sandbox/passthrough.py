"""Run a local binary wrapped in an srt sandbox profile."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

import typer

from agent_sandbox.sandbox import Sandboxed, SandboxProfileNotFoundError


def run_sandboxed_binary(
    binary: str,
    profile: str,
    extra_args: Sequence[str] | None,
    env: dict[str, str] | None = None,
    extra_allow_read: Sequence[str] = (),
) -> NoReturn:
    """Run ``binary [*extra_args]`` under srt with *profile*; inherit the parent TTY.

    Blocks until the child exits and raises :class:`typer.Exit` with the child's
    return code. ``SandboxProfileNotFoundError`` exits 2 with a clean message;
    a missing binary exits 127.

    ``env`` is forwarded to :meth:`agent_sandbox.sandbox.Sandboxed.popen`. When
    ``None`` (default) the sandbox uses its scrubbed allowlist; callers that need
    to inject extra vars pass an explicit dict — typically composed via
    :func:`agent_sandbox.sandbox.sandbox_run_env`.

    ``extra_allow_read`` is a sequence of filesystem paths opened for READ inside
    the sandbox (e.g. a ``--secrets`` file); it is threaded into the resolved
    profile by the engine.
    """
    args = list(extra_args) if extra_args else []
    sandboxed = Sandboxed(
        cmd=[binary, *args],
        profile=profile,
        extra_allow_read=tuple(extra_allow_read),
    )
    try:
        with sandboxed.popen(env=env) as proc:
            result = proc.wait()
    except SandboxProfileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2)
    except FileNotFoundError:
        typer.echo(f"error: `{binary}` not found on PATH", err=True)
        raise typer.Exit(code=127)
    raise typer.Exit(code=result.exit_code)
