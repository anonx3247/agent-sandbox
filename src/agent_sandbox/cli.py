"""Console entry point for the ``asb`` command."""

from __future__ import annotations

import typer

from agent_sandbox import __version__

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """agent-sandbox: a general srt-based sandbox wrapper around any coding agent."""


@app.command()
def version() -> None:
    """Print the agent-sandbox version."""
    typer.echo(__version__)
