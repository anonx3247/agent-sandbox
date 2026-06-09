"""agent-sandbox — a general srt-based sandbox wrapper around coding agents."""

from __future__ import annotations

from agent_sandbox.sandbox import (
    Sandboxed,
    SandboxedPopen,
    SandboxProfileNotFoundError,
    SandboxResult,
    SandboxUnavailableError,
    SandboxVariableError,
    SandboxViolation,
    is_sandbox_available,
    list_profiles,
    resolve_profile,
    sandbox_run_env,
    wrap_command,
)

__version__ = "0.1.0"

__all__ = [
    "Sandboxed",
    "SandboxedPopen",
    "SandboxResult",
    "SandboxViolation",
    "SandboxProfileNotFoundError",
    "SandboxUnavailableError",
    "SandboxVariableError",
    "is_sandbox_available",
    "sandbox_run_env",
    "resolve_profile",
    "list_profiles",
    "wrap_command",
    "__version__",
]
