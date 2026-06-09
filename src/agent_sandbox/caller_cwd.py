"""Restore the user's pre-wrapper working directory."""

from __future__ import annotations

import os
from pathlib import Path

#: Env var an outer shell wrapper may export (``ASB_ORIGINAL_CWD=$PWD``) before
#: ``cd``\\ ing elsewhere so that ``uv run`` resolves the right workspace. The
#: ``asb`` command restores it so the sandboxed child inherits the user's real
#: working directory rather than wherever the wrapper landed.
ORIGINAL_CWD_ENV = "ASB_ORIGINAL_CWD"


def restore_caller_cwd() -> None:
    """Chdir back to the user's pre-wrapper PWD, if one was captured.

    No-ops when :data:`ORIGINAL_CWD_ENV` is unset (direct invocation bypassing
    any wrapper) or points at a path that no longer exists.
    """
    original = os.environ.get(ORIGINAL_CWD_ENV)
    if original and Path(original).is_dir():
        os.chdir(original)
