"""Sandbox execution primitive backed by the `srt` CLI (Anthropic sandbox-runtime).

This module provides a cross-platform (macOS + Linux) sandbox abstraction.
`srt` handles platform-specific enforcement transparently:
  - macOS: Apple Seatbelt (via srt internals)
  - Linux: bubblewrap (via srt internals)

Two entrypoints, one abstraction:

:meth:`Sandboxed.run` — one-shot capture (for commands that finish quickly
and whose stdout/stderr fit in memory)::

    result = Sandboxed(cmd=["git", "log"], profile="git").run()

:meth:`Sandboxed.popen` — long-lived / streaming / interactive (for agents
that need to forward stdout line-by-line, or TTY-inherit child stdio)::

    with Sandboxed(cmd=["claude", "-p", "..."], profile="git").popen(
        stdout=subprocess.PIPE, text=True, capture_stderr=True,
    ) as proc:
        for line in proc.stdout:
            ship_to_slack(line)
        result = proc.wait()  # SandboxResult with parsed violations

``profile`` is the name of one of the built-in base profiles (``"locked"``,
``"sealed"``, ``"git"``, ``"open"``) — **not** a path to a settings file.
:class:`Sandboxed` materialises the resolved profile to a per-invocation
tempfile internally.  Omit it (or pass ``None``) to fall back to the default
``"git"`` profile.

When ``srt`` is not on PATH the command runs unsandboxed (passthrough mode)
and the returned :class:`SandboxResult` carries the child's exit code / output
unchanged.  Callers that require strict enforcement
should pass ``strict=True`` — :meth:`Sandboxed.run` then raises
:class:`SandboxUnavailableError` instead of silently passing through.
:func:`is_sandbox_available` is also available for callers that prefer to
branch themselves.

Environment isolation — the child process does NOT inherit ``os.environ``
wholesale.  ``Sandboxed`` replaces env inheritance with a tight allowlist
(``PATH``, ``HOME``, ``LANG``, ``LC_*``, ``USER``, ``LOGNAME``, ``TERM``,
terminal color vars, editor vars, ``SHELL``, ``TMPDIR``, ``TZ``) so parent
credentials like ``AWS_*``, ``GITHUB_TOKEN``, ``ANTHROPIC_API_KEY``, or any
``.env``-sourced secret can't leak into the sandboxed child.  The filesystem
profile already denies the on-disk counterparts under ``_SECRETS_DENY``; this
closes the matching env-var channel.  Callers that legitimately need to pass
additional vars must add them back explicitly.

This module performs no logging of its own — it never writes to stdout/stderr
or any logger.  Everything the caller needs is returned in the
:class:`SandboxResult` (exit code, cleaned child stdout/stderr, and a
structured list of parsed violations) so the caller decides what, if anything,
to surface.

Violation detection is best-effort.  We parse ``[SandboxDebug]`` lines emitted
by srt (when ``SRT_DEBUG=1`` is set) for network blocks and fall back to EPERM
patterns and exit code 134 for filesystem/SIGABRT signals.  A result with an
empty ``violations`` tuple does NOT guarantee the child was clean — srt's
internal violation store is not accessible from this process.

Profile schema (srt semantics, often surprising — calling them out so callers
reading a profile dict can reason about it):

* ``denyRead`` = paths blocked for reads.  ``allowRead`` is NOT a whitelist;
  it carves out re-allows WITHIN denied regions, and Seatbelt's last-match-wins
  means a broad allowRead over a specific denyRead silently reopens it.  Keep
  ``allowRead`` empty unless you're explicitly carving out a sub-region.
* ``allowWrite`` IS allow-only — writes are denied unless explicitly listed.
  ``denyWrite`` takes precedence within allowed regions.
* ``allowGitConfig: bool`` lives inside the ``filesystem`` section (not the
  profile root).  srt always denies writes to ``.git/config`` unless this flag
  is set.  Top-level placement is silently dropped by srt's zod parser.
* ``.git/hooks/`` is **always** denied by srt — not configurable.  ``git
  init`` and ``git clone`` fail out-of-the-box inside the sandbox because they
  seed sample hook files.  Workaround: pass ``--template=<empty-dir>`` (or
  set ``GIT_TEMPLATE_DIR=<empty-dir>``) to skip hook templating.  Once the
  repo exists, all further git ops work cleanly under the ``git`` / ``open``
  profiles.
* Jujutsu workspaces use ``.jj/working_copy`` under the workspace but may point
  ``.jj/repo`` at a shared repo store outside the workspace.  The ``git``
  profile grants writes to the current jj workspace root, the default
  workspace's ``.jj/repo`` store, and the backing Git repository reported by
  ``jj git root``.
* Unrooted globs (``id_rsa*``, ``*.pem``) in deny lists are no-ops — srt
  passes them literally to Seatbelt's regex matcher which doesn't anchor to
  a real directory.  Use absolute subpaths (``~/.ssh`` covers every key
  inside) or explicit full paths.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from types import TracebackType
from typing import IO, Any

_UNRESOLVED_RE = re.compile(r"[$~]")
_SRT_DEBUG_LINE_RE = re.compile(r"^\[SandboxDebug\].*$", re.MULTILINE)
_NETWORK_BLOCK_RE = re.compile(r"Connection blocked to\s+(?P<host>[^\s:]+)(?::(?P<port>\d+))?")
_FS_EPERM_RE = re.compile(r"(?i)(permission denied|operation not permitted|eperm)")
_FS_DENY_RE = re.compile(r"fs_deny:\s+\S+\(\d+\)\s+deny\(\d+\)\s+(?P<operation>file-\S+)(?:\s+(?P<target>\S+))?")


# Secret paths denied for reads and writes.  These are absolute subpath entries
# (`~/.ssh`, not `~/.ssh/`) because srt on macOS uses Seatbelt `subpath` matchers
# which recurse automatically — trailing slashes or unrooted globs like
# `id_rsa*` are silently no-ops.  Covering the containing directory (`~/.ssh`)
# blocks every private key regardless of filename.
_GIT_WORKTREE_ROOT_SENTINEL: str = "__GIT_WORKTREE_ROOT__"
_GIT_REPO_ROOT_SENTINEL: str = "__GIT_REPO_ROOT__"
_JJ_WORKSPACE_ROOT_SENTINEL: str = "__JJ_WORKSPACE_ROOT__"
_JJ_REPO_ROOT_SENTINEL: str = "__JJ_REPO_ROOT__"
_JJ_GIT_BACKEND_SENTINEL: str = "__JJ_GIT_BACKEND__"
_CWD_SENTINEL: str = "__CWD__"
_DEFAULT_PROFILE_NAME: str = "git"
_OVERRIDE_FILENAME: str = "security_profile.json"

_SECRETS_DENY: tuple[str, ...] = (
    "~/.ssh",
    "~/.aws",
    "~/.gnupg",
    # ~/.config/gh is intentionally absent: on macOS gh stores credentials in
    # the system keychain, not on disk, so ~/.config/gh contains only
    # non-secret configuration (hostname, git protocol preference, etc.).
    "~/.netrc",
    "~/.git-credentials",
    "~/.npmrc",
    "~/.pypirc",
    "~/.cargo/credentials.toml",
    "~/.docker/config.json",
)

# Project-local dotenv files anywhere in the tree, denied for READS.  Unlike the
# home-dir entries in `_SECRETS_DENY` (absolute subpaths), these are `**/`-rooted
# globs: srt expands `**/` natively on macOS — the same proven mechanism as
# `_WRITE_ONLY_DENY`'s `**/security_profile.json` — so a `.env` in cwd or any
# subdir is blocked regardless of where claude is launched.  Without these the
# env-var scrubbing in `_sandbox_env` only stops `.env` *values* that were
# sourced into the parent shell; the file on disk stayed fully readable.  The
# broad `**/.env.*` covers every secret-bearing variant (`.env.local`,
# `.env.staging`, plus any novel name) so nothing exfiltratable is ever
# readable.  Example/template variants (`.env.example`, `.env.local.example`,
# `.env.example.prod`, `givetrack.env.example`, ...) are caught here too, but
# `_DOTENV_EXAMPLE_ALLOW` carves every `example`-bearing name back out of the
# read-deny (the conventional non-secret templates).  `.envrc` (direnv) is
# DELIBERATELY not matched by either glob and is therefore never denied: srt's
# `globToRegex` escapes the literal dot and anchors with `$`, so `**/.env`
# (`^(.*/)?\.env$`) needs the name to END at `.env` and `**/.env.*`
# (`^(.*/)?\.env\.[^/]*$`) needs a literal dot after `.env` — `.envrc` (the `rc`
# runs straight on, no dot) satisfies neither, so it stays fully readable AND
# writable (it is also absent from every deny list; see
# `test_envrc_never_denied`).  On Linux these are literals and no-op,
# consistent with `_WRITE_ONLY_DENY`.
_DOTENV_READ_DENY: tuple[str, ...] = ("**/.env", "**/.env.*")

# Project-local dotenv files denied for WRITES.  This is an ENUMERATION, not the
# broad `**/.env.*` used for reads, and the asymmetry is deliberate.  srt's
# `generateWriteRules()` emits allow rules FIRST and deny rules AFTER, and
# Seatbelt is last-match-wins — so a write-deny CANNOT be re-allowed by a later
# `allowWrite`/`allowRead` carve-out (a `**/.env.example` write-allow would be a
# silent no-op, beaten by a following `**/.env.*` deny).  Reads are the mirror
# image: srt emits read-denies first and read-allows last, which is why
# `_DOTENV_EXAMPLE_ALLOW` works for reads but has no write analogue.  To keep
# `.env.example` WRITABLE we therefore rely on the repo-tree `allowWrite` grant
# and simply OMIT `.env.example` from this list — every standard secret-bearing
# dotenv name is enumerated and stays write-denied, while `.env.example` (and
# any non-enumerated novel name) falls through to the tree grant and is
# writable.  Tradeoff: this write-deny is a BLOCKLIST that fails OPEN for a
# brand-new name like `.env.somethingnovel` (writable) — acceptable because
# `_DOTENV_READ_DENY` stays broad, so such a file still can't be read back.
# Names below are the common framework conventions; NEVER add `.env.example`.
_DOTENV_WRITE_DENY: tuple[str, ...] = (
    "**/.env",
    "**/.env.local",
    "**/.env.development",
    "**/.env.development.local",
    "**/.env.production",
    "**/.env.production.local",
    "**/.env.staging",
    "**/.env.staging.local",
    "**/.env.test",
    "**/.env.test.local",
)

# READS carved back out of the broad `_DOTENV_READ_DENY`.  Example/template
# dotenv files are the conventional non-secret templates — placeholder keys with
# no real credentials, committed alongside the code — so blocking them only
# frustrates an agent trying to learn which config vars a project expects.  The
# rule of thumb: if `example` appears in the dotenv filename it is a template,
# not a secret.  Rather than one broad `**/*example*` wildcard (which would also
# re-open unrelated denied files such as `~/.ssh/example_key` — srt's `*` is
# `[^/]*`, crossing dots freely), we ENUMERATE a small set of `.env`-anchored
# globs that provably cover every real-world shape while never matching a
# non-dotenv secret:
#   `**/.env.example`     → `.env.example`
#   `**/.env.*.example`   → `.env.local.example`, `web/.env.<env>.example`
#   `**/.env.example.*`   → `.env.example.local`, `.env.example.prod`
#   `**/*.env.example`    → `givetrack.env.example` (name does not start `.env`)
# Each glob is added to `allowRead` in every broad-read profile; srt emits
# `allowRead` rules AFTER `denyRead` (`getFsReadConfig` maps allowRead →
# `allowWithinDeny`, emitted last), so Seatbelt's last-match-wins reopens these
# templates specifically while `.env`, `.env.local`, `.env.production`, etc. stay
# denied (none of them contain `example`).  None of these globs is the textual
# twin of any `_DOTENV_READ_DENY`/`_DOTENV_WRITE_DENY` entry, so `_apply_deny_wins`
# (exact-string match) never drops them and they survive into the final profile.
# This is a READ-only carve-out: there is NO write analogue, because srt emits
# write-denies last (last-match-wins would beat any write-allow) — these
# templates stay writable via the repo-tree `allowWrite` grant plus their
# omission from `_DOTENV_WRITE_DENY` (no enumerated deny glob matches an
# `example` name), not via any entry here.  `locked` is deliberately excluded:
# its reads are already confined to cwd (broad `~` denyRead plus a `__CWD__`
# re-allow), so an in-cwd template is readable and a global re-allow would punch
# a hole in that confinement.  On Linux these are literals and no-op, consistent
# with the deny globs.
_DOTENV_EXAMPLE_ALLOW: tuple[str, ...] = (
    "**/.env.example",
    "**/.env.*.example",
    "**/.env.example.*",
    "**/*.env.example",
)

# Paths denied for writes only (reads are permitted).  `**/security_profile.json`
# is write-denied to prevent a compromised agent from injecting a malicious
# override that the next session would load via `_find_override_file`.  Reads
# are allowed so agents can inspect the active profile.  srt handles `**/`
# natively on macOS; on Linux the pattern is treated as a literal (acceptable —
# Linux is not the primary target).
_WRITE_ONLY_DENY: tuple[str, ...] = (f"**/{_OVERRIDE_FILENAME}",)

# Explicit schema for what an override's ``security_profile.json`` may contain.
# Enforced at :func:`_merge_profile` time so a typo (``allowdDomains``, misspelt
# ``netwrok``) fails loudly with a ValueError instead of silently no-op-ing —
# srt's own zod parser drops unknown top-level keys without warning, so the
# profile layer above it is where we catch operator mistakes.  Extending the
# sandbox with a new knob means adding it here in the same commit.
#
# Shape: ``{section: {key: expected_type}}``.  ``list`` in the type column is
# shorthand for "list of str" — ``_validate_override_schema`` verifies the
# element type separately.
_PROFILE_SCHEMA: dict[str, dict[str, type]] = {
    "network": {
        "allowedDomains": list,
        "deniedDomains": list,
        "allowAllDomains": bool,
        "allowMachLookup": list,
    },
    "filesystem": {
        "allowRead": list,
        "allowWrite": list,
        "denyRead": list,
        "denyWrite": list,
        "allowGitConfig": bool,
    },
}

_LIST_UNION_KEYS: frozenset[tuple[str, str]] = frozenset(
    {
        ("network", "allowedDomains"),
        ("network", "deniedDomains"),
        ("network", "allowMachLookup"),
        ("filesystem", "allowRead"),
        ("filesystem", "allowWrite"),
        ("filesystem", "denyRead"),
        ("filesystem", "denyWrite"),
    }
)

_GLOB_CHARS: frozenset[str] = frozenset({"*", "?", "["})

# Per-user state paths for the AI CLI agents we expect to run inside the
# sandbox most often (Claude Code, Codex, pi).  Covers the common dotfile
# location each tool uses by default plus the XDG-style variants and the
# top-level state files that sit next to those dirs — claude writes project
# settings / auth / MCP config to
# ``~/.claude.json`` (with a sibling ``.backup``), which isn't covered by
# a ``~/.claude`` subpath entry.  Without ``~/.claude.json`` in
# ``allowWrite``, ``claude`` hangs indefinitely at TUI startup under the
# ``git`` profile because the config-persist call blocks silently.  pi keeps
# its agent state (sessions, global instructions, config) under ``~/.pi``
# (e.g. ``~/.pi/agent/sessions/``), so the whole subtree is granted for the
# same session-persistence reason.
# ``sealed`` and ``open`` already permit writes to all of ``~`` so these
# are redundant there — we only wire them into the git-like profiles, whose
# tight ``allowWrite`` (git root + /tmp) would otherwise block agent session
# persistence and crash the CLIs on turn N+1.  ``locked`` stays strict by
# design.
_AGENT_STATE_PATHS: tuple[str, ...] = (
    "~/.claude",
    "~/.claude.json",
    "~/.claude.json.backup",
    "~/.codex",
    "~/.codex.json",
    "~/.codex.json.backup",
    "~/.pi",
    "~/.config/claude",
    "~/.config/codex",
    "~/.config/pi",
    "~/.local/state/claude",
    "~/.local/state/codex",
)

_EDITOR_STATE_PATHS: tuple[str, ...] = ("~/.local/state/nvim",)

# pi-lens's per-user LSP / diagnostic cache.  Sandboxed agents must be able to
# both READ and WRITE this directory, so per an explicit operator request it is
# wired into the allowWrite list of EVERY base profile — including ``locked``,
# where it deliberately punches a documented hole in the otherwise strict
# "reads/writes confined to cwd" confinement (see ``_LOCKED_PROFILE`` below for
# the matching allowRead carve-out).
_PI_LENS_STATE_PATHS: tuple[str, ...] = ("~/.pi-lens",)

# Shared per-user cache root for tools used during edit/test/commit loops.
# Keep this separate from agent state so we do not need one-off cache grants
# for uv, pre-commit, nvim, and similar tools.
_SHARED_CACHE_PATHS: tuple[str, ...] = ("~/.cache",)

# macOS XPC/Mach services that the user's clipboard tooling needs to look up.
# ``com.apple.pasteboard.1`` is the modern pboard mach service backing
# ``pbpaste`` / ``pbcopy`` and any AppKit/Carbon clipboard call — without
# permission to look it up the kernel returns ``mach-lookup`` denials before
# the call ever reaches the daemon, so the agent sees the clipboard as
# permanently empty (and ``pbpaste`` exits non-zero).  Granting the lookup
# enables both reads and writes via the same daemon — there is no read-only
# clipboard ACL on macOS — but the surface area is the user's own pasteboard
# (no privilege escalation), and interactive callers paste into the agent
# more often than the agent leaks back out.  Linux is unaffected: srt's
# allowMachLookup is macOS-only and silently no-ops elsewhere.
_CLIPBOARD_MACH_SERVICES: tuple[str, ...] = ("com.apple.pasteboard.1",)

# macOS Keychain mach service. Claude Code (bundled keytar) and Codex
# (``keyring-rs``) both persist OAuth tokens for their HTTP MCP servers in
# the user's login keychain — Codex via the keychain entry "Codex MCP
# Credentials", Claude Code via "Claude Code-credentials". Without
# permission to look up ``com.apple.SecurityServer`` the sandboxed binary
# can't reach securityd, every keyring call returns an error before it hits
# the daemon, and every HTTP MCP that authenticates with cached OAuth tokens
# fails to start. Granting the lookup widens the agent's reach to talk to
# securityd, but the keychain's own per-item ACLs (each entry stores the
# code-signed identities allowed to read it) remain the gatekeeper for
# which secrets it can actually read; entries scoped to other apps (gh,
# 1Password, browser passwords) stay denied because their ACL doesn't
# include the calling binary. Linux is unaffected — keyring-rs uses the
# secret service / D-Bus there, not Mach.
_KEYCHAIN_MACH_SERVICES: tuple[str, ...] = ("com.apple.SecurityServer",)

# srt filesystem read semantics ("deny-then-allow-back"):
#   - denyRead lists paths that are blocked
#   - allowRead carves OUT of those denies (narrow re-allow, NOT a whitelist of
#     the only paths readable) — later Seatbelt rules win, so a broad allowRead
#     like ["~"] over a specific denyRead like ["~/.ssh"] reopens the whole home
#     and the deny is effectively a no-op.
# We therefore keep allowRead narrow: empty in the broad-read profiles save for
# the `_DOTENV_EXAMPLE_ALLOW` carve-out (a deliberate sub-region re-open of the
# `example`-bearing dotenv templates out of the `**/.env.*` deny), and in
# `locked` we deny the
# entire $HOME subtree and re-allow cwd via __CWD__ for a genuine "reads confined
# to cwd for user data" model.  System paths (/etc, /usr, /bin) stay readable
# because the kernel needs them to exec binaries.

_LOCKED_PROFILE: dict = {
    # Pseudo-terminal access is required for interactive TUI callers
    # (interactive TUI agents such as claude or codex) — without it the child's
    # ``tcsetattr`` / ``ioctl`` on the inherited ``/dev/ttys*`` returns
    # EPERM, Node's ``setRawMode`` throws, and claude / codex hang at
    # startup before drawing a single pixel of UI.  The privilege is narrow
    # (grants ``file-ioctl`` on ``/dev/ptmx`` and ``/dev/ttys*`` plus
    # ``pseudo-tty``); headless callers that don't touch a TTY see no
    # change, so we enable it in every built-in profile.
    "allowPty": True,
    "network": {"allowedDomains": [], "deniedDomains": []},
    "filesystem": {
        # ``_PI_LENS_STATE_PATHS`` is a deliberate exception to locked's
        # "reads confined to cwd" model: it is carved back OUT of the broad
        # ``denyRead: ["~"]`` so pi-lens's per-user cache stays readable.
        "allowRead": [_CWD_SENTINEL, *_PI_LENS_STATE_PATHS],
        "allowWrite": [*_PI_LENS_STATE_PATHS],
        "denyRead": ["~", *_SECRETS_DENY, *_DOTENV_READ_DENY],
        "denyWrite": [*_SECRETS_DENY, *_DOTENV_WRITE_DENY, *_WRITE_ONLY_DENY],
    },
}

_SEALED_PROFILE: dict = {
    "allowPty": True,
    "network": {
        "allowedDomains": [],
        "deniedDomains": [],
        "allowMachLookup": [*_CLIPBOARD_MACH_SERVICES, *_KEYCHAIN_MACH_SERVICES],
    },
    "filesystem": {
        "allowGitConfig": True,
        "allowRead": [*_DOTENV_EXAMPLE_ALLOW],
        "allowWrite": [".", "~", "/tmp", *_PI_LENS_STATE_PATHS],
        "denyRead": [*_SECRETS_DENY, *_DOTENV_READ_DENY],
        "denyWrite": [*_SECRETS_DENY, *_DOTENV_WRITE_DENY, *_WRITE_ONLY_DENY],
    },
}

_GIT_PROFILE: dict = {
    "allowPty": True,
    "network": {
        "allowedDomains": [],
        "deniedDomains": [],
        "allowAllDomains": True,
        "allowMachLookup": [*_CLIPBOARD_MACH_SERVICES, *_KEYCHAIN_MACH_SERVICES],
    },
    "filesystem": {
        # allowGitConfig belongs INSIDE `filesystem` per srt's
        # FilesystemConfigSchema — srt silently drops unknown top-level keys.
        # It lifts srt's hardcoded mandatory deny for the named path; without
        # it the deny rule is appended after all allowWrite entries and wins
        # (Seatbelt last-match-wins).
        "allowGitConfig": True,
        "allowRead": [*_DOTENV_EXAMPLE_ALLOW],
        # Write scope = current repo + /tmp + the per-user agent/editor state
        # paths and shared cache root.  Without these paths Claude Code /
        # Codex can't persist their session store and common tools can't write
        # caches; with them, the full edit-commit loop works cleanly while
        # arbitrary writes elsewhere in $HOME stay denied.
        "allowWrite": [
            _GIT_WORKTREE_ROOT_SENTINEL,
            _GIT_REPO_ROOT_SENTINEL,
            _JJ_WORKSPACE_ROOT_SENTINEL,
            _JJ_REPO_ROOT_SENTINEL,
            _JJ_GIT_BACKEND_SENTINEL,
            "/tmp",
            *_AGENT_STATE_PATHS,
            *_EDITOR_STATE_PATHS,
            *_SHARED_CACHE_PATHS,
            *_PI_LENS_STATE_PATHS,
            "~/.config/gh",
            "~/.config/graphite",
            "~/.local/share/graphite",
        ],
        "denyRead": [*_SECRETS_DENY, *_DOTENV_READ_DENY],
        "denyWrite": [*_SECRETS_DENY, *_DOTENV_WRITE_DENY, *_WRITE_ONLY_DENY],
    },
}

_OPEN_PROFILE: dict = {
    "allowPty": True,
    "network": {
        "allowedDomains": [],
        "deniedDomains": [],
        "allowAllDomains": True,
        "allowMachLookup": [*_CLIPBOARD_MACH_SERVICES, *_KEYCHAIN_MACH_SERVICES],
    },
    "filesystem": {
        # open already allows writes to cwd and ~; keeping allowGitConfig
        # enabled makes git commit / git config --local work consistently with
        # sealed and git-like profiles.
        "allowGitConfig": True,
        "allowRead": [*_DOTENV_EXAMPLE_ALLOW],
        "allowWrite": [".", "~", "/tmp", *_PI_LENS_STATE_PATHS],
        "denyRead": [*_SECRETS_DENY, *_DOTENV_READ_DENY],
        "denyWrite": [*_SECRETS_DENY, *_DOTENV_WRITE_DENY, *_WRITE_ONLY_DENY],
    },
}

_BASE_PROFILES: dict[str, dict] = {
    "locked": _LOCKED_PROFILE,
    "sealed": _SEALED_PROFILE,
    "git": _GIT_PROFILE,
    "open": _OPEN_PROFILE,
}


class SandboxVariableError(Exception):
    """Raised when a settings path still contains unresolved ``$VAR`` or ``~`` tokens.

    This happens when the caller passes a raw path string such as
    ``"$UNDEFINED/settings.json"`` and the variable is not set in ``os.environ``,
    or when ``expanduser`` leaves a ``~`` in place (e.g. ``~nonexistent``).
    """


class SandboxProfileNotFoundError(Exception):
    """Raised when a sandbox profile name is not one of the known base profiles.

    The error message lists the available profile names so callers can correct
    their input without consulting documentation.
    """


class SandboxUnavailableError(Exception):
    """Raised from :meth:`Sandboxed.run` when ``strict=True`` and ``srt`` is
    not present on ``PATH``.

    ``Sandboxed`` advertises a hard enforcement boundary — callers who opted
    into strict mode would rather fail loudly than have their command run
    unsandboxed.  The non-strict default still passes through with a warning
    (see :class:`Sandboxed` docstring for the rationale).
    """


@dataclasses.dataclass(frozen=True)
class SandboxViolation:
    """A single parsed sandbox violation.

    Attributes:
        kind: Category — one of ``network_block``, ``fs_eperm``, ``sigabrt``.
        detail: Short, NON-sensitive description of the trigger.  Populated
            for ``sigabrt`` (fixed message) and ``network_block`` (the raw
            ``[SandboxDebug]`` line, produced by srt — not the child).
            Empty for ``fs_eperm`` — the original snippet came from child
            stderr and could carry secrets, so we drop it deliberately; use
            :attr:`SandboxResult.stderr` if the caller really needs to
            forensically inspect it under their own trust assumptions.
        target: Destination identifier when applicable — ``host:port`` for
            ``network_block``, path for ``fs_deny``, empty for ``fs_eperm``
            and ``sigabrt``.
    """

    kind: str
    detail: str = ""
    target: str = ""


@dataclasses.dataclass(frozen=True)
class SandboxConfig:
    """Resolved configuration for a sandboxed invocation.

    Attributes:
        settings_path: Absolute, canonicalised path to the srt settings JSON
            file (produced by :func:`expand_settings_path`).
    """

    settings_path: Path


@dataclasses.dataclass(frozen=True)
class SandboxResult:
    """Captured result of a sandboxed (or passthrough) subprocess invocation.

    Attributes:
        exit_code: The exit code returned by the child process (srt passes it
            through unchanged).
        stdout: Captured standard output from the child process.
        stderr: Captured standard error from the child process, with srt's
            ``[SandboxDebug]`` lines stripped.  Callers receive only the
            child's own error output.
        sandbox_violations: Convenience bool — ``True`` when
            :attr:`violations` is non-empty.
        violations: Structured tuple of parsed violations.  An empty tuple
            does NOT guarantee clean execution — srt's internal violation
            store is not accessible.
    """

    exit_code: int
    stdout: str
    stderr: str
    sandbox_violations: bool
    violations: tuple[SandboxViolation, ...] = ()


def is_sandbox_available() -> bool:
    """Return ``True`` when ``srt`` is present on ``PATH``, ``False`` otherwise.

    Never raises.  No platform gate — srt supports macOS and Linux transparently.
    """
    return shutil.which("srt") is not None


def expand_settings_path(raw: str) -> Path:
    """Expand ``~`` and ``$VAR`` tokens in *raw* and return a resolved :class:`Path`.

    Raises:
        SandboxVariableError: If any ``$`` or ``~`` characters remain after
            expansion (i.e. the variable was undefined or ``expanduser`` was a
            no-op for an unknown user).
    """
    expanded = os.path.expanduser(raw)
    expanded = os.path.expandvars(expanded)
    if _UNRESOLVED_RE.search(expanded):
        raise SandboxVariableError(
            f"Unresolved variable in settings path after expansion: {expanded!r}. "
            "Ensure all $VAR references are defined in the environment and '~' "
            "can be resolved to a home directory."
        )
    return Path(expanded).resolve()


_CHILD_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Executable resolution and filesystem basics
        "PATH",
        "HOME",
        "TMPDIR",
        # Identity — git and other tools read these to populate author info
        # and cache key-derivation material
        "USER",
        "LOGNAME",
        # Locale — dropping these makes many tools emit 'locale not set'
        # warnings on stderr and can cause Unicode paths to fail
        "LANG",
        # Terminal — required for TUI tools (claude, codex) to render
        "TERM",
        "COLORTERM",
        "TERM_PROGRAM",
        "TERM_PROGRAM_VERSION",
        "CLICOLOR",
        "CLICOLOR_FORCE",
        "FORCE_COLOR",
        "NO_COLOR",
        # Interactive editor selection for TUI commands that shell out
        "EDITOR",
        "VISUAL",
        "SHELL",
        # Time — tools formatting timestamps need this
        "TZ",
        # API keys are forwarded so inference can authenticate, anthropic technically has a flag:
        # CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1 to pass these to claude without claude having access
        # to them in its tools but this disables --dangerously-skip-permissions so isn't done here.
        # OpenAI has no equivalent of this flag for codex
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        # LiteLLM proxy credentials — forwarded so agents pointed at a LiteLLM
        # gateway can authenticate and resolve the proxy base URL.
        "LITELLM_API_KEY",
        "LITELLM_BASE_URL",
        # Sandbox marker forwarded to claude-hud via the --extra-cmd helper.
        "CLAUDE_HUD_SANDBOX",
    }
)
_CHILD_ENV_ALLOWLIST_PREFIXES: tuple[str, ...] = ("LC_",)


def _minimal_child_env(srt_available: bool, *, include_srt_debug: bool = True) -> dict[str, str]:
    """Return a child-process env with only safe, tool-critical parent vars.

    We intentionally do NOT inherit ``os.environ`` wholesale into sandboxed
    children.  The parent process routinely carries credentials the child has
    no business seeing: ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` /
    ``AWS_SESSION_TOKEN`` from the dev's AWS CLI session, ``GITHUB_TOKEN`` /
    ``GH_TOKEN`` from ``gh auth``, ``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``
    from LLM experimentation, arbitrary ``*_TOKEN`` / ``*_SECRET`` / ``*_KEY``
    from project ``.env`` files that got sourced into the shell.  The
    filesystem profile already denies the on-disk counterparts (``~/.aws``,
    ``~/.config/gh``, etc. in ``_SECRETS_DENY``); this closes the matching
    env-var channel.

    We forward only a tight allowlist of vars that tools actually need to
    function — ``PATH`` so the child can resolve subcommands, ``HOME`` so
    ``git`` / ``claude`` / ``codex`` find their config (inside the
    filesystem restrictions), ``LANG`` / ``LC_*`` so locale-sensitive output
    doesn't emit warnings, terminal color vars so TUIs render correctly,
    editor vars so prompt editors can launch, and a handful of identity vars
    that aren't secret.  Anything outside that set is dropped.

    ``SRT_DEBUG=1`` is added when ``srt`` is on PATH so srt emits its
    ``[SandboxDebug]`` violation lines to stderr where ``_parse_violations``
    can pick them up.  Passthrough mode (srt absent) gets no SRT_DEBUG.

    Callers who genuinely need to pass extra vars to the child (e.g. a
    ``GITHUB_TOKEN`` that the child tool specifically requires) must add
    them back explicitly — the default posture is "trust nothing from the
    parent beyond what the child literally cannot run without".
    """
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in _CHILD_ENV_ALLOWLIST or key.startswith(_CHILD_ENV_ALLOWLIST_PREFIXES):
            env[key] = value
    if srt_available and include_srt_debug:
        env["SRT_DEBUG"] = "1"
    return env


def _split_srt_stderr(stderr: str) -> tuple[str, list[str]]:
    """Separate srt's ``[SandboxDebug]`` lines from the child process's stderr.

    With ``SRT_DEBUG=1`` srt interleaves its own diagnostic lines (prefixed
    ``[SandboxDebug]``) on the same stderr stream as the child.  For callers
    that surface stderr to users (Slack, logs) this bleed is undesirable —
    returning the cleaned child stderr, and the raw srt lines separately,
    lets us keep each audience happy.

    Returns:
        ``(child_stderr, srt_debug_lines)`` — the cleaned child stderr with
        srt lines removed, and the list of captured ``[SandboxDebug]`` lines.
    """
    srt_lines = _SRT_DEBUG_LINE_RE.findall(stderr)
    cleaned = _SRT_DEBUG_LINE_RE.sub("", stderr)
    # Collapse the blank lines left behind by removed srt lines.
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip("\n")
    # Preserve a trailing newline when the original had one and content remains.
    if cleaned and stderr.endswith("\n"):
        cleaned = cleaned + "\n"
    return cleaned, srt_lines


def _parse_violations(
    returncode: int,
    child_stderr: str,
    srt_lines: list[str],
) -> list[SandboxViolation]:
    """Build a structured violation list from srt debug lines and child stderr.

    Emits at most one ``sigabrt`` entry (exit 134), one entry per
    ``Connection blocked to`` line in *srt_lines*, one deduplicated entry per
    ``fs_deny:`` line in *srt_lines* (macOS Seatbelt file-* denials emitted by
    srt's log monitor when ``SRT_DEBUG=1``), and — as a fallback when no srt
    network-block line explained the failure — a single ``fs_eperm`` entry for
    EPERM-like patterns in *child_stderr*.
    """
    out: list[SandboxViolation] = []

    if returncode == 134:
        out.append(
            SandboxViolation(
                kind="sigabrt",
                detail="child terminated by SIGABRT (exit 134)",
            )
        )

    seen_fs_deny: set[tuple[str, str]] = set()
    for line in srt_lines:
        net_match = _NETWORK_BLOCK_RE.search(line)
        if net_match is not None:
            host = net_match.group("host") or ""
            port = net_match.group("port") or ""
            target = f"{host}:{port}" if port else host
            out.append(
                SandboxViolation(
                    kind="network_block",
                    detail=line.strip(),
                    target=target,
                )
            )
            continue
        fs_match = _FS_DENY_RE.search(line)
        if fs_match is not None:
            operation = fs_match.group("operation") or ""
            target = fs_match.group("target") or ""
            key = (operation, target)
            if key not in seen_fs_deny:
                seen_fs_deny.add(key)
                out.append(SandboxViolation(kind="fs_deny", detail=operation, target=target))

    # fs_eperm fires only when no srt-emitted network_block already explained
    # the failure.  We intentionally do NOT carry the child stderr snippet —
    # it came from the untrusted child and could contain secrets (leaked env
    # vars, credentials in error paths).  The returned violation keeps `kind`
    # so callers see "an EPERM happened" without ever exposing that payload.
    if not any(v.kind == "network_block" for v in out) and _FS_EPERM_RE.search(child_stderr):
        out.append(SandboxViolation(kind="fs_eperm"))

    return out


def _wrap_cmd(cmd: list[str], config: SandboxConfig) -> list[str]:
    """Return *cmd* wrapped with ``srt --settings <path> --`` when srt is on ``PATH``.

    The ``--`` separator prevents srt from stealing short flags (e.g. ``-s``)
    from the wrapped command's own arguments.  Passes *cmd* through unchanged
    when ``srt`` is not found on ``PATH``.  Never mutates the caller's list —
    always returns a new list.
    """
    if not is_sandbox_available():
        return list(cmd)
    return ["srt", "--settings", str(config.settings_path), "--", *cmd]


def _find_git_root(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default cwd) looking for a ``.git`` entry.

    ``.git`` is a directory in a plain repo and a file in worktrees/submodules;
    both qualify as a git root marker.  Returns the directory containing
    ``.git``, or None if no marker is found up to the filesystem root.
    Tolerates :class:`OSError` (permission denied on an unreadable ancestor)
    by continuing past that ancestor.
    """
    origin = start if start is not None else Path.cwd()
    cwd = origin.resolve()
    for ancestor in [cwd, *cwd.parents]:
        try:
            if (ancestor / ".git").exists():
                return ancestor
        except OSError:
            continue
    return None


def _find_jj_workspace_root(start: Path | None = None) -> Path | None:
    """Return the current jj workspace root using jj's own resolver."""
    return _run_jj_path_command(["workspace", "root"], cwd=start)


def _find_jj_git_backend_root(start: Path | None = None) -> Path | None:
    """Return the backing Git repository path for a Git-backed jj repo."""
    return _run_jj_path_command(["git", "root"], cwd=start)


def _find_jj_repo_root(start: Path | None = None) -> Path | None:
    """Return the default workspace's jj repo store for the current jj repo."""
    if _find_jj_workspace_root(start=start) is None:
        return None
    default_workspace_root = _run_jj_path_command(["workspace", "root", "--name", "default"], cwd=start)
    if default_workspace_root is None:
        return None
    repo_path = default_workspace_root / ".jj" / "repo"
    try:
        if not repo_path.is_dir():
            return None
        return repo_path.resolve()
    except OSError:
        return None


def _run_jj_path_command(args: list[str], cwd: Path | None = None) -> Path | None:
    """Run a read-only jj path command and return its single path output."""
    command = ["jj", "--ignore-working-copy", "--no-pager", *args]
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError, TypeError):
        return None
    raw = result.stdout.strip()
    if raw == "":
        return None
    first_line = raw.splitlines()[0]
    if first_line == "":
        return None
    return Path(first_line).resolve()


def _deduplicate_preserving_order(entries: list[str]) -> list[str]:
    deduplicated_entries: list[str] = []
    seen_entries: set[str] = set()
    for entry in entries:
        if entry in seen_entries:
            continue
        seen_entries.add(entry)
        deduplicated_entries.append(entry)
    return deduplicated_entries


def _substitute_or_remove(settings: dict, sentinel: str, replacement: str | None) -> dict:
    """Replace one ``allowWrite`` sentinel, dropping it when no root exists."""
    allow_write = settings.get("filesystem", {}).get("allowWrite", [])
    if not any(sentinel in entry for entry in allow_write):
        return settings
    if replacement is None:
        settings["filesystem"]["allowWrite"] = [entry for entry in allow_write if sentinel not in entry]
        return settings
    resolved_allow_write: list[str] = []
    for entry in allow_write:
        if sentinel in entry:
            resolved_allow_write.append(entry.replace(sentinel, replacement))
        else:
            resolved_allow_write.append(entry)
    settings["filesystem"]["allowWrite"] = resolved_allow_write
    return settings


def _substitute_jj_workspace_root(settings: dict) -> dict:
    """Replace the ``__JJ_WORKSPACE_ROOT__`` sentinel with the current jj workspace root."""
    allow_write = settings.get("filesystem", {}).get("allowWrite", [])
    if not any(_JJ_WORKSPACE_ROOT_SENTINEL in entry for entry in allow_write):
        return settings
    workspace_root = _find_jj_workspace_root()
    replacement = None
    if workspace_root is not None:
        replacement = str(workspace_root.resolve())
    return _substitute_or_remove(settings, _JJ_WORKSPACE_ROOT_SENTINEL, replacement)


def _substitute_jj_repo_root(settings: dict) -> dict:
    """Replace the ``__JJ_REPO_ROOT__`` sentinel with the default jj repo store."""
    allow_write = settings.get("filesystem", {}).get("allowWrite", [])
    if not any(_JJ_REPO_ROOT_SENTINEL in entry for entry in allow_write):
        return settings
    repo_root = _find_jj_repo_root()
    replacement = None
    if repo_root is not None:
        replacement = str(repo_root)
    return _substitute_or_remove(settings, _JJ_REPO_ROOT_SENTINEL, replacement)


def _substitute_jj_git_backend_root(settings: dict) -> dict:
    """Replace the ``__JJ_GIT_BACKEND__`` sentinel with the backing Git repository."""
    allow_write = settings.get("filesystem", {}).get("allowWrite", [])
    if not any(_JJ_GIT_BACKEND_SENTINEL in entry for entry in allow_write):
        return settings
    replacement = None
    git_backend_path = _find_jj_git_backend_root()
    if git_backend_path is not None:
        replacement = str(git_backend_path)
    return _substitute_or_remove(settings, _JJ_GIT_BACKEND_SENTINEL, replacement)


def _find_git_common_dir() -> Path | None:
    """Return the git common directory for the current repo.

    For plain repos this is the ``.git`` directory itself; for linked worktrees
    it is the shared ``.git`` directory of the main worktree — the path that
    ``git rev-parse --git-common-dir`` resolves to.  Returns None when cwd is
    not inside a git repo or the command fails.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        )
        raw = result.stdout.strip()
        if not raw:
            return None
        return Path(raw).resolve()
    except (subprocess.CalledProcessError, OSError):
        return None


def _find_git_repo_root() -> Path | None:
    """Return the root of the main repository.

    In a plain repo this equals the worktree root.  In a linked worktree it
    is the parent of the shared ``.git`` directory — the root of the original
    repository.  Returns None when cwd is not inside a git repo or the git
    command fails.
    """
    common_dir = _find_git_common_dir()
    if common_dir is None:
        return None
    return common_dir.parent


def _find_override_file(start: Path | None = None) -> Path | None:
    """Walk up from the git repo root looking for ``security_profile.json``.

    The walk begins at the main repository root — the parent of
    ``git rev-parse --git-common-dir`` — so that worktrees checked out
    outside ``$HOME`` (e.g. in ``/tmp``) still pick up a
    ``security_profile.json`` placed in the original repo or any ancestor
    up to ``$HOME``.  Falls back to cwd when not inside a git repo.
    Pass *start* explicitly to override the starting point (used in tests).

    Only ancestors at or below the caller's ``$HOME`` are considered — the
    walk refuses to cross the home-directory boundary and never reads from
    ``/``, ``/etc``, ``/opt``, ``/var``, or similar system-owned prefixes.
    This is a trust-boundary guard: a ``security_profile.json`` placed
    outside ``$HOME`` could have been written by any principal on the
    machine (root, CI runners, co-tenants on shared infra) and silently
    loosening the user's sandbox based on that file would be a privilege-
    escalation surface.  Files under ``$HOME`` are assumed to be under the
    user's control.

    Returns the first matching path (closest-to-root wins), or None if no
    ancestor up to ``$HOME`` carries an override file.  Tolerates
    :class:`OSError` on unreadable ancestors by skipping past them.
    """
    if start is not None:
        cwd = start.resolve()
    else:
        repo_root = _find_git_repo_root()
        cwd = repo_root if repo_root is not None else Path.cwd().resolve()
    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        # No home directory resolvable — refuse to walk anywhere; treat as
        # no-override.  Better to fall back to a stricter base profile than
        # to walk into territory we can't reason about.
        return None
    for ancestor in [cwd, *cwd.parents]:
        try:
            is_home_or_below = ancestor == home or home in ancestor.parents
        except OSError:
            continue
        if not is_home_or_below:
            # We've walked above $HOME (or started above it).  Everything
            # further up is outside the user's trust boundary; stop here.
            return None
        try:
            candidate = ancestor / _OVERRIDE_FILENAME
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _normalise_path_entry(entry: str) -> str:
    """Strip a single trailing slash for deny-wins comparison.

    ``"/a"`` and ``"/a/"`` compare equal after normalisation, so a deny entry
    without a trailing slash still matches an allow entry with one (and vice
    versa).  Single-character ``"/"`` is left unchanged.
    """
    if len(entry) > 1 and entry.endswith("/"):
        return entry[:-1]
    return entry


def _validate_override_schema(override: dict) -> None:
    """Validate *override* against ``_PROFILE_SCHEMA`` — fail fast on typos.

    srt's own zod parser silently drops unknown top-level keys, which means a
    typo like ``allowdDomains`` in a user's ``security_profile.json`` would
    parse, merge, and produce a profile that looks fine but enforces nothing
    it claimed to.  We catch that here instead, with error messages that
    include both the offending key and the valid set.

    Raises:
        ValueError: override has an unknown section, an unknown key within a
            section, a value of the wrong primitive type, a list whose
            elements are not all strings, or a section whose value is not a
            dict.  Messages always name the section (and key, when
            applicable) and list the valid alternatives so typos are obvious
            to correct.
    """
    if not isinstance(override, dict):
        raise ValueError(f"Override must be a dict, got {type(override).__name__}")
    for section, section_value in override.items():
        if section not in _PROFILE_SCHEMA:
            raise ValueError(f"Unknown override section {section!r}. Valid sections: {sorted(_PROFILE_SCHEMA)}")
        if not isinstance(section_value, dict):
            raise ValueError(f"Override section {section!r} must be a dict, got {type(section_value).__name__}")
        section_schema = _PROFILE_SCHEMA[section]
        for key, value in section_value.items():
            if key not in section_schema:
                raise ValueError(
                    f"Unknown override key {section}.{key!r}. Valid keys for {section!r}: {sorted(section_schema)}"
                )
            expected_type = section_schema[key]
            # bool is a subclass of int in Python — accept only the exact
            # type for primitive keys so "allowAllDomains": 1 doesn't sneak
            # past as a truthy int.
            if expected_type is bool:
                if not isinstance(value, bool):
                    raise ValueError(f"Override {section}.{key} must be bool, got {type(value).__name__}")
            elif expected_type is list:
                if not isinstance(value, list):
                    raise ValueError(f"Override {section}.{key} must be list, got {type(value).__name__}")
                for i, entry in enumerate(value):
                    if not isinstance(entry, str):
                        raise ValueError(f"Override {section}.{key}[{i}] must be str, got {type(entry).__name__}")
            elif not isinstance(value, expected_type):
                raise ValueError(
                    f"Override {section}.{key} must be {expected_type.__name__}, got {type(value).__name__}"
                )


def _merge_profile(base: dict, override: dict) -> dict:
    """Deep-merge *override* into a copy of *base*.

    List fields in ``_LIST_UNION_KEYS`` are appended in order; profile
    resolution deduplicates list entries after path expansion.
    Scalar fields in the schema (``allowAllDomains``, ``allowGitConfig``) are
    replaced by the override value.  Sections absent from *override* leave
    the corresponding base section unchanged.  Never mutates the inputs.

    Override schema (sections, keys, types) is validated up-front via
    :func:`_validate_override_schema` — typos and type mismatches raise
    :class:`ValueError` before any merging happens.
    """
    _validate_override_schema(override)
    merged = copy.deepcopy(base)
    for section, section_delta in override.items():
        if section not in merged:
            merged[section] = copy.deepcopy(section_delta)
            continue
        for key, value in section_delta.items():
            if (section, key) in _LIST_UNION_KEYS:
                existing = merged[section].get(key, [])
                merged[section][key] = [*existing, *value]
            else:
                merged[section][key] = copy.deepcopy(value)
    return merged


def _apply_deny_wins(settings: dict) -> dict:
    """Return a new settings dict with allow entries removed when denied.

    For each paired allow/deny list (``allowRead``/``denyRead``,
    ``allowWrite``/``denyWrite`` in ``filesystem``; ``allowedDomains``/
    ``deniedDomains`` in ``network``), drop any allow entry that matches a
    deny entry (after trailing-slash normalisation for paths).  The deny
    lists themselves are preserved unchanged.  Never mutates the input.
    """
    result = copy.deepcopy(settings)
    fs = result.get("filesystem", {})
    for allow_key, deny_key in (("allowRead", "denyRead"), ("allowWrite", "denyWrite")):
        allow = fs.get(allow_key, [])
        deny_normalised = {_normalise_path_entry(d) for d in fs.get(deny_key, [])}
        fs[allow_key] = [a for a in allow if _normalise_path_entry(a) not in deny_normalised]
    net = result.get("network", {})
    allowed = net.get("allowedDomains", [])
    denied = set(net.get("deniedDomains", []))
    net["allowedDomains"] = [a for a in allowed if a not in denied]
    return result


def _carve_extra_allow_read(settings: dict, extra_allow_read: tuple[str, ...]) -> dict:
    """Return *settings* with *extra_allow_read* paths forced into ``filesystem.allowRead``.

    Applied **after** :func:`_apply_deny_wins`, deliberately: these are explicit,
    operator-authorized re-allows (e.g. a ``--secrets <file>`` path) that must
    survive even when the path matches a ``denyRead`` glob like ``**/.env``.
    srt's last-match-wins then reopens the named path, while every *other*
    ``.env`` stays denied — deny-wins is exact-match and never sees these
    post-hoc entries.  Paths are expanded (``~``/``$VAR``/macOS symlink
    canonicalisation) for parity with base entries and deduplicated.  A no-op
    for the empty-tuple default, so callers that pass nothing keep current
    behaviour.  Never mutates the input.
    """
    if not extra_allow_read:
        return settings
    result = copy.deepcopy(settings)
    fs = result.setdefault("filesystem", {})
    expanded = [_expand_path_entry(path) for path in extra_allow_read]
    fs["allowRead"] = _deduplicate_preserving_order([*fs.get("allowRead", []), *expanded])
    return result


def _expand_path_entry(entry: str) -> str:
    """Expand ``~`` / ``$VAR`` and canonicalise symlinks in a filesystem path.

    Canonicalisation matters on macOS where common paths are symlinks — ``/tmp``
    → ``/private/tmp``, ``/var`` → ``/private/var`` — and srt matches against
    the real filesystem path at syscall time.  A profile entry of ``"/tmp"``
    without canonicalisation fails to authorise writes to ``/tmp/foo`` because
    the kernel sees ``/private/tmp/foo``.  We therefore resolve the path for
    absolute, non-glob entries; relative entries like ``"."`` and glob patterns
    (``*.pem``, ``id_rsa*``) are left alone.

    Globs are left untouched — srt handles them natively on macOS; Linux treats
    them as literals (a known limitation tracked by v2 ``LINUX-01``).
    """
    expanded = os.path.expanduser(entry)
    expanded = os.path.expandvars(expanded)
    if expanded.startswith("/") and not any(c in expanded for c in _GLOB_CHARS):
        try:
            expanded = str(Path(expanded).resolve())
        except OSError:
            pass
    return expanded


def _substitute_git_worktree_root(settings: dict) -> dict:
    """Replace the ``__GIT_WORKTREE_ROOT__`` sentinel in ``allowWrite`` with the worktree root.

    Walks up from cwd looking for ``.git``; falls back to cwd when no repo is
    found.  In a linked worktree this resolves to the worktree directory itself
    (e.g. ``/tmp/worktrees/feat-xxx``), not the original repo root.
    Mutates *settings* in place and returns it; the caller is expected to pass
    a dict it owns (see :func:`resolve_profile` for the single deepcopy).
    """
    allow_write = settings.get("filesystem", {}).get("allowWrite", [])
    if not any(_GIT_WORKTREE_ROOT_SENTINEL in entry for entry in allow_write):
        return settings
    git_root = _find_git_root()
    replacement = str(git_root) if git_root is not None else str(Path.cwd().resolve())
    settings["filesystem"]["allowWrite"] = [
        entry.replace(_GIT_WORKTREE_ROOT_SENTINEL, replacement) if _GIT_WORKTREE_ROOT_SENTINEL in entry else entry
        for entry in allow_write
    ]
    return settings


def _substitute_git_repo_root(settings: dict) -> dict:
    """Replace the ``__GIT_REPO_ROOT__`` sentinel in ``allowWrite`` with the main repo root.

    For plain repos this equals the worktree root.  For linked worktrees it is
    the parent of the shared ``.git`` directory — the root of the original
    repository — so the original repo remains writable even when the worktree
    lives elsewhere (e.g. ``/tmp``).  Falls back to cwd when no repo is found.
    Mutates *settings* in place and returns it; the caller is expected to pass
    a dict it owns (see :func:`resolve_profile` for the single deepcopy).
    """
    allow_write = settings.get("filesystem", {}).get("allowWrite", [])
    if not any(_GIT_REPO_ROOT_SENTINEL in entry for entry in allow_write):
        return settings
    repo_root = _find_git_repo_root()
    replacement = str(repo_root) if repo_root is not None else str(Path.cwd().resolve())
    settings["filesystem"]["allowWrite"] = [
        entry.replace(_GIT_REPO_ROOT_SENTINEL, replacement) if _GIT_REPO_ROOT_SENTINEL in entry else entry
        for entry in allow_write
    ]
    return settings


def _substitute_cwd(settings: dict) -> dict:
    """Replace the ``__CWD__`` sentinel anywhere in the filesystem section with cwd.

    Used by the ``locked`` profile to re-allow reads inside the current working
    directory while denying the rest of ``$HOME``.  Resolves cwd to its
    canonical path so srt's symlink-aware matching lines up (``/tmp`` →
    ``/private/tmp`` on macOS).  Mutates *settings* in place and returns it;
    the caller is expected to pass a dict it owns.
    """
    fs = settings.get("filesystem", {})
    if not any(_CWD_SENTINEL in fs.get(key, []) for key in ("allowRead", "allowWrite", "denyRead", "denyWrite")):
        return settings
    cwd = str(Path.cwd().resolve())
    for key in ("allowRead", "allowWrite", "denyRead", "denyWrite"):
        entries = fs.get(key, [])
        if _CWD_SENTINEL in entries:
            fs[key] = [cwd if entry == _CWD_SENTINEL else entry for entry in entries]
    return settings


def _expand_all_path_entries(settings: dict) -> dict:
    """Expand ``~`` / ``$VAR`` in every FS list in *settings*, mutating in place."""
    fs = settings.get("filesystem", {})
    for key in ("allowRead", "allowWrite", "denyRead", "denyWrite"):
        if key in fs:
            fs[key] = [_expand_path_entry(entry) for entry in fs[key]]
    return settings


def _deduplicate_profile_lists(settings: dict) -> dict:
    """Deduplicate profile list entries after substitutions and path expansion."""
    for section, section_schema in _PROFILE_SCHEMA.items():
        section_settings = settings.get(section, {})
        for key, expected_type in section_schema.items():
            if expected_type is not list:
                continue
            if key not in section_settings:
                continue
            section_settings[key] = _deduplicate_preserving_order(section_settings[key])
    return settings


def resolve_profile(name: str | None = None, extra_allow_read: tuple[str, ...] = ()) -> dict:
    """Resolve a base profile name to a concrete srt settings dict.

    *extra_allow_read* is an explicit, operator-authorized set of paths
    force-added to ``filesystem.allowRead`` **after** deny-wins (step 11), so a
    named secrets file is readable inside the sandbox even though ``**/.env`` is
    deny-read; every other ``.env`` stays denied.  See
    :func:`_carve_extra_allow_read`.

    Resolution steps:
      1. ``name is None`` → ``"git"`` (the default alias).
      2. Look up ``_BASE_PROFILES[name]``; raise
         :class:`SandboxProfileNotFoundError` listing available names if missing.
      3. Deep-copy the base profile (module constants are never mutated).
      4. Substitute the ``__GIT_WORKTREE_ROOT__`` sentinel by walking up from
         cwd for ``.git``; in a linked worktree this is the worktree directory
         itself (e.g. ``/tmp/worktrees/feat-xxx``).  Falls back to cwd.
      4b. Substitute the ``__GIT_REPO_ROOT__`` sentinel with the parent of the
         git common dir — the root of the original repository.  In a plain repo
         this equals the worktree root; in a linked worktree it is the main
         repo root so that the original repo stays writable.  Falls back to cwd.
      4c. Substitute each jj write-root sentinel using jj's own path commands.
         Profiles without jj sentinels are unchanged; sentinels are dropped
         when cwd is not inside a jj workspace.
      5. Substitute the ``__CWD__`` sentinel (``locked`` profile) with the
         canonicalised current working directory.
      6. Expand ``~`` / ``$VAR`` in every filesystem path entry.
      7. If a ``security_profile.json`` exists in cwd or any ancestor up to
         ``$HOME`` (first match wins; see :func:`_find_override_file` for
         the trust-boundary rationale), load it and deep-merge as partial
         deltas: list fields appended, scalar fields replaced.  Override
         schema is validated up-front — typos like ``allowdDomains`` raise
         :class:`ValueError`.
      8. Re-run path expansion across the merged dict so override entries
         using ``~``, ``$VAR``, or macOS symlink-shadowed paths (``/tmp``
         → ``/private/tmp``) land in the same canonical form as base
         entries.  Without this pass an override ``denyRead: ["~/.ssh"]``
         would be stored as the literal string ``~/.ssh`` and srt would
         never match it against the kernel's resolved path at syscall time.
      9. Deduplicate list entries after substitutions and path expansion.
      10. Apply deny-wins conflict resolution — any allow entry that also
         appears in the corresponding deny list is removed from allow
         (trailing-slash normalisation for filesystem paths).
      11. Force *extra_allow_read* paths into ``allowRead`` after deny-wins, so
         an explicitly-authorized secrets file is readable even when it matches
         a ``denyRead`` glob.

    Raises:
        SandboxProfileNotFoundError: *name* is not a known base profile.
        ValueError: the override file exists but fails schema validation
            (unknown section/key, wrong type, non-string list element).
        json.JSONDecodeError: the override file exists but is not valid JSON.
    """
    resolved_name = _DEFAULT_PROFILE_NAME if name is None else name
    if resolved_name not in _BASE_PROFILES:
        raise SandboxProfileNotFoundError(
            f"Unknown sandbox profile {resolved_name!r}. Available: {sorted(_BASE_PROFILES)}"
        )
    # One deepcopy up-front — substitutions then mutate this private copy in
    # place.  Avoids re-deepcopying the same dict 4× per resolve_profile call.
    working = copy.deepcopy(_BASE_PROFILES[resolved_name])
    _substitute_git_worktree_root(working)
    _substitute_git_repo_root(working)
    _substitute_jj_workspace_root(working)
    _substitute_jj_repo_root(working)
    _substitute_jj_git_backend_root(working)
    _substitute_cwd(working)
    expanded = _expand_all_path_entries(working)
    override_path = _find_override_file()
    if override_path is None:
        deduplicated = _deduplicate_profile_lists(expanded)
        final = _apply_deny_wins(deduplicated)
    else:
        with override_path.open("r", encoding="utf-8") as fh:
            override_data = json.load(fh)
        merged = _merge_profile(expanded, override_data)
        # Re-run expansion so override paths get the same treatment as base
        # entries — ``~``/``$VAR``/macOS symlink canonicalisation.  Without
        # this, ``{"filesystem": {"denyRead": ["~/.ssh"]}}`` in the override
        # would land in the merged dict as the literal ``~/.ssh`` and srt's
        # kernel-level matching (which sees ``/Users/<me>/.ssh``) would
        # silently no-op the deny.
        merged = _expand_all_path_entries(merged)
        deduplicated = _deduplicate_profile_lists(merged)
        final = _apply_deny_wins(deduplicated)
    final = _carve_extra_allow_read(final, extra_allow_read)
    return final


def list_profiles() -> list[str]:
    """Return the sorted list of known base sandbox profile names."""
    return sorted(_BASE_PROFILES)


def wrap_command(cmd: list[str], profile: str) -> list[str]:
    """Return *cmd* prefixed with ``srt --settings <path> --`` when srt is available.

    Pure wrapping helper — no subprocess execution, no violation parsing, no
    stderr capture.  Use this when you need to launch a process under the sandbox but
    manage its lifecycle yourself (e.g. an interactive TUI via
    :class:`subprocess.Popen` with inherited stdio, or a long-lived agent where
    :meth:`Sandboxed.run`'s captured one-shot model doesn't fit).

    When ``srt`` is not on ``PATH`` a copy of *cmd* is returned unchanged.
    Callers requiring strict enforcement should check
    :func:`is_sandbox_available` first.

    Args:
        cmd: The command and its arguments.
        profile: Path string to an srt settings JSON file.  ``~`` and ``$VAR``
            tokens are expanded via :func:`expand_settings_path`.

    Returns:
        New list containing the wrapped (or passed-through) command.  Never
        mutates *cmd*.

    Raises:
        SandboxVariableError: If *profile* contains unresolved ``$VAR`` or
            ``~`` tokens.
    """
    settings_path = expand_settings_path(profile)
    config = SandboxConfig(settings_path=settings_path)
    return _wrap_cmd(cmd, config)


def _write_profile_tempfile(settings_dict: dict) -> str:
    """Materialise *settings_dict* to a per-invocation ``.json`` tempfile.

    Returns the tempfile's path; the caller owns cleanup (typically via
    :func:`_cleanup_tempfile` in a ``finally`` block or context-manager exit).
    Using ``delete=False`` because srt reads the file after we close it in
    this process — we can't rely on NamedTemporaryFile's auto-delete.
    """
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    try:
        json.dump(settings_dict, tmp)
    finally:
        tmp.close()
    return tmp.name


def _cleanup_tempfile(path: str) -> None:
    """Unlink *path*, swallowing :class:`OSError` for idempotence.

    Tempfile cleanup must never raise — callers invoke this from ``finally``
    blocks and context-manager ``__exit__`` methods where an exception would
    mask the original error.
    """
    try:
        os.unlink(path)
    except OSError:
        pass


def sandbox_run_env(available: bool, *, include_srt_debug: bool = True) -> dict[str, str]:
    """Return the scrubbed child env for a sandboxed ``.run()`` / ``.popen()`` call.

    Thin wrapper over :func:`_minimal_child_env` — kept as a separate symbol so
    ``.run()`` and ``.popen()`` share a single call site and future env-policy
    tweaks land in one place.  Always returns a dict (never ``None``) so the
    child never silently inherits the parent's full environment.  See
    :func:`_minimal_child_env` for the allowlist rationale and threat model.

    ``include_srt_debug`` gates the ``SRT_DEBUG=1`` injection.  ``.run()``
    always captures stderr for violation parsing and keeps the default
    ``True``.  ``.popen()`` callers that leave stderr inherited (interactive
    TTY passthrough) pass ``False`` — otherwise srt's ``[SandboxDebug]``
    startup chatter dumps straight into the user's terminal and garbles
    fullscreen TUIs.
    """
    return _minimal_child_env(available, include_srt_debug=include_srt_debug)


class SandboxedPopen:
    """Long-lived sandboxed subprocess — a :class:`subprocess.Popen` wrapper.

    Use :meth:`Sandboxed.popen` to construct.  Supports streaming stdout/stderr
    line-by-line from agents that need to ship output incrementally (Slack, UI,
    log forwarders), as well as interactive TTY pass-through by leaving stdio
    at ``None``.  The profile tempfile is cleaned up on :meth:`wait` or
    context-manager exit — whichever comes first.

    Example (streaming stdout)::

        with Sandboxed(cmd=["claude", "-p", "..."], profile="git").popen(
            stdout=subprocess.PIPE, text=True, capture_stderr=True,
        ) as proc:
            for line in proc.stdout:
                ship_to_slack(line)
            result = proc.wait()
            if result.sandbox_violations:
                for v in result.violations:
                    handle_violation(v.kind, v.target)

    Example (interactive TTY)::

        with Sandboxed(cmd=["claude"], profile="sealed").popen() as proc:
            proc.wait()  # stdio inherited from parent; no violation parsing

    Delegates :attr:`stdin`, :attr:`stdout`, :attr:`stderr`, :attr:`returncode`,
    :meth:`poll`, :meth:`terminate`, :meth:`kill`, :meth:`send_signal` to the
    underlying :class:`subprocess.Popen`.
    """

    def __init__(
        self,
        proc: subprocess.Popen[str],
        *,
        tmp_path: str,
        capture_stderr: bool,
    ) -> None:
        self._proc = proc
        self._tmp_path = tmp_path
        self._result: SandboxResult | None = None
        # Background stderr drainer — started at construction so the OS pipe
        # never fills while the caller is busy reading stdout.  Reading
        # stderr only after ``proc.wait()`` deadlocks when the child writes
        # more than the pipe buffer (~64 KiB on Linux/macOS): child blocks
        # on ``write()``, parent blocks in ``wait()``, neither progresses.
        # ``SRT_DEBUG=1`` makes this trivially reachable — srt logs one
        # ``[SandboxDebug]`` line per syscall, so a moderately chatty
        # subagent overruns 64 KiB in seconds.  The drainer consumes the
        # stream into :attr:`_stderr_chunks`; :meth:`wait` joins it and
        # concatenates.
        self._stderr_chunks: list[str] = []
        self._stderr_drainer: threading.Thread | None = None
        if capture_stderr and proc.stderr is not None:
            self._stderr_drainer = threading.Thread(
                target=self._drain_stderr,
                name="agent-sandbox-stderr-drainer",
                daemon=True,
            )
            self._stderr_drainer.start()

    def _drain_stderr(self) -> None:
        """Read the child's stderr stream into ``self._stderr_chunks`` until EOF.

        Runs in a daemon thread started at :meth:`__init__`.  Stops when
        ``read()`` returns an empty chunk (child closed stderr, typically at
        exit) or the stream is closed out from under us (``ValueError`` on
        I/O against a closed file).  Never raises into the thread — the
        chunks collected so far remain available to :meth:`wait`.
        """
        stream = self._proc.stderr
        if stream is None:
            return
        try:
            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                self._stderr_chunks.append(chunk)
        except (ValueError, OSError):
            pass

    @property
    def proc(self) -> subprocess.Popen[str]:
        """The underlying :class:`subprocess.Popen` — escape hatch for callers needing raw access."""
        return self._proc

    @property
    def stdin(self) -> IO[str] | None:
        return self._proc.stdin

    @property
    def stdout(self) -> IO[str] | None:
        return self._proc.stdout

    @property
    def stderr(self) -> IO[str] | None:
        return self._proc.stderr

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    @property
    def pid(self) -> int:
        return self._proc.pid

    def poll(self) -> int | None:
        """Proxy for :meth:`subprocess.Popen.poll` — non-blocking exit check."""
        return self._proc.poll()

    def terminate(self) -> None:
        """Proxy for :meth:`subprocess.Popen.terminate` — sends SIGTERM."""
        self._proc.terminate()

    def kill(self) -> None:
        """Proxy for :meth:`subprocess.Popen.kill` — sends SIGKILL."""
        self._proc.kill()

    def send_signal(self, sig: int) -> None:
        """Proxy for :meth:`subprocess.Popen.send_signal`."""
        self._proc.send_signal(sig)

    def wait(self, timeout: float | None = None) -> SandboxResult:
        """Block until the child exits and return a :class:`SandboxResult`.

        Idempotent — a second call returns the cached result.  Parses
        violations from captured stderr when the wrapper was constructed with
        ``capture_stderr=True``; otherwise :attr:`SandboxResult.violations` is
        an empty tuple.

        Raises:
            subprocess.TimeoutExpired: *timeout* elapsed before the child
                exited — call :meth:`wait` again (without a timeout, or with a
                longer one) to collect the result.
        """
        if self._result is not None:
            return self._result

        self._proc.wait(timeout=timeout)

        if self._stderr_drainer is not None:
            # Child has exited → stderr is closed → drainer will break out
            # of its read loop imminently.  join() is bounded in practice;
            # guard with a timeout anyway so a rogue buffering layer can't
            # hang the caller.
            self._stderr_drainer.join(timeout=5.0)
            raw_stderr = "".join(self._stderr_chunks)
        else:
            raw_stderr = ""
        child_stderr, srt_lines = _split_srt_stderr(raw_stderr)
        violations = _parse_violations(self._proc.returncode or 0, child_stderr, srt_lines)

        self._result = SandboxResult(
            exit_code=self._proc.returncode or 0,
            stdout="",
            stderr=child_stderr,
            sandbox_violations=bool(violations),
            violations=tuple(violations),
        )
        return self._result

    def close(self) -> None:
        """Clean up the profile tempfile — safe to call more than once.

        Called automatically by :meth:`__exit__`.  Callers not using the
        context-manager form should invoke this after the child has exited
        to avoid leaking tempfiles in ``/tmp``.  Does not terminate or wait
        on the child process.
        """
        _cleanup_tempfile(self._tmp_path)

    def __enter__(self) -> "SandboxedPopen":
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        # If the child is still running on context exit — whether because of
        # an exception in the with-block or because the caller never waited —
        # terminate it cleanly.  Escalate to SIGKILL if SIGTERM doesn't land.
        if self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        try:
            self.wait()
        except subprocess.TimeoutExpired:
            pass
        self.close()


@dataclasses.dataclass(frozen=True)
class Sandboxed:
    """A sandboxed subprocess invocation backed by ``srt``.

    Example::

        result = Sandboxed(
            cmd=["git", "clone", "https://github.com/org/repo"],
            profile="git",
        ).run()
        if result.sandbox_violations:
            for v in result.violations:
                handle_violation(v.kind, v.target)

    Attributes:
        cmd: The command and its arguments to execute.
        profile: Name of a base sandbox profile (``"locked"``, ``"sealed"``,
            ``"git"``, ``"open"``).  ``None`` (or omitted) resolves to the
            default alias ``"git"``.  Unknown names raise
            :class:`SandboxProfileNotFoundError` at :meth:`run` time.
        strict: When ``True`` and ``srt`` is not on ``PATH``, :meth:`run`
            raises :class:`SandboxUnavailableError` instead of falling back
            to unsandboxed execution.  Defaults to ``False`` so that dev
            environments without the ``srt`` binary installed keep working,
            but callers that treat ``Sandboxed`` as a hard enforcement
            boundary (security-sensitive subprocesses, audited pipelines)
            should opt in.
        extra_allow_read: Explicit, operator-authorized paths force-added to
            ``filesystem.allowRead`` after deny-wins, so a named secrets file
            stays readable inside the sandbox even though it matches a
            ``denyRead`` glob like ``**/.env``.  Defaults to ``()`` (no-op).
    """

    cmd: list[str]
    profile: str | None = None
    strict: bool = False
    extra_allow_read: tuple[str, ...] = ()

    def run(self) -> SandboxResult:
        """Execute :attr:`cmd` inside the sandbox described by :attr:`profile`.

        Resolves :attr:`profile` via :func:`resolve_profile` and materialises
        the resulting srt settings to a per-invocation tempfile, which is
        deleted after the child process exits (whether it succeeds, fails, or
        raises).  When ``srt`` is not on ``PATH`` and :attr:`strict` is
        ``False`` (the default) the command runs unsandboxed (passthrough
        mode) — the tempfile is still written (harmless) so the wrapping path
        stays uniform.  With :attr:`strict` = ``True`` the same condition
        raises :class:`SandboxUnavailableError` before the child is spawned.

        Returns:
            :class:`SandboxResult` with cleaned child stderr and structured
            violations list.

        Raises:
            SandboxProfileNotFoundError: :attr:`profile` is not a known base
                profile name.
            SandboxUnavailableError: :attr:`strict` is ``True`` and ``srt``
                is not on ``PATH``.
        """
        available = is_sandbox_available()
        if self.strict and not available:
            raise SandboxUnavailableError(
                "srt is not on PATH and strict=True was requested — "
                "refusing to run command unsandboxed. Install srt or pass "
                "strict=False to opt into passthrough mode."
            )

        settings_dict = resolve_profile(self.profile, extra_allow_read=self.extra_allow_read)

        tmp_path = _write_profile_tempfile(settings_dict)
        try:
            wrapped_cmd = wrap_command(self.cmd, tmp_path)
            proc = subprocess.run(
                wrapped_cmd,
                capture_output=True,
                text=True,
                env=sandbox_run_env(available),
            )
        finally:
            _cleanup_tempfile(tmp_path)

        child_stderr, srt_lines = _split_srt_stderr(proc.stderr)
        violations = _parse_violations(proc.returncode, child_stderr, srt_lines)

        return SandboxResult(
            exit_code=proc.returncode,
            stdout=proc.stdout,
            stderr=child_stderr,
            sandbox_violations=bool(violations),
            violations=tuple(violations),
        )

    def popen(
        self,
        *,
        capture_stderr: bool = False,
        **popen_kwargs: Any,
    ) -> SandboxedPopen:
        """Spawn :attr:`cmd` under the sandbox and return a :class:`SandboxedPopen`.

        Use this when :meth:`run`'s capture-and-return model doesn't fit —
        streaming stdout line-by-line, interactive TTY pass-through, or
        long-lived agents where the parent wants to observe the child mid-run.

        *popen_kwargs* are forwarded to :class:`subprocess.Popen` unchanged.
        Common patterns:

        - **Streaming stdout:** ``stdout=subprocess.PIPE, text=True``
        - **Interactive TTY:** leave stdio kwargs out — child inherits the
          parent's TTY and nothing is captured

        *capture_stderr* forces ``stderr=subprocess.PIPE`` so the wrapper can
        read srt's ``[SandboxDebug]`` lines at :meth:`SandboxedPopen.wait`
        time and populate :attr:`SandboxResult.violations`.  Pass ``True``
        from streaming callers that want jail-break observability; leave
        ``False`` (the default) for interactive use where an inherited stderr
        is the whole point.  Passing *capture_stderr* alongside an explicit
        ``stderr=`` kwarg raises :class:`ValueError` — callers must pick one.

        ``SRT_DEBUG=1`` is injected into the child env only when stderr is
        routed somewhere the caller controls (``capture_stderr=True`` or an
        explicit non-``None`` ``stderr=`` kwarg).  Under inherited stderr —
        the interactive-TUI default — SRT_DEBUG stays unset so srt's
        ``[SandboxDebug]`` startup chatter never dumps into the user's
        terminal and garbles fullscreen apps like claude / codex.

        Returns:
            :class:`SandboxedPopen` wrapping a live child process.  Use it as
            a context manager, or call :meth:`SandboxedPopen.wait` + :meth:`SandboxedPopen.close`
            manually.  Not waiting leaks the profile tempfile.

        Raises:
            SandboxProfileNotFoundError: :attr:`profile` is not a known base
                profile name.
            SandboxUnavailableError: :attr:`strict` is ``True`` and ``srt``
                is not on ``PATH``.
            ValueError: *capture_stderr* is ``True`` and *popen_kwargs* also
                specifies ``stderr=``.
        """
        if capture_stderr and "stderr" in popen_kwargs:
            raise ValueError(
                "capture_stderr=True conflicts with an explicit stderr= kwarg. "
                "Pick one: capture internally for violation parsing, or handle stderr yourself."
            )

        available = is_sandbox_available()
        if self.strict and not available:
            raise SandboxUnavailableError(
                "srt is not on PATH and strict=True was requested — "
                "refusing to spawn child unsandboxed. Install srt or pass "
                "strict=False to opt into passthrough mode."
            )

        settings_dict = resolve_profile(self.profile, extra_allow_read=self.extra_allow_read)

        tmp_path = _write_profile_tempfile(settings_dict)
        try:
            wrapped_cmd = wrap_command(self.cmd, tmp_path)

            if capture_stderr:
                popen_kwargs["stderr"] = subprocess.PIPE

            # SRT_DEBUG=1 only adds value when the caller routes stderr
            # somewhere they control — a pipe, a file, ``DEVNULL`` — so srt's
            # ``[SandboxDebug]`` lines can be parsed or stored.  When stderr
            # inherits the parent TTY those lines dump
            # straight into the user's terminal and garble fullscreen apps
            # at startup.  Auto-gate on that signal instead of adding a
            # separate flag.
            include_srt_debug = capture_stderr or popen_kwargs.get("stderr") is not None

            # Env policy:
            # - Caller passes nothing (default) → scrubbed env from
            #   `sandbox_run_env` (allowlist + SRT_DEBUG when wrapping AND
            #   stderr is routed).  This is the secure default; no parent
            #   creds leak into the sandboxed child.
            # - Caller passes an explicit env dict → respect it, but fold in
            #   SRT_DEBUG=1 when srt is on PATH and stderr is routed so
            #   violation parsing still works.  If the caller chose to
            #   include AWS_* etc. that's their explicit choice.
            caller_env = popen_kwargs.get("env")
            if caller_env is None:
                popen_kwargs["env"] = sandbox_run_env(available, include_srt_debug=include_srt_debug)
            elif available and include_srt_debug:
                popen_kwargs["env"] = {**caller_env, "SRT_DEBUG": "1"}

            proc = subprocess.Popen(wrapped_cmd, **popen_kwargs)
        except BaseException:
            _cleanup_tempfile(tmp_path)
            raise

        return SandboxedPopen(
            proc=proc,
            tmp_path=tmp_path,
            capture_stderr=capture_stderr,
        )
