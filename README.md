# agent-sandbox

General Agent Sandboxing with Sane Defaults — `--yolo` without the hassle.

`agent-sandbox` (`asb`) runs **any** coding agent (claude, codex, pi, …) under the
`srt` binary sandbox. It wraps your agent in a sane-default
filesystem/network sandbox so you can grant broad autonomy without worrying about
secrets leaking or your home directory getting clobbered.

## Install

From source with uv:

```
uv tool install agent-sandbox
# or, in a project:
uv pip install agent-sandbox
```

With AWS support (optional `[aws]` extra, pulls in boto3):

```
uv pip install "agent-sandbox[aws]"
```

Then bootstrap the runtime:

```
asb install
```

`asb install` provisions everything `asb` needs:

- the `srt` sandbox binary (via npm),
- the `sx` / `sxd` secret-broker (via cargo),
- a default `security_profile.json` in your repo.

## Usage

```
asb [-p PROFILE] [--secrets FILE] [--aws-profile NAME] [--aws-region REGION] -- <command...>
```

Everything after `--` is the agent command to run inside the sandbox.

Examples:

```
# Run claude with the "git" profile, resuming the last session
asb -p git -- claude --resume

# Run codex in yolo mode, exposing a single secrets file
asb --secrets .env -- codex --yolo

# Run pi with short-lived read-only AWS credentials
asb --aws-profile readonly -- pi
```

## Profiles

Select a profile with `-p` / `--profile`:

- **`git`** — read/write the current git repo; network and home access locked down.
- **`open`** — permissive profile for trusted tasks (broad filesystem + network).
- **`sealed`** — no network; filesystem limited to the working tree.
- **`locked`** — most restrictive; minimal filesystem, no network, no secrets.

## Secrets — `--secrets FILE`

By default the sandbox denies reads of `**/.env` and similar secret files.
`--secrets FILE` opens **one** named file for reading past that deny rule. The
file's contents are made readable to the agent on request — the values are
**never** injected into the environment.

## AWS — `--aws-profile` / `--aws-region`

`--aws-profile NAME` mints short-lived STS credentials for the named profile and
overlays them into the sandbox **via environment variables only**. Your real
`~/.aws` directory stays denied inside the sandbox, so long-lived credentials are
never exposed. `--aws-region` selects the region for the minted session.

AWS support lives behind the optional `[aws]` extra (boto3).

## `security_profile.json`

`asb` looks for a `security_profile.json` override file at the repository root,
discovered by walking up from the working directory to the git common dir. When
present it overrides the built-in profile defaults, letting a repo ship its own
sandbox policy.
