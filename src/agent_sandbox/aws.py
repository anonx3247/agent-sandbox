"""Mint AWS STS credentials from a named profile for injection into sandboxed children.

``asb`` strips AWS access from the sandbox by default (``~/.aws`` is denied, no
AWS env vars in the allowlist). The ``--aws-profile`` flag opts the user back in
by minting fresh STS credentials from a named SSO profile via boto3 and handing
them to the launcher to overlay into the child env. Credentials flow through env
vars only — the sandbox still denies ``~/.aws`` so the child can never refresh on
its own.

Credential lifetime is bounded by the profile's permission set ``session_duration``
on the AWS side.

boto3 is an optional dependency (the ``aws`` extra). It is imported lazily inside
:func:`mint_profile_creds` so the base install stays lean.
"""

from __future__ import annotations

from dataclasses import dataclass

import typer


@dataclass(frozen=True)
class AwsRuntimeCreds:
    access_key_id: str
    secret_access_key: str
    session_token: str


def mint_profile_creds(profile_name: str) -> AwsRuntimeCreds:
    """Return frozen STS credentials for *profile_name*.

    Raises ``typer.BadParameter`` with a user-actionable hint when boto3 is not
    installed, the profile is missing, the SSO token has expired, or the profile
    yields static IAM credentials (we require a session token so the agent's
    access is bounded).
    """
    try:
        import boto3
        from botocore.exceptions import ProfileNotFound, SSOTokenLoadError, UnauthorizedSSOTokenError
    except ImportError as exc:
        raise typer.BadParameter(
            "--aws-profile requires boto3. Install with: uv pip install 'agent-sandbox[aws]'"
        ) from exc

    try:
        session = boto3.Session(profile_name=profile_name)
    except ProfileNotFound as exc:
        raise typer.BadParameter(
            f"AWS profile '{profile_name}' not found in ~/.aws/config. "
            f"Run `aws sso login --profile {profile_name}` to refresh, then verify with "
            "`grep '\\[profile ' ~/.aws/config`."
        ) from exc

    credentials = session.get_credentials()
    if credentials is None:
        raise typer.BadParameter(
            f"AWS profile '{profile_name}' has no credentials configured. "
            f"Run `aws sso login --profile {profile_name}` to refresh."
        )

    try:
        frozen = credentials.get_frozen_credentials()
    except (UnauthorizedSSOTokenError, SSOTokenLoadError) as exc:
        raise typer.BadParameter(
            f"SSO session for profile '{profile_name}' is expired or missing. "
            f"Run `aws sso login --profile {profile_name}` to refresh."
        ) from exc

    if not frozen.token:
        raise typer.BadParameter(
            f"AWS profile '{profile_name}' yielded static IAM credentials (no session token). "
            "Use an SSO profile so the agent's access is time-bounded."
        )
    if not frozen.access_key or not frozen.secret_key:
        raise typer.BadParameter(
            f"AWS profile '{profile_name}' returned incomplete credentials "
            f"(missing access key or secret). Run `aws sso login --profile {profile_name}` to refresh."
        )

    return AwsRuntimeCreds(
        access_key_id=frozen.access_key,
        secret_access_key=frozen.secret_key,
        session_token=frozen.token,
    )


def build_aws_env(creds: AwsRuntimeCreds, region: str = "us-west-2") -> dict[str, str]:
    """Standard AWS_* env vars for a child process. ``AWS_REGION`` mirrors ``AWS_DEFAULT_REGION``."""
    return {
        "AWS_ACCESS_KEY_ID": creds.access_key_id,
        "AWS_SECRET_ACCESS_KEY": creds.secret_access_key,
        "AWS_SESSION_TOKEN": creds.session_token,
        "AWS_DEFAULT_REGION": region,
        "AWS_REGION": region,
    }
