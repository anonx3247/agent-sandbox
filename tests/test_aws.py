"""Tests for AWS STS credential minting and env construction."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import typer

from agent_sandbox.aws import AwsRuntimeCreds, build_aws_env, mint_profile_creds


# --- fakes -----------------------------------------------------------------


class _FakeFrozen:
    def __init__(self, access_key: str, secret_key: str, token: str) -> None:
        self.access_key = access_key
        self.secret_key = secret_key
        self.token = token


class _FakeCredentials:
    def __init__(self, frozen: _FakeFrozen | None = None, raise_on_freeze: Exception | None = None) -> None:
        self._frozen = frozen
        self._raise = raise_on_freeze

    def get_frozen_credentials(self) -> _FakeFrozen | None:
        if self._raise is not None:
            raise self._raise
        return self._frozen


def _exc_module() -> SimpleNamespace:
    """A fake ``botocore.exceptions`` with real exception subclasses."""

    class ProfileNotFound(Exception):
        pass

    class SSOTokenLoadError(Exception):
        pass

    class UnauthorizedSSOTokenError(Exception):
        pass

    return SimpleNamespace(
        ProfileNotFound=ProfileNotFound,
        SSOTokenLoadError=SSOTokenLoadError,
        UnauthorizedSSOTokenError=UnauthorizedSSOTokenError,
    )


@pytest.fixture
def fake_boto3(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Install fake ``boto3``/``botocore.exceptions`` modules.

    Tests assign ``boto3.Session`` to drive each branch.
    """
    boto3 = SimpleNamespace()

    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "botocore.exceptions", _exc_module())
    yield boto3


def _session_returning(credentials: _FakeCredentials | None) -> type:
    class _FakeSession:
        def __init__(self, profile_name: str | None = None) -> None:
            self.profile_name = profile_name

        def get_credentials(self) -> _FakeCredentials | None:
            return credentials

    return _FakeSession


# --- build_aws_env ---------------------------------------------------------


def test_build_aws_env_returns_all_keys() -> None:
    creds = AwsRuntimeCreds(access_key_id="AKIA", secret_access_key="secret", session_token="token")

    env = build_aws_env(creds, region="eu-central-1")

    assert env == {
        "AWS_ACCESS_KEY_ID": "AKIA",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_SESSION_TOKEN": "token",
        "AWS_DEFAULT_REGION": "eu-central-1",
        "AWS_REGION": "eu-central-1",
    }


def test_build_aws_env_default_region_mirrors() -> None:
    env = build_aws_env(AwsRuntimeCreds("a", "b", "c"))

    assert env["AWS_DEFAULT_REGION"] == env["AWS_REGION"] == "us-west-2"


# --- mint_profile_creds happy path -----------------------------------------


def test_mint_profile_creds_returns_frozen(fake_boto3: SimpleNamespace) -> None:
    frozen = _FakeFrozen(access_key="AKIA", secret_key="secret", token="token")
    fake_boto3.Session = _session_returning(_FakeCredentials(frozen=frozen))

    creds = mint_profile_creds("dev")

    assert creds == AwsRuntimeCreds(access_key_id="AKIA", secret_access_key="secret", session_token="token")


# --- mint_profile_creds guard branches -------------------------------------


def test_mint_profile_creds_profile_not_found(fake_boto3: SimpleNamespace) -> None:
    not_found = sys.modules["botocore.exceptions"].ProfileNotFound

    def _raise(profile_name: str | None = None) -> None:
        raise not_found("nope")

    fake_boto3.Session = _raise

    with pytest.raises(typer.BadParameter, match="not found in ~/.aws/config"):
        mint_profile_creds("missing")


def test_mint_profile_creds_no_credentials(fake_boto3: SimpleNamespace) -> None:
    fake_boto3.Session = _session_returning(None)

    with pytest.raises(typer.BadParameter, match="no credentials configured"):
        mint_profile_creds("dev")


def test_mint_profile_creds_expired_sso_token(fake_boto3: SimpleNamespace) -> None:
    expired = sys.modules["botocore.exceptions"].UnauthorizedSSOTokenError("expired")
    fake_boto3.Session = _session_returning(_FakeCredentials(raise_on_freeze=expired))

    with pytest.raises(typer.BadParameter, match="expired or missing"):
        mint_profile_creds("dev")


def test_mint_profile_creds_static_creds_no_token(fake_boto3: SimpleNamespace) -> None:
    frozen = _FakeFrozen(access_key="AKIA", secret_key="secret", token="")
    fake_boto3.Session = _session_returning(_FakeCredentials(frozen=frozen))

    with pytest.raises(typer.BadParameter, match="static IAM credentials"):
        mint_profile_creds("dev")


def test_mint_profile_creds_incomplete_creds(fake_boto3: SimpleNamespace) -> None:
    frozen = _FakeFrozen(access_key="", secret_key="secret", token="token")
    fake_boto3.Session = _session_returning(_FakeCredentials(frozen=frozen))

    with pytest.raises(typer.BadParameter, match="incomplete credentials"):
        mint_profile_creds("dev")


# --- optional-dep path -----------------------------------------------------


def test_mint_profile_creds_missing_boto3(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure importing boto3 raises ImportError regardless of install state.
    monkeypatch.setitem(sys.modules, "boto3", None)

    with pytest.raises(typer.BadParameter, match="requires boto3"):
        mint_profile_creds("dev")
