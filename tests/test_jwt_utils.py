import jwt
import pytest

from app.auth.jwt_utils import create_access_token, verify_jwt
from app.core import settings


@pytest.fixture(autouse=True)
def _reset_jwt_signing_key_cache():
    settings.clear_jwt_signing_key_cache()
    yield
    settings.clear_jwt_signing_key_cache()


def test_short_jwt_secret_is_derived_to_32_bytes(monkeypatch):
    monkeypatch.setattr(settings, "JWT_SECRET_KEY", "short-20-byte-key!!")
    monkeypatch.setattr(settings, "_JWT_KEY_DERIVATION_LOGGED", False)
    signing_key = settings.resolve_jwt_signing_key(settings.JWT_SECRET_KEY)
    assert len(signing_key) == 32


def test_long_jwt_secret_is_used_directly(monkeypatch):
    secret = "a" * 32
    monkeypatch.setattr(settings, "JWT_SECRET_KEY", secret)
    signing_key = settings.resolve_jwt_signing_key(secret)
    assert signing_key == secret.encode("utf-8")


def test_create_and_verify_token_with_derived_key(monkeypatch):
    monkeypatch.setattr(settings, "JWT_SECRET_KEY", "short-20-byte-key!!")
    monkeypatch.setattr(settings, "_JWT_KEY_DERIVATION_LOGGED", False)

    token = create_access_token("user-123")
    payload = verify_jwt(token)
    assert payload["sub"] == "user-123"


def test_verify_rejects_tampered_token(monkeypatch):
    monkeypatch.setattr(settings, "JWT_SECRET_KEY", "short-20-byte-key!!")
    monkeypatch.setattr(settings, "_JWT_KEY_DERIVATION_LOGGED", False)
    settings.clear_jwt_signing_key_cache()

    token = create_access_token("user-123")
    header, payload, signature = token.split(".")
    idx = len(payload) // 2
    flipped = "A" if payload[idx] != "A" else "B"
    tampered = f"{header}.{payload[:idx]}{flipped}{payload[idx + 1:]}.{signature}"

    with pytest.raises(jwt.InvalidTokenError):
        verify_jwt(tampered)
