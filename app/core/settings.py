import hashlib
import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = "HS256"
JWT_EXPIRES_MINUTES = 43_200  # 30 days

_JWT_KEY_DERIVATION_LOGGED = False


def resolve_jwt_signing_key(raw: str | None) -> bytes:
    global _JWT_KEY_DERIVATION_LOGGED

    if not raw:
        raise ValueError("JWT_SECRET_KEY environment variable is required")

    encoded = raw.encode("utf-8")
    if len(encoded) >= 32:
        return encoded

    if not _JWT_KEY_DERIVATION_LOGGED:
        logger.warning(
            "JWT_SECRET_KEY is %d bytes (minimum 32 recommended for HS256). "
            "Deriving signing key via SHA-256; existing tokens will be invalidated. "
            "Set a secret of at least 32 characters in production.",
            len(encoded),
        )
        _JWT_KEY_DERIVATION_LOGGED = True

    return hashlib.sha256(encoded).digest()


@lru_cache(maxsize=1)
def get_jwt_signing_key() -> bytes:
    return resolve_jwt_signing_key(JWT_SECRET_KEY)


def clear_jwt_signing_key_cache() -> None:
    get_jwt_signing_key.cache_clear()
