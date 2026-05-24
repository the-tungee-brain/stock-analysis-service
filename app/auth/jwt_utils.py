from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt
from app.core.settings import JWT_ALGORITHM, JWT_EXPIRES_MINUTES, get_jwt_signing_key
from jwt import ExpiredSignatureError, InvalidTokenError


def create_access_token(user_id: str, extra: Dict[str, Any] | None = None) -> str:
    data: Dict[str, Any] = {"sub": user_id}
    if extra:
        data.update(extra)
    exp = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRES_MINUTES)
    data["exp"] = exp
    return jwt.encode(data, get_jwt_signing_key(), algorithm=JWT_ALGORITHM)


def verify_jwt(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            get_jwt_signing_key(),
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except ExpiredSignatureError as e:
        raise InvalidTokenError("Token has expired") from e
    except InvalidTokenError:
        raise

    return payload
