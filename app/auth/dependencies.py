from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from app.services.user_service import UserService
from app.auth.jwt_utils import verify_jwt
from app.dependencies.service_dependencies import get_user_service
from app.core.settings import JWT_SECRET_KEY, JWT_ALGORITHM
import jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    user_service: UserService = Depends(get_user_service),
):
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = verify_jwt(token)
        user_id: str | None = payload.get("sub")
        if user_id is None:
            raise credentials_exc
    except InvalidTokenError:
        raise credentials_exc

    user = user_service.get_user_by_identity_sub(identity_sub=user_id)
    if not user:
        raise credentials_exc

    return user


async def get_current_user_id(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        return user_id
    except jwt.PyJWTError:
        raise credentials_exception
