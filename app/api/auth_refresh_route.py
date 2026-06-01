from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt import InvalidTokenError
from pydantic import BaseModel

from app.auth.jwt_utils import create_access_token, verify_jwt_for_refresh

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class AuthRefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/auth/refresh", response_model=AuthRefreshResponse)
def refresh_access_token(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = verify_jwt_for_refresh(token)
        user_id = payload.get("sub")
        if not user_id:
            raise credentials_exception
    except InvalidTokenError as exc:
        raise credentials_exception from exc

    return AuthRefreshResponse(access_token=create_access_token(user_id))
