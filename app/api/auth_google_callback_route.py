from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from app.core.settings import GOOGLE_CLIENT_ID
from app.auth.jwt_utils import create_access_token
from app.services.user_service import UserService
from app.dependencies.service_dependencies import get_user_service
from app.models.user_models import IdentityPayload

router = APIRouter(prefix="/api/v1/auth/google", tags=["auth-google"])


class GoogleSignInRequest(BaseModel):
    id_token: str


class GoogleSignInResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/callback", response_model=GoogleSignInResponse)
async def auth_google_callback(
    payload: GoogleSignInRequest, user_service: UserService = Depends(get_user_service)
):
    try:
        idinfo = id_token.verify_oauth2_token(
            payload.id_token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token",
        )

    sub = idinfo.get("sub")
    email = idinfo.get("email")
    name = idinfo.get("name")
    picture = idinfo.get("picture")

    if not sub or not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing required Google profile fields",
        )

    user = user_service.create_or_link_user(
        payload=IdentityPayload(
            identity_sub=sub,
            identity_provider="google",
            email=email,
            full_name=name,
            avatar_url=picture,
        )
    )

    access_token = create_access_token(
        user_id=str(user.identity_sub),
        extra={"email": user.email},
    )

    return GoogleSignInResponse(access_token=access_token)
