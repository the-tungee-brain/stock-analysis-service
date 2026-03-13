from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from app.services.schwab_auth_service import SchwabAuthService
from app.dependencies.service_dependencies import get_schwab_auth_service
from pydantic import BaseModel
import secrets

router = APIRouter()


class AuthSchwabConnectRequest(BaseModel):
    user_id: str


@router.post("connect")
def auth_schwab_connect(
    request: AuthSchwabConnectRequest,
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
):
    state = secrets.token_urlsafe(32)

    user_id = request.user_id
    schwab_auth_service.cache_state(state=state, user_id=user_id)
    auth_url = schwab_auth_service.build_authorization_url(state=state)
    return RedirectResponse(url=auth_url)
