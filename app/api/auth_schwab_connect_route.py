from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from app.services.schwab_auth_service import SchwabAuthService
from app.dependencies.service_dependencies import get_schwab_auth_service
from app.auth.dependencies import get_current_user_id
import secrets

router = APIRouter()


@router.get("/connect")
def auth_schwab_connect(
    user_id: str = Depends(get_current_user_id),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
):
    state = secrets.token_urlsafe(32)
    schwab_auth_service.cache_state(state=state, user_id=user_id)
    auth_url = schwab_auth_service.build_authorization_url(state=state)
    return RedirectResponse(url=auth_url)
