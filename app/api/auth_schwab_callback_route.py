from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from app.services.schwab_auth_service import SchwabAuthService
from app.dependencies.service_dependencies import get_schwab_auth_service
import os

router = APIRouter()


def redirect_to_oauth_result(frontend_uri: str, status: str) -> RedirectResponse:
    return RedirectResponse(url=f"{frontend_uri}/oauth?status={status}")


@router.get("/callback")
def auth_schwab_callback(
    code: str,
    state: str | None = None,
    error: str | None = None,
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
):
    powerpocket_frontend_uri = os.getenv("POWERPOCKET_FRONTEND_URI")
    if error is not None:
        return redirect_to_oauth_result(powerpocket_frontend_uri, "error")

    if code is None:
        return redirect_to_oauth_result(powerpocket_frontend_uri, "invalid")

    user_id = schwab_auth_service.get_user_id_by_state(state=state)
    if not user_id:
        return redirect_to_oauth_result(powerpocket_frontend_uri, "error_state")

    schwab_auth_service.delete_cache(key=state)
    print("Getting access token: ", user_id, code, state)
    try:
        schwab_auth_service.claim_access_token(user_id=user_id, auth_code=code)
    except:
        return redirect_to_oauth_result(powerpocket_frontend_uri, "error_token")

    return redirect_to_oauth_result(powerpocket_frontend_uri, "success")
