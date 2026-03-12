from fastapi import APIRouter
from fastapi.responses import RedirectResponse
import os

router = APIRouter()


def redirect_to_onboard(frontend_uri: str, status: str) -> RedirectResponse:
    return RedirectResponse(f"{frontend_uri}/onboard-complete?status={status}")


@router.get("/callback")
def auth_schwab_callback(code: str, state: str | None = None, error: str | None = None):
    powerpocket_frontend_uri = os.getenv("POWERPOCKET_FRONTEND_URI")
    if error is not None:
        return redirect_to_onboard(powerpocket_frontend_uri, "error")

    if code is None:
        return redirect_to_onboard(powerpocket_frontend_uri, "invalid")

    return redirect_to_onboard(powerpocket_frontend_uri, "success")
