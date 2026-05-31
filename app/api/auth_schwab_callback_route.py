from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from app.services.schwab_auth_service import SchwabAuthService
from app.dependencies.service_dependencies import get_schwab_auth_service, get_transaction_service
from app.services.schwab_auth_service import SchwabAuthService
from app.services.transaction_service import TransactionService
import os
import traceback

router = APIRouter()


def redirect_to_oauth_result(
    frontend_uri: str,
    status: str,
    *,
    path: str = "/portfolio",
) -> RedirectResponse:
    return RedirectResponse(url=f"{frontend_uri}{path}?status={status}")


def redirect_for_oauth_client(
    *,
    oauth_client: str,
    status: str,
    frontend_uri: str,
) -> RedirectResponse:
    if oauth_client == "ios":
        ios_uri = os.getenv("POWERPOCKET_IOS_OAUTH_REDIRECT_URI", "tomcrest://schwab")
        return RedirectResponse(url=f"{ios_uri}?status={status}")

    path = "/portfolio" if status == "success" else "/settings"
    return redirect_to_oauth_result(frontend_uri, status, path=path)


@router.get("/callback")
def auth_schwab_callback(
    code: str,
    state: str | None = None,
    error: str | None = None,
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
    transaction_service: TransactionService = Depends(get_transaction_service),
):
    powerpocket_frontend_uri = os.getenv("POWERPOCKET_FRONTEND_URI", "")
    oauth_state = schwab_auth_service.get_oauth_state(state=state) if state else None
    oauth_client = oauth_state.client if oauth_state else "web"

    if error is not None:
        return redirect_for_oauth_client(
            oauth_client=oauth_client,
            status="error",
            frontend_uri=powerpocket_frontend_uri,
        )

    if code is None:
        return redirect_for_oauth_client(
            oauth_client=oauth_client,
            status="invalid",
            frontend_uri=powerpocket_frontend_uri,
        )

    if not oauth_state:
        return redirect_for_oauth_client(
            oauth_client=oauth_client,
            status="error_state",
            frontend_uri=powerpocket_frontend_uri,
        )

    user_id = oauth_state.user_id
    schwab_auth_service.delete_cache(key=f"oauth:{state}")
    print("Getting access token: ", user_id, code, state)
    try:
        schwab_auth_service.claim_access_token(user_id=user_id, auth_code=code)
        transaction_service.invalidate_recent_orders_cache(user_id=user_id)
    except Exception as e:
        print("Error in callback:" + user_id + ":" + code, e, flush=True)
        traceback.print_exc()
        return redirect_for_oauth_client(
            oauth_client=oauth_client,
            status="error_token",
            frontend_uri=powerpocket_frontend_uri,
        )

    return redirect_for_oauth_client(
        oauth_client=oauth_client,
        status="success",
        frontend_uri=powerpocket_frontend_uri,
    )
