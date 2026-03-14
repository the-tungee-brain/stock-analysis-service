from fastapi import APIRouter, Depends
from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import get_schwab_auth_service
from app.services.schwab_auth_service import SchwabAuthService

router = APIRouter()


@router.get("/status")
def auth_schwab_status(
    user_id: str = Depends(get_current_user_id),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
):
    authorized = schwab_auth_service.is_schwab_authorized(user_id=user_id)
    return {"authorized": authorized}
