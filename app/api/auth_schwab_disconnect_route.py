from fastapi import APIRouter, Depends

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import get_schwab_auth_service
from app.services.schwab_auth_service import SchwabAuthService

router = APIRouter()


@router.delete("/disconnect")
def auth_schwab_disconnect(
    user_id: str = Depends(get_current_user_id),
    schwab_auth_service: SchwabAuthService = Depends(get_schwab_auth_service),
):
    schwab_auth_service.disconnect_user(user_id=user_id)
    return {"disconnected": True}
