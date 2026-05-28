from fastapi import APIRouter, Depends

from app.auth.dependencies import get_current_user_id
from app.dependencies.service_dependencies import get_account_deletion_service
from app.services.account_deletion_service import AccountDeletionService

router = APIRouter()


@router.delete("/account")
def delete_account(
    user_id: str = Depends(get_current_user_id),
    account_deletion_service: AccountDeletionService = Depends(
        get_account_deletion_service
    ),
):
    account_deletion_service.delete_account(user_id)
    return {"deleted": True}
