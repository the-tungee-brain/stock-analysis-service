from fastapi import APIRouter, Depends

from app.auth.dependencies import get_current_user
from app.core.llm_config import settings
from app.core.paid_access import is_paid_user
from app.core.plan_features import paid_features_for_user
from app.models.user_models import AppUserItem

router = APIRouter()


@router.get("/account/plan")
def get_account_plan(user: AppUserItem = Depends(get_current_user)):
    identity_sub = str(user.identity_sub)
    paid = is_paid_user(identity_sub)
    return {
        "plan": "pro" if paid else "free",
        "isPaid": paid,
        "identitySub": identity_sub,
        "email": user.email,
        "freeModel": settings.OPENAI_FREE_MODEL,
        "defaultModel": settings.OPENAI_MODEL,
        "features": paid_features_for_user(identity_sub),
    }
