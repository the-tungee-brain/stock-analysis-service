from fastapi import APIRouter, Depends

from app.auth.dependencies import get_current_user_id
from app.core.llm_config import settings
from app.core.llm_model_policy import is_paid_user

router = APIRouter()


@router.get("/account/plan")
def get_account_plan(user_id: str = Depends(get_current_user_id)):
    paid = is_paid_user(user_id)
    return {
        "plan": "pro" if paid else "free",
        "isPaid": paid,
        "freeModel": settings.OPENAI_FREE_MODEL,
        "defaultModel": settings.OPENAI_MODEL,
    }
