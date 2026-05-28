from __future__ import annotations

from fastapi import HTTPException

from app.core.llm_model_policy import is_paid_user

PRO_FEATURE_WHEEL_BACKTEST = "wheel_backtest"
PRO_FEATURE_DIVIDEND_SNOWBALL = "dividend_snowball"

PRO_FEATURES = frozenset(
    {
        PRO_FEATURE_WHEEL_BACKTEST,
        PRO_FEATURE_DIVIDEND_SNOWBALL,
    }
)


def paid_features_for_user(user_id: str) -> dict[str, bool]:
    paid = is_paid_user(user_id)
    return {feature: paid for feature in PRO_FEATURES}


def require_paid_feature(user_id: str, feature: str) -> None:
    if feature not in PRO_FEATURES:
        raise ValueError(f"Unknown pro feature: {feature}")
    if is_paid_user(user_id):
        return
    raise HTTPException(
        status_code=403,
        detail="Pro plan required for this feature. Upgrade in Settings.",
    )
