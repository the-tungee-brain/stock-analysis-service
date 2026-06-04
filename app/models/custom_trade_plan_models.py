from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.strategy_models import _STRATEGY_MODEL_CONFIG

CustomTradePlanDirection = Literal["LONG"]


class CustomTradePlanRequest(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    symbol: str
    direction: CustomTradePlanDirection = "LONG"
    account_equity_usd: float | None = Field(default=None, alias="accountEquityUsd")


class CustomTradePlanResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    symbol: str
    setup_name: str = Field(alias="setupName")
    direction: CustomTradePlanDirection = "LONG"
    entry_price: float = Field(alias="entryPrice")
    stop_price: float = Field(alias="stopPrice")
    target_price: float = Field(alias="targetPrice")
    risk_reward: float = Field(alias="riskReward")
    warnings: list[str]
    educational_only: bool = Field(default=True, alias="educationalOnly")
