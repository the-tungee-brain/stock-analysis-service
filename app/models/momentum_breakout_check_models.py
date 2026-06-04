from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.models.momentum_breakout_alert_models import AlertRiskGateResultDto
from app.models.strategy_models import _STRATEGY_MODEL_CONFIG

MomentumBreakoutCheckStatus = Literal[
    "TRADABLE_BREAKOUT",
    "REJECTED_BREAKOUT",
    "NO_BREAKOUT_SETUP",
    "DATA_UNAVAILABLE",
]


class MomentumBreakoutCheckResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    symbol: str
    status: MomentumBreakoutCheckStatus
    verdict_title: str = Field(alias="verdictTitle")
    verdict_message: str = Field(alias="verdictMessage")
    failed_setup_rules: list[str] = Field(default_factory=list, alias="failedSetupRules")
    rejection_reasons: list[str] = Field(default_factory=list, alias="rejectionReasons")
    entry_price: float | None = Field(default=None, alias="entryPrice")
    stop_price: float | None = Field(default=None, alias="stopPrice")
    target_price: float | None = Field(default=None, alias="targetPrice")
    stop_distance_pct: float | None = Field(default=None, alias="stopDistancePct")
    historical_win_rate: float | None = Field(default=None, alias="historicalWinRate")
    historical_profit_factor: float | None = Field(
        default=None, alias="historicalProfitFactor"
    )
    historical_total_trades: int | None = Field(
        default=None, alias="historicalTotalTrades"
    )
    risk_gate: AlertRiskGateResultDto | None = Field(default=None, alias="riskGate")
    can_track_breakout_plan: bool = Field(alias="canTrackBreakoutPlan")
