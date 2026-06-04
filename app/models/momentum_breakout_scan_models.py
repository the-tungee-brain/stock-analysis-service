from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.momentum_breakout_alert_models import AlertRiskGateResultDto
from app.models.strategy_models import _STRATEGY_MODEL_CONFIG


class MomentumBreakoutScanCandidateDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    symbol: str
    entry_price: float = Field(alias="entryPrice")
    stop_price: float = Field(alias="stopPrice")
    target_price: float = Field(alias="targetPrice")
    risk_reward: float = Field(alias="riskReward")
    historical_win_rate: float | None = Field(default=None, alias="historicalWinRate")
    historical_profit_factor: float | None = Field(
        default=None, alias="historicalProfitFactor"
    )
    historical_total_trades: int | None = Field(
        default=None, alias="historicalTotalTrades"
    )
    setup_score: float = Field(alias="setupScore")
    volume_ratio: float | None = Field(default=None, alias="volumeRatio")
    rs_percentile: float | None = Field(default=None, alias="rsPercentile")
    market_regime: str | None = Field(default=None, alias="marketRegime")
    risk_gate: AlertRiskGateResultDto = Field(alias="riskGate")


class MomentumBreakoutScanResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    scan_time: str = Field(alias="scanTime")
    total_symbols_scanned: int = Field(alias="totalSymbolsScanned")
    candidates_found: int = Field(alias="candidatesFound")
    candidates: list[MomentumBreakoutScanCandidateDto]
