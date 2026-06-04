from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.momentum_breakout_alert_models import AlertRiskGateResultDto
from app.models.strategy_models import _STRATEGY_MODEL_CONFIG

DEFAULT_MIN_HISTORICAL_PROFIT_FACTOR = 1.2
DEFAULT_MIN_HISTORICAL_TRADES = 20
DEFAULT_MAX_STOP_DISTANCE_PCT = 8.0


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
    stop_distance_pct: float = Field(alias="stopDistancePct")
    volume_ratio: float | None = Field(default=None, alias="volumeRatio")
    rs_percentile: float | None = Field(default=None, alias="rsPercentile")
    market_regime: str | None = Field(default=None, alias="marketRegime")
    risk_gate: AlertRiskGateResultDto = Field(alias="riskGate")


UNIVERSE_SOURCE_DESCRIPTION = (
    "ranking_pipeline.sqlite universe_members (active snapshot, passed_filters=1, "
    "ORDER BY symbol ASC) intersected with local OHLCV parquet (data/raw); "
    "capped by MB_SCAN_MAX_UNIVERSE (default 500)"
)


class MomentumBreakoutUniverseResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    total_available_symbols: int = Field(alias="totalAvailableSymbols")
    scan_cap: int = Field(alias="scanCap")
    symbols_scanned: int = Field(alias="symbolsScanned")
    excluded_symbols: int = Field(alias="excludedSymbols")
    universe_source: str = Field(alias="universeSource")
    sample_symbols: list[str] = Field(alias="sampleSymbols")


class MomentumBreakoutScanResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    scan_time: str = Field(alias="scanTime")
    total_symbols_scanned: int = Field(alias="totalSymbolsScanned")
    valid_setups_found: int = Field(alias="validSetupsFound")
    tradable_candidates_found: int = Field(alias="tradableCandidatesFound")
    blocked_candidates_count: int = Field(alias="blockedCandidatesCount")
    candidates_found: int = Field(
        alias="candidatesFound",
        description="Number of candidates returned in this response (after limit)",
    )
    candidates: list[MomentumBreakoutScanCandidateDto]
