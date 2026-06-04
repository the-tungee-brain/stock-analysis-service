from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from app.models.strategy_models import _STRATEGY_MODEL_CONFIG


class OpenTradeSnapshotDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    symbol: str
    setup_name: str = Field(alias="setupName")
    entry_price: float = Field(alias="entryPrice")
    stop_price: float = Field(alias="stopPrice")
    direction: str = "LONG"
    position_risk_pct: float | None = Field(default=None, alias="positionRiskPct")


class ClosedTradeSnapshotDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    setup_name: str = Field(alias="setupName")
    return_pct: float = Field(alias="returnPct")
    symbol: str = ""


class AlertRiskSettingsDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    max_open_positions: int = Field(default=5, alias="maxOpenPositions")
    max_risk_per_trade_pct: float = Field(default=0.01, alias="maxRiskPerTradePct")
    max_total_open_risk_pct: float = Field(default=0.06, alias="maxTotalOpenRiskPct")
    consecutive_loss_limit: int = Field(default=4, alias="consecutiveLossLimit")
    rolling_window_trades: int = Field(default=20, alias="rollingWindowTrades")
    rolling_drawdown_limit_pct: float = Field(
        default=-0.10, alias="rollingDrawdownLimitPct"
    )
    mega_cap_correlation_threshold: int = Field(
        default=3, alias="megaCapCorrelationThreshold"
    )
    volume_climax_ratio: float = Field(default=3.0, alias="volumeClimaxRatio")


class MomentumBreakoutAlertRequest(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    symbol: str
    open_trades: list[OpenTradeSnapshotDto] = Field(
        default_factory=list, alias="openTrades"
    )
    recent_closed: list[ClosedTradeSnapshotDto] = Field(
        default_factory=list, alias="recentClosed"
    )
    sector_or_group: str | None = Field(default=None, alias="sectorOrGroup")
    market_regime: str | None = Field(default=None, alias="marketRegime")
    volume_ratio: float | None = Field(default=None, alias="volumeRatio")
    account_equity_usd: float | None = Field(default=None, alias="accountEquityUsd")
    risk_settings: AlertRiskSettingsDto | None = Field(default=None, alias="riskSettings")
    prior_price: float | None = Field(default=None, alias="priorPrice")
    persist_alert: bool = Field(default=True, alias="persistAlert")


class HistoricalStatsDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    total_trades: int | None = Field(default=None, alias="totalTrades")
    win_rate_pct: float | None = Field(default=None, alias="winRatePct")
    profit_factor: float | None = Field(default=None, alias="profitFactor")
    average_holding_days: float | None = Field(default=None, alias="averageHoldingDays")


class TradePlanLevelsDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    symbol: str
    setup_name: str = Field(alias="setupName")
    direction: str
    entry_price: float = Field(alias="entryPrice")
    stop_price: float = Field(alias="stopPrice")
    target_price: float = Field(alias="targetPrice")
    risk_reward: float = Field(alias="riskReward")
    confidence_score: float = Field(alias="confidenceScore")
    entry_is_stop: bool = Field(alias="entryIsStop")


class AlertRiskGateResultDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    allowed: bool
    action: str
    reasons: list[str]
    recommended_position_risk_pct: float = Field(alias="recommendedPositionRiskPct")
    max_notional_usd: float | None = Field(default=None, alias="maxNotionalUsd")
    alert_priority: str = Field(alias="alertPriority")
    educational_only: bool = Field(default=True, alias="educationalOnly")


class MomentumBreakoutAlertResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    disclaimer: str
    plan_available: bool = Field(alias="planAvailable")
    plan: TradePlanLevelsDto | None = None
    historical_stats: HistoricalStatsDto | None = Field(
        default=None, alias="historicalStats"
    )
    risk_gate: AlertRiskGateResultDto | None = Field(default=None, alias="riskGate")
    price_alerts: list[str] = Field(default_factory=list, alias="priceAlerts")
    alert_id: str | None = Field(default=None, alias="alertId")
    lifecycle_status: str | None = Field(default=None, alias="lifecycleStatus")


class AlertLifecycleEventDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    event_id: str = Field(alias="eventId")
    event_type: str = Field(alias="eventType")
    from_status: str | None = Field(default=None, alias="fromStatus")
    to_status: str = Field(alias="toStatus")
    price: float | None = None
    recorded_at: datetime = Field(alias="recordedAt")
    message: str


class MomentumBreakoutAlertRecordDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    alert_id: str = Field(alias="alertId")
    symbol: str
    setup_name: str = Field(alias="setupName")
    created_at: datetime = Field(alias="createdAt")
    signal_date: date = Field(alias="signalDate")
    entry_price: float = Field(alias="entryPrice")
    stop_price: float = Field(alias="stopPrice")
    target_price: float = Field(alias="targetPrice")
    entry_is_stop: bool = Field(alias="entryIsStop")
    status: str
    expires_at: datetime = Field(alias="expiresAt")
    triggered_at: datetime | None = Field(default=None, alias="triggeredAt")
    exit_at: datetime | None = Field(default=None, alias="exitAt")
    exit_price: float | None = Field(default=None, alias="exitPrice")
    outcome_return_pct: float | None = Field(default=None, alias="outcomeReturnPct")
    risk_gate_action: str = Field(alias="riskGateAction")
    risk_gate_reasons: list[str] = Field(alias="riskGateReasons")
    historical_win_rate: float | None = Field(default=None, alias="historicalWinRate")
    historical_profit_factor: float | None = Field(
        default=None, alias="historicalProfitFactor"
    )
    historical_total_trades: int | None = Field(
        default=None, alias="historicalTotalTrades"
    )
    lifecycle_events: list[AlertLifecycleEventDto] = Field(
        default_factory=list, alias="lifecycleEvents"
    )


class MomentumBreakoutAlertListResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    disclaimer: str
    alerts: list[MomentumBreakoutAlertRecordDto]


class AlertStatusChangeDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    alert_id: str = Field(alias="alertId")
    symbol: str
    prior_status: str = Field(alias="priorStatus")
    new_status: str = Field(alias="newStatus")


class MomentumBreakoutAlertRefreshResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    disclaimer: str
    processed: int
    updated: int
    skipped_market_hours: bool = Field(alias="skippedMarketHours")
    warnings: list[str]
    changes: list[AlertStatusChangeDto]
    alerts: list[MomentumBreakoutAlertRecordDto]


class MomentumBreakoutPriceUpdateRequest(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    symbol: str
    price: float
    timestamp: datetime | None = None
