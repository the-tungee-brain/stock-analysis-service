from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from app.models.strategy_models import _STRATEGY_MODEL_CONFIG
from trade_planner.alerts.paper_trade_models import (
    LIVE_PAPER_TRADING_DISCLAIMER,
    LIVE_PAPER_TRADING_LABEL,
)


class PaperTradePerformanceMetaDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    label: str = LIVE_PAPER_TRADING_LABEL
    disclaimer: str = LIVE_PAPER_TRADING_DISCLAIMER
    source: str = Field(default="LIVE_PAPER_TRADING", alias="source")


class PaperTradeSummaryDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    total_alerts: int = Field(alias="totalAlerts")
    triggered_alerts: int = Field(alias="triggeredAlerts")
    expired_alerts: int = Field(alias="expiredAlerts")
    win_rate: float | None = Field(default=None, alias="winRate")
    average_win: float | None = Field(default=None, alias="averageWin")
    average_loss: float | None = Field(default=None, alias="averageLoss")
    expectancy: float | None = None
    profit_factor: float | None = Field(default=None, alias="profitFactor")
    average_holding_days: float | None = Field(default=None, alias="averageHoldingDays")
    max_drawdown: float | None = Field(default=None, alias="maxDrawdown")
    current_open_trades: int = Field(alias="currentOpenTrades")


class PaperTradeBucketDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    key: str
    trade_count: int = Field(alias="tradeCount")
    win_rate: float | None = Field(default=None, alias="winRate")
    expectancy: float | None = None
    profit_factor: float | None = Field(default=None, alias="profitFactor")
    average_return_pct: float | None = Field(default=None, alias="averageReturnPct")


class PaperTradeRecordDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    alert_id: str = Field(alias="alertId")
    symbol: str
    setup_name: str = Field(alias="setupName")
    signal_date: date = Field(alias="signalDate")
    entry_triggered_at: datetime | None = Field(default=None, alias="entryTriggeredAt")
    entry_price: float = Field(alias="entryPrice")
    stop_price: float = Field(alias="stopPrice")
    target_price: float = Field(alias="targetPrice")
    exit_at: datetime | None = Field(default=None, alias="exitAt")
    exit_price: float | None = Field(default=None, alias="exitPrice")
    status: str
    outcome_return_pct: float | None = Field(default=None, alias="outcomeReturnPct")
    holding_days: int | None = Field(default=None, alias="holdingDays")
    risk_gate_action: str = Field(default="", alias="riskGateAction")
    market_regime: str | None = Field(default=None, alias="marketRegime")
    volume_ratio: float | None = Field(default=None, alias="volumeRatio")
    rs_percentile: float | None = Field(default=None, alias="rsPercentile")
    created_at: datetime = Field(alias="createdAt")


class PaperTradePerformanceSummaryResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    meta: PaperTradePerformanceMetaDto
    summary: PaperTradeSummaryDto
    by_risk_gate: list[PaperTradeBucketDto] = Field(default_factory=list, alias="byRiskGate")


class PaperTradePerformanceTradesResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    meta: PaperTradePerformanceMetaDto
    trades: list[PaperTradeRecordDto]


class PaperTradePerformanceBucketsResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    meta: PaperTradePerformanceMetaDto
    buckets: list[PaperTradeBucketDto]
