from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from app.models.strategy_models import _STRATEGY_MODEL_CONFIG


class WheelBacktestTrade(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    date: str
    action: str
    put_call: str | None = Field(default=None, alias="putCall")
    strike: float | None = None
    premium_usd: float = Field(default=0.0, alias="premiumUsd")
    fees_usd: float = Field(default=0.0, alias="feesUsd")
    close: float
    note: str | None = None


class WheelBacktestEquityPoint(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    date: str
    equity_usd: float = Field(alias="equityUsd")
    cash_usd: float = Field(alias="cashUsd")
    shares: int
    phase: str


class WheelBacktestAnnualRow(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    year: int
    start_equity_usd: float = Field(alias="startEquityUsd")
    end_equity_usd: float = Field(alias="endEquityUsd")
    return_pct: float = Field(alias="returnPct")
    premium_usd: float = Field(alias="premiumUsd")
    fees_usd: float = Field(alias="feesUsd")


class WheelBacktestResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    symbol: str
    lookback_years: int = Field(alias="lookbackYears")
    start_date: date = Field(alias="startDate")
    end_date: date = Field(alias="endDate")
    trading_days: int = Field(alias="tradingDays")
    config: dict[str, float | int]
    assumptions: list[str]
    starting_cash_usd: float = Field(alias="startingCashUsd")
    ending_equity_usd: float = Field(alias="endingEquityUsd")
    total_return_pct: float = Field(alias="totalReturnPct")
    cagr_pct: float | None = Field(default=None, alias="cagrPct")
    buy_and_hold_return_pct: float = Field(alias="buyAndHoldReturnPct")
    buy_and_hold_cagr_pct: float | None = Field(default=None, alias="buyAndHoldCagrPct")
    total_premium_collected_usd: float = Field(alias="totalPremiumCollectedUsd")
    total_fees_usd: float = Field(alias="totalFeesUsd")
    total_dividends_usd: float = Field(alias="totalDividendsUsd")
    put_assignments: int = Field(alias="putAssignments")
    puts_expired_otm: int = Field(alias="putsExpiredOtm")
    calls_assigned: int = Field(alias="callsAssigned")
    calls_expired_otm: int = Field(alias="callsExpiredOtm")
    completed_wheel_cycles: int = Field(alias="completedWheelCycles")
    skipped_trades_insufficient_cash: int = Field(
        alias="skippedTradesInsufficientCash"
    )
    trades: list[WheelBacktestTrade]
    equity_curve: list[WheelBacktestEquityPoint] = Field(alias="equityCurve")
    annual_summary: list[WheelBacktestAnnualRow] = Field(alias="annualSummary")
