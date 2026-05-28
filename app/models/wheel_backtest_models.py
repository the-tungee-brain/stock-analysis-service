from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from app.models.strategy_models import _STRATEGY_MODEL_CONFIG


class WheelBacktestTrade(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    date: str
    action: str
    label: str
    put_call: str | None = Field(default=None, alias="putCall")
    strike: float | None = None
    premium_usd: float = Field(default=0.0, alias="premiumUsd")
    fees_usd: float = Field(default=0.0, alias="feesUsd")
    close: float
    stock_price: float | None = Field(default=None, alias="stockPrice")
    effective_entry_price: float | None = Field(
        default=None, alias="effectiveEntryPrice"
    )
    effective_exit_price: float | None = Field(default=None, alias="effectiveExitPrice")
    premium_per_share: float | None = Field(default=None, alias="premiumPerShare")
    dte_days: int | None = Field(default=None, alias="dteDays")
    expiration_date: str | None = Field(default=None, alias="expirationDate")
    iv_percent: float | None = Field(default=None, alias="ivPercent")
    wheel_cycle: int | None = Field(default=None, alias="wheelCycle")
    cash_flow_usd: float | None = Field(default=None, alias="cashFlowUsd")
    note: str | None = None


class WheelBacktestCycle(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    cycle: int
    put_strike: float | None = Field(default=None, alias="putStrike")
    stock_entry_date: str | None = Field(default=None, alias="stockEntryDate")
    stock_entry_close: float | None = Field(default=None, alias="stockEntryClose")
    effective_entry_price: float | None = Field(
        default=None, alias="effectiveEntryPrice"
    )
    call_strike: float | None = Field(default=None, alias="callStrike")
    stock_exit_date: str | None = Field(default=None, alias="stockExitDate")
    stock_exit_close: float | None = Field(default=None, alias="stockExitClose")
    effective_exit_price: float | None = Field(default=None, alias="effectiveExitPrice")
    stock_round_trip_pl_usd: float | None = Field(
        default=None, alias="stockRoundTripPlUsd"
    )
    completed: bool = False


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
    pl_usd: float = Field(alias="plUsd")
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
    config: dict[str, float | int | bool]
    assumptions: list[str]
    starting_cash_usd: float = Field(alias="startingCashUsd")
    ending_equity_usd: float = Field(alias="endingEquityUsd")
    total_pl_usd: float = Field(alias="totalPlUsd")
    total_return_pct: float = Field(alias="totalReturnPct")
    cagr_pct: float | None = Field(default=None, alias="cagrPct")
    buy_and_hold_return_pct: float = Field(alias="buyAndHoldReturnPct")
    buy_and_hold_cagr_pct: float | None = Field(default=None, alias="buyAndHoldCagrPct")
    buy_and_hold_ending_usd: float = Field(alias="buyAndHoldEndingUsd")
    capital_top_ups_usd: float = Field(alias="capitalTopUpsUsd")
    spot_price_at_start: float = Field(alias="spotPriceAtStart")
    spot_price_at_end: float = Field(alias="spotPriceAtEnd")
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
    wheel_cycles: list[WheelBacktestCycle] = Field(alias="wheelCycles")
    trades: list[WheelBacktestTrade]
    equity_curve: list[WheelBacktestEquityPoint] = Field(alias="equityCurve")
    annual_summary: list[WheelBacktestAnnualRow] = Field(alias="annualSummary")
