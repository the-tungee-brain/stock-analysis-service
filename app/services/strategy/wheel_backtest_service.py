from __future__ import annotations

from datetime import date

import pandas as pd

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.models.wheel_backtest_models import (
    WheelBacktestAnnualRow,
    WheelBacktestCycle,
    WheelBacktestEquityPoint,
    WheelBacktestResponse,
    WheelBacktestTrade,
)
from app.services.strategy.wheel_backtest_engine import (
    ALLOWED_LOOKBACK_YEARS,
    PriceBar,
    WheelBacktestConfig,
    run_wheel_backtest,
)

_PERIOD_BY_YEARS = {5: "5y", 10: "10y", 15: "15y"}


class WheelBacktestService:
    def __init__(self, yfinance_adapter: YFinanceAdapter):
        self._yfinance = yfinance_adapter

    def run_backtest(
        self,
        symbol: str,
        *,
        lookback_years: int,
        target_delta_min: float = 0.20,
        target_delta_max: float = 0.30,
        dte_days: int = 30,
        contracts: int = 1,
        maintain_one_lot: bool = True,
    ) -> WheelBacktestResponse:
        symbol_upper = symbol.strip().upper()
        if lookback_years not in ALLOWED_LOOKBACK_YEARS:
            raise ValueError(
                f"lookback_years must be one of {sorted(ALLOWED_LOOKBACK_YEARS)}"
            )

        if target_delta_min <= 0 or target_delta_max <= 0:
            raise ValueError("target delta must be positive")
        if target_delta_min > target_delta_max:
            raise ValueError("target_delta_min cannot exceed target_delta_max")
        if dte_days < 1 or dte_days > 60:
            raise ValueError("dte_days must be between 1 and 60")
        if contracts < 1 or contracts > 20:
            raise ValueError("contracts must be between 1 and 20")

        target_delta = (target_delta_min + target_delta_max) / 2.0
        period = _PERIOD_BY_YEARS[lookback_years]

        hist = self._yfinance.get_history(
            symbol_upper,
            period=period,
            interval="1d",
            auto_adjust=True,
        )
        bars = _bars_from_history(hist)
        if not bars:
            raise ValueError(f"No price history for {symbol_upper}")

        # Split/dividend adjusted closes — do not apply splits or cash dividends again.
        dividends: dict[date, float] = {}
        splits: dict[date, float] = {}

        result = run_wheel_backtest(
            bars,
            dividends=dividends,
            splits=splits,
            config=WheelBacktestConfig(
                symbol=symbol_upper,
                lookback_years=lookback_years,
                target_delta=target_delta,
                dte_days=dte_days,
                contracts=contracts,
                maintain_one_lot=maintain_one_lot,
            ),
        )

        return WheelBacktestResponse(
            symbol=result.symbol,
            lookbackYears=result.lookback_years,
            startDate=result.start_date,
            historyStartDate=result.history_start_date,
            firstTradeDate=result.first_trade_date,
            lastTradeDate=result.last_trade_date,
            cspRounds=result.csp_rounds,
            endDate=result.end_date,
            tradingDays=result.trading_days,
            config=result.config,
            assumptions=result.assumptions,
            startingCashUsd=result.starting_cash_usd,
            endingEquityUsd=result.ending_equity_usd,
            totalPlUsd=result.total_pl_usd,
            totalReturnPct=result.total_return_pct,
            cagrPct=result.cagr_pct,
            buyAndHoldReturnPct=result.buy_and_hold_return_pct,
            buyAndHoldCagrPct=result.buy_and_hold_cagr_pct,
            totalPremiumCollectedUsd=result.total_premium_collected_usd,
            totalFeesUsd=result.total_fees_usd,
            totalDividendsUsd=result.total_dividends_usd,
            putAssignments=result.put_assignments,
            putsExpiredOtm=result.puts_expired_otm,
            callsAssigned=result.calls_assigned,
            callsExpiredOtm=result.calls_expired_otm,
            completedWheelCycles=result.completed_wheel_cycles,
            skippedTradesInsufficientCash=result.skipped_trades_insufficient_cash,
            capitalTopUpsUsd=result.capital_top_ups_usd,
            buyAndHoldEndingUsd=result.buy_and_hold_ending_usd,
            spotPriceAtStart=result.spot_price_at_start,
            spotPriceAtEnd=result.spot_price_at_end,
            initialStockPriceUsd=result.initial_stock_price_usd,
            initialPutStrikeUsd=result.initial_put_strike_usd,
            initialCollateralUsd=result.initial_collateral_usd,
            wheelCycles=[
                WheelBacktestCycle.model_validate(row) for row in result.wheel_cycles
            ],
            trades=[WheelBacktestTrade.model_validate(row) for row in result.trades],
            equityCurve=[
                WheelBacktestEquityPoint.model_validate(row)
                for row in result.equity_curve
            ],
            annualSummary=[
                WheelBacktestAnnualRow.model_validate(row)
                for row in result.annual_summary
            ],
        )


def _bars_from_history(hist: pd.DataFrame) -> list[PriceBar]:
    if hist.empty or "Close" not in hist.columns:
        return []

    bars: list[PriceBar] = []
    for ts, row in hist.iterrows():
        close = float(row["Close"])
        if close <= 0:
            continue
        trading_date = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        bars.append(PriceBar(trading_date=trading_date, close=close))
    return bars
