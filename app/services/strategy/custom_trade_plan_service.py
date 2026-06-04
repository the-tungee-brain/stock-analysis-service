"""Educational custom trade plans (not Momentum Breakout)."""

from __future__ import annotations

from data.benchmarks import BENCHMARK_SYMBOL
from data.loader import load_symbol
from trade_planner.config import TradePlannerConfig
from trade_planner.indicators import (
    average_true_range,
    highest_high,
    prior_lowest_low,
    simple_moving_average,
    volume_expansion_ratio,
)
from trade_planner.research.data import align_benchmark_to_stock, ohlcv_bars_from_dataframe
from trade_planner.setups.base import long_target_from_rr
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup
from trade_planner.setups.momentum_breakout_diagnostics import diagnose_momentum_breakout_setup
from trade_planner.types import StockData

from app.models.custom_trade_plan_models import CustomTradePlanRequest, CustomTradePlanResponse

_CUSTOM_SETUP_NAME = "Custom Trade Plan"
_TARGET_RR = 2.0
_ATR_STOP_MULTIPLE = 1.5
_SWING_LOOKBACK = 10
_HIGH_LOOKBACK = 20
_MAX_STOP_DISTANCE_PCT = 12.0
_HIGH_VOL_RATIO = 2.5
_WEAK_TREND_SPREAD_PCT = 0.01


class CustomTradePlanService:
    def __init__(self, *, momentum_setup: MomentumBreakoutSetup | None = None) -> None:
        self._momentum = momentum_setup or MomentumBreakoutSetup(
            TradePlannerConfig().momentum
        )

    def generate(self, request: CustomTradePlanRequest) -> CustomTradePlanResponse:
        if request.direction != "LONG":
            raise ValueError("Only LONG direction is supported in v1")

        sym = request.symbol.strip().upper()
        if not sym:
            raise ValueError("symbol is required")

        stock_df = load_symbol(sym)
        bench_df = load_symbol(BENCHMARK_SYMBOL)
        stock_bars = ohlcv_bars_from_dataframe(stock_df)
        bench_bars = align_benchmark_to_stock(
            stock_bars, ohlcv_bars_from_dataframe(bench_df)
        )
        data = StockData.from_bars(sym, stock_bars, benchmark_bars=bench_bars)

        if len(stock_bars) < 60:
            raise ValueError("Insufficient price history for a custom educational plan")

        last = stock_bars[-1]
        period_high = highest_high(stock_bars, _HIGH_LOOKBACK) or last.high
        entry = round(max(last.close, period_high) + 0.01, 4)

        swing_stop = prior_lowest_low(stock_bars, _SWING_LOOKBACK)
        atr = average_true_range(stock_bars, 14)
        atr_stop = round(last.close - _ATR_STOP_MULTIPLE * atr, 4) if atr else None

        stop_candidates = [price for price in (swing_stop, atr_stop) if price and price > 0]
        if not stop_candidates:
            raise ValueError("Could not derive a stop level for this symbol")

        stop = round(max(stop_candidates), 4)
        if stop >= entry:
            stop = round(entry * 0.95, 4)

        target = long_target_from_rr(entry, stop, _TARGET_RR)
        if target is None:
            raise ValueError("Could not derive a target level for this symbol")
        target = round(target, 4)

        stop_distance_pct = (
            abs(entry - stop) / entry * 100.0 if entry > 0 else 0.0
        )
        risk_reward = (
            abs(target - entry) / abs(entry - stop) if entry > stop else _TARGET_RR
        )

        warnings = self._build_warnings(
            data=data,
            stop_distance_pct=stop_distance_pct,
            entry=entry,
        )

        return CustomTradePlanResponse(
            symbol=sym,
            setupName=_CUSTOM_SETUP_NAME,
            direction="LONG",
            entryPrice=entry,
            stopPrice=stop,
            targetPrice=target,
            riskReward=round(risk_reward, 2),
            warnings=warnings,
            educationalOnly=True,
        )

    def _build_warnings(
        self,
        *,
        data: StockData,
        stop_distance_pct: float,
        entry: float,
    ) -> list[str]:
        warnings: list[str] = []
        closes = [bar.close for bar in data.bars[-220:]]
        sma50 = simple_moving_average(closes, 50)
        sma200 = simple_moving_average(closes, 200)
        if sma50 is not None and sma200 is not None and sma200 > 0:
            spread = (sma50 - sma200) / sma200
            if spread < _WEAK_TREND_SPREAD_PCT or closes[-1] < sma50:
                warnings.append("The intermediate trend still looks weak for this stock.")

        if stop_distance_pct > _MAX_STOP_DISTANCE_PCT:
            warnings.append(
                f"The suggested stop is relatively wide ({stop_distance_pct:.1f}% below entry)."
            )

        vol_ratio = volume_expansion_ratio(data.bars, 20)
        if vol_ratio is not None and vol_ratio >= _HIGH_VOL_RATIO:
            warnings.append(
                "Recent volume is elevated, which can increase volatility around the entry."
            )

        mb_diag = diagnose_momentum_breakout_setup(data, self._momentum)
        if not mb_diag.setup_valid:
            warnings.append(
                "A Momentum Breakout setup is not active on this symbol today — "
                "this plan is a separate custom educational outline."
            )

        _ = entry
        deduped: list[str] = []
        seen: set[str] = set()
        for item in warnings:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped
