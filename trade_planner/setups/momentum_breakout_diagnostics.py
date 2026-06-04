"""Plain-language Momentum Breakout setup diagnostics for single-symbol checks."""

from __future__ import annotations

from dataclasses import dataclass

from trade_planner.config import MomentumBreakoutConfig
from trade_planner.indicators import (
    close_within_pct_of_period_high,
    relative_strength_percentile,
    simple_moving_average,
    volume_ratio,
)
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup
from trade_planner.types import StockData

RULE_INSUFFICIENT_HISTORY = "insufficient_price_history"
RULE_PRICE_BELOW_SMA50 = "price_below_sma50"
RULE_SMA50_BELOW_SMA200 = "sma50_below_sma200"
RULE_NOT_NEAR_20DAY_HIGH = "not_near_20day_high"
RULE_VOLUME_TOO_LOW = "volume_expansion_too_low"
RULE_RS_TOO_WEAK = "relative_strength_too_weak"
RULE_BENCHMARK_UNAVAILABLE = "benchmark_data_unavailable"

_PLAIN_MESSAGES: dict[str, str] = {
    RULE_INSUFFICIENT_HISTORY: "We need more daily price history to evaluate this stock.",
    RULE_PRICE_BELOW_SMA50: "Price is below the 50-day average.",
    RULE_SMA50_BELOW_SMA200: "The 50-day average is below the 200-day average (uptrend not intact).",
    RULE_NOT_NEAR_20DAY_HIGH: "Price is not close enough to the 20-day high.",
    RULE_VOLUME_TOO_LOW: "Volume expansion is too low versus the recent average.",
    RULE_RS_TOO_WEAK: "Relative strength versus the market is too weak.",
    RULE_BENCHMARK_UNAVAILABLE: "Benchmark data is unavailable for relative strength checks.",
}


@dataclass(frozen=True, slots=True)
class MomentumBreakoutSetupDiagnostics:
    setup_valid: bool
    failed_rule_ids: tuple[str, ...]

    @property
    def failed_setup_rules(self) -> list[str]:
        return [_PLAIN_MESSAGES[rule_id] for rule_id in self.failed_rule_ids]


def diagnose_momentum_breakout_setup(
    stock_data: StockData,
    setup: MomentumBreakoutSetup | None = None,
) -> MomentumBreakoutSetupDiagnostics:
    """Evaluate each setup rule independently (all failures returned)."""
    active_setup = setup or MomentumBreakoutSetup()
    cfg: MomentumBreakoutConfig = active_setup._config  # noqa: SLF001
    failed: list[str] = []

    window = active_setup._evaluation_window(stock_data)  # noqa: SLF001
    if window is None:
        return MomentumBreakoutSetupDiagnostics(
            setup_valid=False,
            failed_rule_ids=(RULE_INSUFFICIENT_HISTORY,),
        )

    price_series = [bar.close for bar in window]
    sma_fast = simple_moving_average(price_series, cfg.sma_fast_days)
    sma_slow = simple_moving_average(price_series, cfg.sma_slow_days)
    if sma_fast is None or sma_slow is None:
        return MomentumBreakoutSetupDiagnostics(
            setup_valid=False,
            failed_rule_ids=(RULE_INSUFFICIENT_HISTORY,),
        )

    current_close = window[-1].close
    if current_close <= sma_fast:
        failed.append(RULE_PRICE_BELOW_SMA50)
    if sma_fast <= sma_slow:
        failed.append(RULE_SMA50_BELOW_SMA200)

    if not close_within_pct_of_period_high(
        window,
        high_lookback_days=cfg.high_lookback_days,
        max_distance_pct=cfg.high_proximity_pct,
    ):
        failed.append(RULE_NOT_NEAR_20DAY_HIGH)

    vol = volume_ratio(window, cfg.volume_avg_days)
    if vol is None or vol < cfg.volume_ratio_min:
        failed.append(RULE_VOLUME_TOO_LOW)

    bench_window = active_setup._benchmark_window(stock_data)  # noqa: SLF001
    if cfg.require_benchmark and bench_window is None:
        failed.append(RULE_BENCHMARK_UNAVAILABLE)
    elif bench_window is not None:
        rs_pct = relative_strength_percentile(
            window,
            bench_window,
            rs_lookback=cfg.rs_lookback_days,
            percentile_window=cfg.rs_percentile_window,
        )
        if rs_pct is None or rs_pct < cfg.rs_percentile_min:
            failed.append(RULE_RS_TOO_WEAK)
    elif cfg.require_benchmark:
        failed.append(RULE_RS_TOO_WEAK)

    return MomentumBreakoutSetupDiagnostics(
        setup_valid=len(failed) == 0,
        failed_rule_ids=tuple(failed),
    )
