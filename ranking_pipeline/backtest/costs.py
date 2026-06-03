"""Slippage and liquidity execution cost model."""

from __future__ import annotations

from ranking_pipeline.execution_costs import ExecutionCostConfig

__all__ = ["ExecutionCostConfig", "round_trip_slippage_fraction", "liquidity_penalty_fraction", "net_excess_return"]


def round_trip_slippage_fraction(config: ExecutionCostConfig) -> float:
    """Total slippage as fraction of notional (entry + exit)."""
    bps = config.slippage_bps_per_side * config.round_trip_sides
    return bps / 10_000.0


def liquidity_penalty_fraction(
    avg_dollar_volume_20d: float | None,
    config: ExecutionCostConfig,
) -> float:
    """Extra cost fraction for names below ADV threshold."""
    if avg_dollar_volume_20d is None or avg_dollar_volume_20d >= config.min_adv_dollars:
        return 0.0
    ratio = max(0.0, 1.0 - avg_dollar_volume_20d / config.min_adv_dollars)
    return (config.liquidity_penalty_bps / 10_000.0) * ratio


def net_excess_return(
    gross_excess: float,
    *,
    avg_dollar_volume_20d: float | None = None,
    config: ExecutionCostConfig | None = None,
) -> float:
    """Apply slippage and optional liquidity penalty to gross excess vs SPY."""
    cfg = config or ExecutionCostConfig()
    slip = round_trip_slippage_fraction(cfg)
    liq = liquidity_penalty_fraction(avg_dollar_volume_20d, cfg)
    return gross_excess - slip - liq
