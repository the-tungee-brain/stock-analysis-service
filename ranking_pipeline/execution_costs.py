"""Execution cost configuration (shared, no pipeline imports)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionCostConfig:
    """Realistic trading cost assumptions for simulated returns."""

    slippage_bps_per_side: float = 15.0
    round_trip_sides: int = 2
    liquidity_penalty_bps: float = 10.0
    min_adv_dollars: float = 20e6
