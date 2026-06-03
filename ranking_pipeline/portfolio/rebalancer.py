"""Daily rebalance, smoothing, and trade list generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class TradeSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class RebalanceTrade:
    symbol: str
    side: TradeSide
    weight_change: float
    target_weight: float
    previous_weight: float


def smooth_weights(
    target: pd.Series,
    previous: pd.Series,
    *,
    alpha: float = 0.3,
) -> pd.Series:
    """
    Exponential smoothing: w = (1-α)*w_prev + α*w_target.

    Default α=0.3 → 70% prior, 30% new (per spec).
    """
    if previous.empty:
        return target
    all_symbols = sorted(set(target.index) | set(previous.index))
    w_prev = previous.reindex(all_symbols).fillna(0.0)
    w_tgt = target.reindex(all_symbols).fillna(0.0)
    blended = (1.0 - alpha) * w_prev + alpha * w_tgt
    blended = blended.clip(lower=0.0)
    total = blended.sum()
    if total > 0:
        blended = blended / total
    return blended[blended > 1e-8]


def compute_turnover(previous: pd.Series, current: pd.Series) -> float:
    all_symbols = sorted(set(previous.index) | set(current.index))
    w0 = previous.reindex(all_symbols).fillna(0.0)
    w1 = current.reindex(all_symbols).fillna(0.0)
    return float((w1 - w0).abs().sum())


def compute_trades(
    previous: pd.Series,
    current: pd.Series,
    *,
    min_weight_change: float = 1e-4,
) -> list[RebalanceTrade]:
    """Emit buy/sell/hold rows from weight deltas."""
    all_symbols = sorted(set(previous.index) | set(current.index))
    w_prev = previous.reindex(all_symbols).fillna(0.0)
    w_cur = current.reindex(all_symbols).fillna(0.0)
    trades: list[RebalanceTrade] = []
    for sym in all_symbols:
        prev_w = float(w_prev[sym])
        cur_w = float(w_cur[sym])
        delta = cur_w - prev_w
        if abs(delta) < min_weight_change and prev_w == cur_w:
            if cur_w > 0:
                trades.append(
                    RebalanceTrade(sym, TradeSide.HOLD, 0.0, cur_w, prev_w)
                )
            continue
        if delta > min_weight_change:
            side = TradeSide.BUY
        elif delta < -min_weight_change:
            side = TradeSide.SELL
        else:
            side = TradeSide.HOLD
        trades.append(
            RebalanceTrade(
                symbol=sym,
                side=side,
                weight_change=delta,
                target_weight=cur_w,
                previous_weight=prev_w,
            )
        )
    return trades


def daily_rebalance(
    target: pd.Series,
    previous: pd.Series,
    *,
    smoothing_alpha: float,
) -> pd.Series:
    """Smooth then return final weights (constraints applied separately)."""
    return smooth_weights(target, previous, alpha=smoothing_alpha)
