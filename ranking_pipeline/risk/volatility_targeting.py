"""Realized volatility and weight scaling to a target."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def realized_portfolio_volatility(
    weights: pd.Series,
    returns: pd.DataFrame,
    *,
    annualize: bool = True,
) -> float:
    """Historical vol of weighted portfolio daily returns."""
    if returns.empty or weights.empty:
        return 0.0
    cols = [c for c in weights.index if c in returns.columns and weights[c] > 0]
    if not cols:
        return 0.0
    w = weights[cols] / weights[cols].sum()
    port_ret = (returns[cols] * w).sum(axis=1)
    daily_std = float(port_ret.std(ddof=1)) if len(port_ret) > 1 else 0.0
    if annualize:
        return daily_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return daily_std


def scale_weights_to_target_vol(
    weights: pd.Series,
    returns: pd.DataFrame,
    *,
    target_annual_vol: float = 0.12,
) -> tuple[pd.Series, float, float]:
    """
    Scale weights so realized portfolio vol approaches ``target_annual_vol``.

    Returns (scaled_weights, realized_vol_before, vol_scale_factor).
    """
    if weights.empty:
        return weights, 0.0, 1.0
    realized = realized_portfolio_volatility(weights, returns)
    if realized <= 1e-8:
        return weights, 0.0, 1.0

    factor = min(1.0, target_annual_vol / realized)
    scaled = weights * factor
    total = scaled.sum()
    if total > 0:
        scaled = scaled / total
    return scaled, realized, factor
