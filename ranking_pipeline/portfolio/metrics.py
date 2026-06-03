"""Portfolio-level expected return and risk metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ranking_pipeline.portfolio.rebalancer import compute_turnover
from ranking_pipeline.portfolio.sizing import RankedCandidate


@dataclass(frozen=True)
class PortfolioRiskMetrics:
    expected_return_5d: float
    expected_excess_5d: float
    portfolio_volatility_proxy: float
    turnover: float
    concentration_hhi: float
    top_contributors: list[dict[str, float | str]]


def _hhi(weights: pd.Series) -> float:
    return float((weights**2).sum())


def compute_portfolio_metrics(
    weights: pd.Series,
    candidates: list[RankedCandidate],
    previous_weights: pd.Series,
) -> PortfolioRiskMetrics:
    by_symbol = {c.symbol: c for c in candidates}
    exp_ret = 0.0
    exp_excess = 0.0
    vol_proxy = 0.0
    contributors: list[dict[str, float | str]] = []

    for sym, w in weights.items():
        c = by_symbol.get(sym)
        if c is None:
            continue
        er = c.expected_excess_return or 0.0
        gross = er  # excess; add spy baseline only if needed for display
        exp_excess += w * er
        exp_ret += w * (er)
        atr = c.atr_14 or 1.0
        vol_proxy += w * float(atr)
        contributors.append(
            {
                "symbol": sym,
                "weight": float(w),
                "expected_excess_return": float(er),
                "contribution": float(w * er),
            }
        )

    contributors.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    turnover = compute_turnover(previous_weights, weights)

    return PortfolioRiskMetrics(
        expected_return_5d=float(exp_ret),
        expected_excess_5d=float(exp_excess),
        portfolio_volatility_proxy=float(vol_proxy),
        turnover=turnover,
        concentration_hhi=_hhi(weights),
        top_contributors=contributors[:10],
    )
