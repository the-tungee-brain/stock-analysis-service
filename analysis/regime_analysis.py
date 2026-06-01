"""Regime-conditioned signal diagnostics without model changes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.signal_diagnostics import decile_portfolio_analysis
from backtest.metrics import (
    UP_PROB_COLUMN,
    attach_ranking_score,
    compute_information_coefficient,
    compute_rank_ic,
    sharpe_ratio,
)
from data.loader import load_symbol
from models.labels import EXCESS_RETURN_COLUMN, LABEL_HORIZON_DAYS

TRADES_PER_YEAR = 252 / LABEL_HORIZON_DAYS
VIX_LOW_THRESHOLD = 15.0
VIX_HIGH_THRESHOLD = 25.0


def build_market_regime_frame(
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Daily SPY 200-DMA and VIX regime labels."""
    spy = load_symbol("SPY")["close"].astype("float64")
    vix = load_symbol("^VIX")["close"].astype("float64")

    frame = pd.DataFrame(index=pd.to_datetime(spy.index).normalize())
    frame["spy_close"] = spy.to_numpy()
    frame["vix_level"] = vix.reindex(frame.index).astype("float64")
    frame["spy_sma_200"] = frame["spy_close"].rolling(200, min_periods=200).mean()
    frame["spy_above_200dma"] = frame["spy_close"] > frame["spy_sma_200"]
    frame["market_regime"] = np.where(frame["spy_above_200dma"], "bull", "bear")
    frame["vix_regime"] = pd.cut(
        frame["vix_level"],
        bins=[-np.inf, VIX_LOW_THRESHOLD, VIX_HIGH_THRESHOLD, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)
    frame["spy_trend_regime"] = frame["market_regime"]

    if start is not None:
        frame = frame[frame.index >= pd.Timestamp(start).normalize()]
    if end is not None:
        frame = frame[frame.index <= pd.Timestamp(end).normalize()]
    return frame.dropna(subset=["spy_close", "vix_level"])


def attach_regime_columns(
    predictions: pd.DataFrame,
    regime_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join market regime labels onto prediction rows by date."""
    frame = attach_ranking_score(predictions.copy())
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    regimes = regime_frame if regime_frame is not None else build_market_regime_frame()
    joined = frame.merge(
        regimes[
            [
                "spy_above_200dma",
                "market_regime",
                "vix_regime",
                "spy_trend_regime",
            ]
        ].reset_index(names="date"),
        on="date",
        how="left",
    )
    return joined


def _quintile_spread_for_group(group: pd.DataFrame) -> float:
    if len(group) < 2:
        return float("nan")
    ranked = group.sort_values(UP_PROB_COLUMN, ascending=False)
    k = max(1, int(np.ceil(len(ranked) * 0.2)))
    top = ranked.head(k)[EXCESS_RETURN_COLUMN].astype("float64").mean()
    bottom = ranked.tail(k)[EXCESS_RETURN_COLUMN].astype("float64").mean()
    return float(top - bottom)


def summarize_regime_metrics(group: pd.DataFrame) -> dict[str, Any]:
    """IC, Sharpe proxy, and quintile spread for one regime subset."""
    clean = group.dropna(subset=[UP_PROB_COLUMN, EXCESS_RETURN_COLUMN])
    if clean.empty:
        return {
            "n_predictions": 0,
            "n_days": 0,
            "ic": float("nan"),
            "rank_ic": float("nan"),
            "sharpe": float("nan"),
            "quintile_spread": float("nan"),
        }

    spreads: list[float] = []
    for _, day_group in clean.groupby("date", sort=True):
        spread = _quintile_spread_for_group(day_group)
        if pd.notna(spread):
            spreads.append(spread)

    spread_series = pd.Series(spreads, dtype="float64")
    return {
        "n_predictions": int(len(clean)),
        "n_days": int(clean["date"].nunique()),
        "ic": compute_information_coefficient(clean),
        "rank_ic": compute_rank_ic(clean),
        "sharpe": sharpe_ratio(spread_series, periods_per_year=TRADES_PER_YEAR)
        if not spread_series.empty
        else float("nan"),
        "quintile_spread": float(spread_series.mean()) if not spread_series.empty else float("nan"),
    }


def analyze_regime_performance(
    predictions: pd.DataFrame,
    *,
    regime_frame: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Measure IC, Sharpe, and quintile spread by market regime."""
    frame = attach_regime_columns(predictions, regime_frame=regime_frame)
    frame = frame.dropna(subset=["market_regime", "vix_regime"])

    by_spy_trend: list[dict[str, Any]] = []
    for regime, group in frame.groupby("spy_trend_regime", sort=True):
        row = summarize_regime_metrics(group)
        row["regime"] = regime
        by_spy_trend.append(row)

    by_vix: list[dict[str, Any]] = []
    for regime, group in frame.groupby("vix_regime", sort=True):
        row = summarize_regime_metrics(group)
        row["regime"] = regime
        by_vix.append(row)

    by_spy_dma: list[dict[str, Any]] = []
    for above, group in frame.groupby("spy_above_200dma", sort=True):
        row = summarize_regime_metrics(group)
        row["regime"] = "above_200dma" if bool(above) else "below_200dma"
        by_spy_dma.append(row)

    return {
        "by_spy_trend": pd.DataFrame(by_spy_trend),
        "by_vix_regime": pd.DataFrame(by_vix),
        "by_spy_200dma": pd.DataFrame(by_spy_dma),
        "regime_frame": regime_frame if regime_frame is not None else build_market_regime_frame(),
    }
