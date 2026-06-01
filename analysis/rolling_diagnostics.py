"""Rolling robustness diagnostics for signal stability."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.signal_diagnostics import compute_daily_ic_frame, decile_portfolio_analysis
from backtest.metrics import attach_ranking_score
from backtest.ranking_portfolio import summarize_portfolio_performance
from models.labels import EXCESS_RETURN_COLUMN, LABEL_HORIZON_DAYS

TRADING_DAYS_PER_YEAR = 252
DEFAULT_ROLLING_WINDOW_DAYS = TRADING_DAYS_PER_YEAR * 3


def compute_daily_ic_series(predictions: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Pearson IC time series."""
    frame = attach_ranking_score(predictions.copy())
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.dropna(subset=["ranking_score", EXCESS_RETURN_COLUMN])

    values: dict[pd.Timestamp, float] = {}
    for date, group in frame.groupby("date", sort=True):
        if len(group) < 2:
            continue
        score = group["ranking_score"].astype("float64")
        target = group[EXCESS_RETURN_COLUMN].astype("float64")
        if score.nunique() < 2 or target.nunique() < 2:
            continue
        ic = score.corr(target)
        if pd.notna(ic):
            values[pd.Timestamp(date).normalize()] = float(ic)
    return pd.Series(values).sort_index()


def compute_quintile_spread_series(
    predictions: pd.DataFrame,
    *,
    n_buckets: int = 5,
) -> pd.Series:
    """Daily top-minus-bottom bucket excess return spread."""
    analysis = decile_portfolio_analysis(predictions, n_buckets=n_buckets)
    spread_df = analysis["spread_by_date"]
    if spread_df.empty:
        return pd.Series(dtype="float64")
    series = spread_df.set_index(pd.to_datetime(spread_df["date"]).dt.normalize())["spread"]
    return series.astype("float64").sort_index()


def compute_rolling_portfolio_sharpe(
    period_frame: pd.DataFrame,
    *,
    hold_days: int = LABEL_HORIZON_DAYS,
    window_periods: int | None = None,
) -> pd.Series:
    """Rolling Sharpe on non-overlapping portfolio period returns."""
    if period_frame.empty:
        return pd.Series(dtype="float64")

    rets = period_frame["net_return"].astype("float64")
    periods_per_year = TRADING_DAYS_PER_YEAR / hold_days
    if window_periods is None:
        window_periods = max(3, int(round(3 * periods_per_year)))

    rolling_mean = rets.rolling(window_periods, min_periods=max(3, window_periods // 3)).mean()
    rolling_std = rets.rolling(window_periods, min_periods=max(3, window_periods // 3)).std(ddof=0)
    sharpe = rolling_mean / rolling_std * np.sqrt(periods_per_year)
    sharpe.index = pd.to_datetime(period_frame["entry_date"]).dt.normalize()
    return sharpe


def build_rolling_diagnostics(
    predictions: pd.DataFrame,
    period_frame: pd.DataFrame,
    *,
    hold_days: int = LABEL_HORIZON_DAYS,
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    quintile_buckets: int = 5,
) -> dict[str, Any]:
    """Rolling 3-year IC, Sharpe, and quintile spread diagnostics."""
    daily_ic_frame = compute_daily_ic_frame(predictions)
    daily_ic = daily_ic_frame.set_index("date")["ic"] if not daily_ic_frame.empty else pd.Series(dtype="float64")
    daily_rank_ic = (
        daily_ic_frame.set_index("date")["rank_ic"] if not daily_ic_frame.empty else pd.Series(dtype="float64")
    )
    quintile_spread = compute_quintile_spread_series(predictions, n_buckets=quintile_buckets)
    rolling_ic = daily_ic.rolling(rolling_window_days, min_periods=rolling_window_days // 4).mean()
    rolling_rank_ic = daily_rank_ic.rolling(rolling_window_days, min_periods=rolling_window_days // 4).mean()
    rolling_spread = quintile_spread.rolling(
        rolling_window_days,
        min_periods=rolling_window_days // 4,
    ).mean()
    rolling_sharpe = compute_rolling_portfolio_sharpe(period_frame, hold_days=hold_days)

    latest_ic = float(rolling_ic.dropna().iloc[-1]) if not rolling_ic.dropna().empty else float("nan")
    latest_spread = float(rolling_spread.dropna().iloc[-1]) if not rolling_spread.dropna().empty else float("nan")
    latest_sharpe = float(rolling_sharpe.dropna().iloc[-1]) if not rolling_sharpe.dropna().empty else float("nan")

    portfolio_summary = summarize_portfolio_performance(period_frame, hold_days=hold_days)
    trend = _classify_signal_trend(latest_ic, rolling_ic)

    return {
        "rolling_window_days": rolling_window_days,
        "daily_ic": daily_ic,
        "rolling_ic": rolling_ic,
        "rolling_rank_ic": rolling_rank_ic,
        "quintile_spread": quintile_spread,
        "rolling_quintile_spread": rolling_spread,
        "rolling_sharpe": rolling_sharpe,
        "latest_rolling_ic": latest_ic,
        "latest_rolling_quintile_spread": latest_spread,
        "latest_rolling_sharpe": latest_sharpe,
        "full_sample_sharpe": portfolio_summary["sharpe_ratio"],
        "signal_trend": trend,
    }


def _classify_signal_trend(latest_ic: float, rolling_ic: pd.Series) -> str:
    """Classify whether rolling IC is improving, stable, or decaying."""
    clean = rolling_ic.dropna()
    if clean.empty or pd.isna(latest_ic):
        return "insufficient_data"
    if len(clean) < 20:
        return "insufficient_data"

    first_half = float(clean.iloc[: len(clean) // 2].mean())
    second_half = float(clean.iloc[len(clean) // 2 :].mean())
    delta = second_half - first_half
    if delta > 0.005:
        return "improving"
    if delta < -0.005:
        return "decaying"
    return "stable"
