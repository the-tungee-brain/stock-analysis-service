"""Daily signal and portfolio monitoring metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backtest.metrics import (
    UP_PROB_COLUMN,
    attach_ranking_score,
    compute_information_coefficient,
    compute_rank_ic,
)
from backtest.ranking_portfolio import PortfolioPeriod
from models.labels import EXCESS_RETURN_COLUMN, LABEL_HORIZON_DAYS


def compute_daily_cross_section_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute daily IC, rank IC, hit rate, breadth, and average score."""
    frame = attach_ranking_score(predictions.copy())
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.dropna(subset=[UP_PROB_COLUMN, EXCESS_RETURN_COLUMN])

    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("date", sort=True):
        if len(group) < 2:
            continue
        score = group[UP_PROB_COLUMN].astype("float64")
        target = group[EXCESS_RETURN_COLUMN].astype("float64")
        ic = float(score.corr(target)) if score.nunique() >= 2 and target.nunique() >= 2 else float("nan")
        rank_ic = (
            float(score.corr(target, method="spearman"))
            if score.nunique() >= 2 and target.nunique() >= 2
            else float("nan")
        )
        top_n = max(1, int(np.ceil(len(group) * 0.2)))
        top = group.sort_values(UP_PROB_COLUMN, ascending=False).head(top_n)
        rows.append(
            {
                "date": date,
                "n_symbols": len(group),
                "ic": ic,
                "rank_ic": rank_ic,
                "hit_rate": float((top[EXCESS_RETURN_COLUMN] > 0).mean()),
                "avg_rank_score": float(score.mean()),
                "top_quintile_excess_return": float(top[EXCESS_RETURN_COLUMN].mean()),
                "realized_excess_return": float(target.mean()),
            }
        )
    return pd.DataFrame(rows)


def attach_portfolio_turnover(
    daily_metrics: pd.DataFrame,
    periods: list[PortfolioPeriod],
) -> pd.DataFrame:
    """Map rebalance turnover to entry dates on the daily monitoring frame."""
    out = daily_metrics.copy()
    if out.empty:
        return out
    turnover_map = {pd.Timestamp(p.entry_date).normalize(): p.turnover for p in periods}
    out["turnover"] = out["date"].map(turnover_map).fillna(0.0)
    return out


def summarize_monitoring_window(
    daily_metrics: pd.DataFrame,
    *,
    window_days: int = 63,
) -> dict[str, Any]:
    """Summarize recent monitoring metrics for signal decay detection."""
    if daily_metrics.empty:
        return {
            "window_days": window_days,
            "n_days": 0,
            "ic": float("nan"),
            "rank_ic": float("nan"),
            "hit_rate": float("nan"),
            "avg_turnover": float("nan"),
            "avg_rank_score": float("nan"),
            "realized_excess_return": float("nan"),
        }

    recent = daily_metrics.tail(window_days)
    return {
        "window_days": window_days,
        "n_days": int(len(recent)),
        "ic": float(recent["ic"].mean()),
        "rank_ic": float(recent["rank_ic"].mean()),
        "hit_rate": float(recent["hit_rate"].mean()),
        "avg_turnover": float(recent["turnover"].mean()) if "turnover" in recent.columns else float("nan"),
        "avg_rank_score": float(recent["avg_rank_score"].mean()),
        "realized_excess_return": float(recent["realized_excess_return"].mean()),
        "positive_ic_days": float((recent["ic"] > 0).mean()),
    }


def compare_monitoring_to_baseline(
    daily_metrics: pd.DataFrame,
    *,
    recent_days: int = 63,
    baseline_days: int = 252,
) -> pd.DataFrame:
    """Compare recent monitoring window against prior baseline."""
    if daily_metrics.empty:
        return pd.DataFrame(
            columns=["metric", "recent", "baseline", "delta", "pct_change"]
        )

    recent = summarize_monitoring_window(daily_metrics, window_days=recent_days)
    baseline_slice = daily_metrics.iloc[:-recent_days].tail(baseline_days)
    baseline = summarize_monitoring_window(baseline_slice, window_days=len(baseline_slice))

    rows: list[dict[str, Any]] = []
    for key in ("ic", "rank_ic", "hit_rate", "avg_turnover", "avg_rank_score", "realized_excess_return"):
        recent_val = recent[key]
        baseline_val = baseline[key]
        delta = recent_val - baseline_val if pd.notna(recent_val) and pd.notna(baseline_val) else float("nan")
        pct = delta / abs(baseline_val) if pd.notna(delta) and baseline_val not in (0, float("nan")) else float("nan")
        rows.append(
            {
                "metric": key,
                "recent": recent_val,
                "baseline": baseline_val,
                "delta": delta,
                "pct_change": pct,
            }
        )
    return pd.DataFrame(rows)


def generate_monitoring_report(
    predictions: pd.DataFrame,
    periods: list[PortfolioPeriod] | None = None,
    *,
    recent_days: int = 63,
    baseline_days: int = 252,
) -> dict[str, Any]:
    """Build daily monitoring frame and decay comparison summary."""
    daily = compute_daily_cross_section_metrics(predictions)
    if periods:
        daily = attach_portfolio_turnover(daily, periods)

    return {
        "daily_metrics": daily,
        "overall_ic": compute_information_coefficient(predictions),
        "overall_rank_ic": compute_rank_ic(predictions),
        "recent_summary": summarize_monitoring_window(daily, window_days=recent_days),
        "baseline_comparison": compare_monitoring_to_baseline(
            daily,
            recent_days=recent_days,
            baseline_days=baseline_days,
        ),
        "hold_horizon_days": LABEL_HORIZON_DAYS,
    }
