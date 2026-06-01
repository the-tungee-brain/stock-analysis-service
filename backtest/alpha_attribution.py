"""Per-symbol alpha attribution for ranking portfolios."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from backtest.metrics import UP_PROB_COLUMN, compute_information_coefficient, compute_rank_ic
from backtest.ranking_portfolio import PortfolioPeriod, _weights_for_symbols
from models.labels import EXCESS_RETURN_COLUMN


def attribute_portfolio_periods(
    periods: list[PortfolioPeriod],
    panel: pd.DataFrame,
    *,
    return_col: str = EXCESS_RETURN_COLUMN,
) -> pd.DataFrame:
    """Aggregate weighted return contribution by symbol across portfolio periods."""
    if not periods:
        return pd.DataFrame(
            columns=[
                "symbol",
                "periods_held",
                "long_periods",
                "short_periods",
                "gross_return_contribution",
                "contribution_share",
            ]
        )

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    contributions: dict[str, dict[str, float | int]] = {}
    total_gross = 0.0

    for period in periods:
        cross_section = frame[frame["date"] == period.entry_date]
        if cross_section.empty:
            continue

        if period.short_symbols:
            long_weights = _weights_for_symbols(list(period.long_symbols), gross=0.5)
            short_weights = {
                symbol: -weight
                for symbol, weight in _weights_for_symbols(list(period.short_symbols), gross=0.5).items()
            }
            weights = {**long_weights, **short_weights}
        else:
            weights = _weights_for_symbols(list(period.long_symbols), gross=1.0)

        for symbol, weight in weights.items():
            row = cross_section[cross_section["symbol"] == symbol]
            if row.empty:
                continue
            symbol_return = float(row.iloc[0][return_col])
            contrib = weight * symbol_return
            bucket = contributions.setdefault(
                symbol,
                {
                    "periods_held": 0,
                    "long_periods": 0,
                    "short_periods": 0,
                    "gross_return_contribution": 0.0,
                },
            )
            bucket["periods_held"] = int(bucket["periods_held"]) + 1
            bucket["gross_return_contribution"] = float(bucket["gross_return_contribution"]) + contrib
            total_gross += contrib
            if weight >= 0:
                bucket["long_periods"] = int(bucket["long_periods"]) + 1
            else:
                bucket["short_periods"] = int(bucket["short_periods"]) + 1

    rows: list[dict[str, Any]] = []
    for symbol, stats in contributions.items():
        share = (
            float(stats["gross_return_contribution"]) / total_gross
            if total_gross != 0
            else float("nan")
        )
        rows.append(
            {
                "symbol": symbol,
                "periods_held": int(stats["periods_held"]),
                "long_periods": int(stats["long_periods"]),
                "short_periods": int(stats["short_periods"]),
                "gross_return_contribution": float(stats["gross_return_contribution"]),
                "contribution_share": share,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("gross_return_contribution", ascending=False).reset_index(drop=True)


def attribute_ic_by_symbol(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol time-series IC and leave-one-out daily IC impact."""
    if predictions.empty:
        return pd.DataFrame(columns=["symbol", "time_series_ic", "time_series_rank_ic", "loo_ic_delta"])

    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    baseline_ic = compute_information_coefficient(frame)
    rows: list[dict[str, Any]] = []

    for symbol in sorted(frame["symbol"].astype(str).unique()):
        subset = frame[frame["symbol"] == symbol]
        score = subset[UP_PROB_COLUMN].astype("float64")
        target = subset[EXCESS_RETURN_COLUMN].astype("float64")
        ts_ic = float(score.corr(target)) if len(subset) >= 3 else float("nan")
        ts_rank_ic = float(score.corr(target, method="spearman")) if len(subset) >= 3 else float("nan")
        without = frame[frame["symbol"] != symbol]
        loo_ic = compute_information_coefficient(without) if len(without) >= 10 else float("nan")
        rows.append(
            {
                "symbol": symbol,
                "time_series_ic": ts_ic,
                "time_series_rank_ic": ts_rank_ic,
                "loo_ic_delta": float(baseline_ic - loo_ic) if pd.notna(loo_ic) else float("nan"),
            }
        )

    return pd.DataFrame(rows).sort_values("time_series_ic", ascending=False).reset_index(drop=True)


def attribute_drawdown_contribution(
    periods: list[PortfolioPeriod],
    panel: pd.DataFrame,
    *,
    return_col: str = EXCESS_RETURN_COLUMN,
) -> pd.DataFrame:
    """Attribute portfolio drawdown periods to symbol weighted losses."""
    if not periods:
        return pd.DataFrame(columns=["symbol", "loss_contribution", "loss_share"])

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    symbol_losses: dict[str, float] = {}
    total_loss = 0.0

    for period in periods:
        if period.gross_return >= 0:
            continue
        cross_section = frame[frame["date"] == period.entry_date]
        if cross_section.empty:
            continue

        if period.short_symbols:
            long_weights = _weights_for_symbols(list(period.long_symbols), gross=0.5)
            short_weights = {
                symbol: -weight
                for symbol, weight in _weights_for_symbols(list(period.short_symbols), gross=0.5).items()
            }
            weights = {**long_weights, **short_weights}
        else:
            weights = _weights_for_symbols(list(period.long_symbols), gross=1.0)

        for symbol, weight in weights.items():
            row = cross_section[cross_section["symbol"] == symbol]
            if row.empty:
                continue
            contrib = weight * float(row.iloc[0][return_col])
            if contrib >= 0:
                continue
            symbol_losses[symbol] = symbol_losses.get(symbol, 0.0) + contrib
            total_loss += contrib

    rows = [
        {
            "symbol": symbol,
            "loss_contribution": loss,
            "loss_share": float(loss / total_loss) if total_loss != 0 else float("nan"),
        }
        for symbol, loss in symbol_losses.items()
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("loss_contribution").reset_index(drop=True)


def build_alpha_attribution_report(
    periods: list[PortfolioPeriod],
    panel: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    return_col: str = EXCESS_RETURN_COLUMN,
) -> dict[str, pd.DataFrame]:
    return_contrib = attribute_portfolio_periods(periods, panel, return_col=return_col)
    ic_contrib = attribute_ic_by_symbol(predictions)
    drawdown_contrib = attribute_drawdown_contribution(periods, panel, return_col=return_col)

    merged = return_contrib.merge(ic_contrib, on="symbol", how="outer").merge(
        drawdown_contrib,
        on="symbol",
        how="outer",
    )
    return {
        "by_symbol": merged,
        "return_contribution": return_contrib,
        "ic_contribution": ic_contrib,
        "drawdown_contribution": drawdown_contrib,
    }
