"""Concentration controls and portfolio risk attribution."""

from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING

import numpy as np
import pandas as pd

from backtest.portfolio_weights import capped_equal_weights, weighted_portfolio_return, weights_for_symbols
from data.sector_mapping import UNKNOWN_SECTOR, get_sector
from models.labels import EXCESS_RETURN_COLUMN

if TYPE_CHECKING:
    from backtest.ranking_portfolio import PortfolioPeriod


def compute_sector_exposure(
    weights: Mapping[str, float],
    *,
    sector_map: Mapping[str, str] | None = None,
) -> pd.Series:
    """Aggregate portfolio weights by sector."""
    exposure: dict[str, float] = {}
    for symbol, weight in weights.items():
        sector = get_sector(str(symbol), sector_map)
        exposure[sector] = exposure.get(sector, 0.0) + float(weight)
    if not exposure:
        return pd.Series(dtype="float64")
    return pd.Series(exposure).sort_values(ascending=False)


def compute_position_overlap(periods: list["PortfolioPeriod"]) -> pd.DataFrame:
    """Track symbol overlap between consecutive rebalance periods."""
    if len(periods) < 2:
        return pd.DataFrame(
            columns=[
                "entry_date",
                "prev_entry_date",
                "overlap_count",
                "overlap_ratio",
                "prev_n_long",
                "n_long",
                "entered",
                "exited",
            ]
        )

    rows: list[dict[str, Any]] = []
    for prev, curr in zip(periods[:-1], periods[1:]):
        prev_set = set(prev.long_symbols)
        curr_set = set(curr.long_symbols)
        union = prev_set | curr_set
        overlap = prev_set & curr_set
        rows.append(
            {
                "entry_date": curr.entry_date,
                "prev_entry_date": prev.entry_date,
                "overlap_count": len(overlap),
                "overlap_ratio": len(overlap) / len(union) if union else float("nan"),
                "prev_n_long": len(prev_set),
                "n_long": len(curr_set),
                "entered": ",".join(sorted(curr_set - prev_set)),
                "exited": ",".join(sorted(prev_set - curr_set)),
            }
        )
    return pd.DataFrame(rows)


def compute_symbol_exposure_summary(periods: list["PortfolioPeriod"]) -> pd.DataFrame:
    """Average portfolio weight and selection frequency by symbol."""
    if not periods:
        return pd.DataFrame(
            columns=["symbol", "periods_held", "selection_rate", "avg_weight", "max_weight"]
        )

    stats: dict[str, dict[str, float | int]] = {}
    for period in periods:
        weights = capped_equal_weights(list(period.long_symbols), gross=1.0, max_weight=None)
        for symbol, weight in weights.items():
            bucket = stats.setdefault(
                symbol,
                {"periods_held": 0, "weight_sum": 0.0, "max_weight": 0.0},
            )
            bucket["periods_held"] = int(bucket["periods_held"]) + 1
            bucket["weight_sum"] = float(bucket["weight_sum"]) + float(weight)
            bucket["max_weight"] = max(float(bucket["max_weight"]), float(weight))

    rows: list[dict[str, Any]] = []
    n_periods = len(periods)
    for symbol, bucket in stats.items():
        periods_held = int(bucket["periods_held"])
        rows.append(
            {
                "symbol": symbol,
                "periods_held": periods_held,
                "selection_rate": periods_held / n_periods,
                "avg_weight": float(bucket["weight_sum"]) / periods_held,
                "max_weight": float(bucket["max_weight"]),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_weight", ascending=False).reset_index(drop=True)


def compute_contribution_to_risk(
    periods: list["PortfolioPeriod"],
    panel: pd.DataFrame,
    *,
    return_col: str = EXCESS_RETURN_COLUMN,
    max_weight: float | None = None,
) -> pd.DataFrame:
    """Estimate each symbol's contribution to portfolio return variance."""
    if not periods:
        return pd.DataFrame(columns=["symbol", "risk_contribution", "risk_share", "avg_weight"])

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    symbol_returns: dict[str, list[float]] = {}
    symbol_weights: dict[str, list[float]] = {}

    for period in periods:
        cross_section = frame[frame["date"] == period.entry_date]
        if cross_section.empty:
            continue
        weights = capped_equal_weights(
            list(period.long_symbols),
            gross=1.0,
            max_weight=max_weight,
        )
        for symbol, weight in weights.items():
            row = cross_section[cross_section["symbol"] == symbol]
            if row.empty:
                continue
            ret = float(row.iloc[0][return_col])
            symbol_returns.setdefault(symbol, []).append(weight * ret)
            symbol_weights.setdefault(symbol, []).append(weight)

    rows: list[dict[str, Any]] = []
    for symbol, weighted_rets in symbol_returns.items():
        series = pd.Series(weighted_rets, dtype="float64")
        risk = float(series.var(ddof=0)) if len(series) > 1 else float(series.iloc[0] ** 2)
        rows.append(
            {
                "symbol": symbol,
                "risk_contribution": risk,
                "avg_weight": float(np.mean(symbol_weights[symbol])),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    total_risk = float(out["risk_contribution"].sum())
    out["risk_share"] = out["risk_contribution"] / total_risk if total_risk > 0 else float("nan")
    return out.sort_values("risk_contribution", ascending=False).reset_index(drop=True)


def build_concentration_report(
    periods: list["PortfolioPeriod"],
    panel: pd.DataFrame,
    *,
    sector_map: Mapping[str, str] | None = None,
    max_position_weight: float | None = None,
) -> dict[str, Any]:
    """Bundle concentration, overlap, sector, and risk attribution views."""
    symbol_exposure = compute_symbol_exposure_summary(periods)
    overlap = compute_position_overlap(periods)
    risk = compute_contribution_to_risk(
        periods,
        panel,
        max_weight=max_position_weight,
    )

    sector_rows: list[dict[str, Any]] = []
    if not periods:
        sector_frame = pd.DataFrame(columns=["sector", "avg_weight", "max_weight", "periods"])
    else:
        sector_stats: dict[str, dict[str, float | int]] = {}
        for period in periods:
            weights = capped_equal_weights(
                list(period.long_symbols),
                gross=1.0,
                max_weight=max_position_weight,
            )
            for sector, weight in compute_sector_exposure(weights, sector_map=sector_map).items():
                bucket = sector_stats.setdefault(
                    sector,
                    {"weight_sum": 0.0, "max_weight": 0.0, "periods": 0},
                )
                bucket["weight_sum"] = float(bucket["weight_sum"]) + float(weight)
                bucket["max_weight"] = max(float(bucket["max_weight"]), float(weight))
                bucket["periods"] = int(bucket["periods"]) + 1
        sector_rows = [
            {
                "sector": sector,
                "avg_weight": float(bucket["weight_sum"]) / int(bucket["periods"]),
                "max_weight": float(bucket["max_weight"]),
                "periods": int(bucket["periods"]),
            }
            for sector, bucket in sector_stats.items()
        ]
        sector_frame = pd.DataFrame(sector_rows).sort_values("avg_weight", ascending=False)

    latest_weights: dict[str, float] = {}
    latest_sector = pd.Series(dtype="float64")
    if periods:
        latest_weights = capped_equal_weights(
            list(periods[-1].long_symbols),
            gross=1.0,
            max_weight=max_position_weight,
        )
        latest_sector = compute_sector_exposure(latest_weights, sector_map=sector_map)

    return {
        "symbol_exposure": symbol_exposure,
        "sector_exposure": sector_frame,
        "latest_sector_exposure": latest_sector,
        "position_overlap": overlap,
        "contribution_to_risk": risk,
        "latest_weights": latest_weights,
        "avg_overlap_ratio": float(overlap["overlap_ratio"].mean()) if not overlap.empty else float("nan"),
        "max_symbol_avg_weight": float(symbol_exposure["avg_weight"].max())
        if not symbol_exposure.empty
        else float("nan"),
    }
