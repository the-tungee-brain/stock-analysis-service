"""Cross-sectional ranking within the model training universe."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.pattern_intelligence.benchmarks import is_model_benchmark_symbol
from data.symbols import get_training_universe
from analysis.research_decision.features import build_feature_history
from analysis.research_decision.signal_change import predict_from_feature_row
from models.prediction_service import KEY_INDICATORS, LoadedModel, predict_for_symbol

SCORE_CHANGE_MATERIAL = 0.05


def predict_universe_scores(
    loaded: LoadedModel,
    *,
    universe: str = "top20",
) -> list[dict[str, Any]]:
    symbols = get_training_universe(universe)
    rows: list[dict[str, Any]] = []

    for symbol in symbols:
        if is_model_benchmark_symbol(symbol):
            continue
        try:
            payload = predict_for_symbol(symbol, loaded)
        except (FileNotFoundError, ValueError):
            continue
        score = payload.get("ranking_score", payload.get("up_prob"))
        if score is None:
            continue
        rows.append(
            {
                "symbol": symbol.upper(),
                "ranking_score": float(score),
                "prediction": int(payload["prediction"]),
                "indicators": dict(payload.get("indicators") or {}),
                "date": str(payload["date"]),
            }
        )

    rows.sort(key=lambda row: row["ranking_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        n = len(rows)
        row["percentile"] = int(round(100 * (1 - (rank - 1) / max(n - 1, 1))))
    return rows


def predict_universe_scores_prior(
    loaded: LoadedModel,
    *,
    universe: str = "top20",
) -> list[dict[str, Any]]:
    """Cross-sectional scores as of the prior feature-ready session."""
    symbols = get_training_universe(universe)
    rows: list[dict[str, Any]] = []

    for symbol in symbols:
        if is_model_benchmark_symbol(symbol):
            continue
        try:
            history = build_feature_history(symbol, lookback=2)
            if len(history) < 2:
                continue
            prior_row = history.iloc[-2]
            pred = predict_from_feature_row(prior_row, loaded)
            score = pred.get("ranking_score")
            if score is None:
                continue
            indicators = {
                name: float(prior_row[name])
                for name in KEY_INDICATORS
                if name in prior_row.index and pd.notna(prior_row[name])
            }
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "ranking_score": float(score),
                    "prediction": int(pred["prediction"]),
                    "indicators": indicators,
                    "date": pd.Timestamp(prior_row.name).strftime("%Y-%m-%d"),
                }
            )
        except (FileNotFoundError, ValueError):
            continue

    rows.sort(key=lambda row: row["ranking_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        n = len(rows)
        row["percentile"] = int(round(100 * (1 - (rank - 1) / max(n - 1, 1))))
    return rows


def ranking_explanation(
    *,
    rank: int,
    universe_size: int,
    percentile: int,
) -> dict[str, Any]:
    expected = _expected_outcome(rank, universe_size, percentile)
    return {
        "rank": rank,
        "universe_size": universe_size,
        "percentile": percentile,
        "percentile_label": _percentile_label(percentile),
        "expected_outcome": expected,
        "rank_display": f"{rank} / {universe_size}",
    }


def _percentile_label(percentile: int) -> str:
    suffix = "th"
    if percentile % 100 not in (11, 12, 13):
        if percentile % 10 == 1:
            suffix = "st"
        elif percentile % 10 == 2:
            suffix = "nd"
        elif percentile % 10 == 3:
            suffix = "rd"
    return f"{percentile}{suffix} percentile"


def _expected_outcome(rank: int, universe_size: int, percentile: int) -> str:
    if universe_size <= 0:
        return "Insufficient universe data"
    top_cut = max(1, universe_size // 5)
    bottom_cut = universe_size - top_cut + 1

    if rank <= top_cut:
        return "Likely outperform SPY"
    if rank <= universe_size // 2:
        return "Slightly outperform SPY"
    if rank >= bottom_cut:
        return "Likely underperform SPY"
    if percentile >= 45:
        return "Near inline with SPY"
    return "Slightly underperform SPY"


def find_symbol_ranking(
    rows: list[dict[str, Any]],
    symbol: str,
) -> dict[str, Any] | None:
    symbol_upper = symbol.strip().upper()
    for row in rows:
        if row["symbol"] == symbol_upper:
            return row
    return None


def rank_movers(
    today_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]],
    *,
    limit: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    prior_by_symbol = {row["symbol"]: row for row in prior_rows}
    changes: list[dict[str, Any]] = []

    for row in today_rows:
        prior = prior_by_symbol.get(row["symbol"])
        if prior is None:
            continue
        score_delta = row["ranking_score"] - prior["ranking_score"]
        rank_delta = prior["rank"] - row["rank"]
        if abs(score_delta) < 0.005 and rank_delta == 0:
            continue
        changes.append(
            {
                **row,
                "prior_rank": prior["rank"],
                "prior_score": prior["ranking_score"],
                "rank_change": rank_delta,
                "score_change": score_delta,
            }
        )

    upgrades = sorted(changes, key=lambda item: item["rank_change"], reverse=True)[:limit]
    downgrades = sorted(changes, key=lambda item: item["rank_change"])[:limit]
    return {"upgrades": upgrades, "downgrades": downgrades}
