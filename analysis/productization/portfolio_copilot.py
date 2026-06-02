"""Portfolio-level copilot intelligence."""

from __future__ import annotations

from typing import Any

from analysis.productization.verdict import score_from_ranking, verdict_from_score
from analysis.research_decision.ranking import find_symbol_ranking, predict_universe_scores
from data.sector_mapping import sector_map_for_symbols
from models.prediction_service import LoadedModel


def build_portfolio_copilot(
    symbols: list[str],
    loaded: LoadedModel,
) -> dict[str, Any]:
    cleaned = _clean_symbols(symbols)
    if not cleaned:
        return _empty_copilot()

    universe_rows = predict_universe_scores(loaded)
    universe_by_symbol = {row["symbol"]: row for row in universe_rows}

    holdings: list[dict[str, Any]] = []
    for symbol in cleaned:
        row = universe_by_symbol.get(symbol)
        if row is None:
            try:
                from models.prediction_service import predict_for_symbol

                payload = predict_for_symbol(symbol, loaded)
                score = payload.get("ranking_score", payload.get("up_prob"))
                row = {
                    "symbol": symbol,
                    "ranking_score": float(score) if score is not None else 0.5,
                    "indicators": dict(payload.get("indicators") or {}),
                }
            except (FileNotFoundError, ValueError):
                row = {"symbol": symbol, "ranking_score": 0.5, "indicators": {}}
        rank_row = find_symbol_ranking(universe_rows, symbol)
        score = row.get("ranking_score", 0.5)
        quality = score_from_ranking(score)
        indicators = row.get("indicators") or {}
        holdings.append(
            {
                "symbol": symbol,
                "ranking_score": float(score),
                "quality_score": quality,
                "verdict": verdict_from_score(quality),
                "rank": rank_row["rank"] if rank_row else None,
                "percentile": rank_row["percentile"] if rank_row else None,
                "relative_strength": indicators.get("rs_vs_spy_21d"),
                "trend_strength": indicators.get("ret_21d"),
            }
        )

    n = len(holdings)
    equal_weight = 100.0 / n
    sectors = sector_map_for_symbols(cleaned)

    sector_weights: dict[str, float] = {}
    rs_values: list[float] = []
    trend_values: list[float] = []
    for item in holdings:
        sector = sectors.get(item["symbol"], "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + equal_weight
        if item["relative_strength"] is not None:
            rs_values.append(float(item["relative_strength"]))
        if item["trend_strength"] is not None:
            trend_values.append(float(item["trend_strength"]))

    portfolio_quality = int(round(sum(item["quality_score"] for item in holdings) / n))
    sorted_holdings = sorted(holdings, key=lambda item: item["quality_score"], reverse=True)
    universe_sorted = sorted(universe_rows, key=lambda row: row["ranking_score"], reverse=True)
    held = {item["symbol"] for item in holdings}

    rotation_candidates = [
        row["symbol"]
        for row in universe_sorted[:5]
        if row["symbol"] not in held
    ][:3]
    trim_candidates = [
        item["symbol"]
        for item in sorted(holdings, key=lambda item: item["quality_score"])[:3]
    ]

    overweight = [
        item["symbol"]
        for item in holdings
        if item["percentile"] is not None and item["percentile"] < 30
    ][:3]
    underweight_vs_model = rotation_candidates

    return {
        "portfolio_quality_score": portfolio_quality,
        "holdings_count": n,
        "exposure": {
            "sectors": [
                {"sector": sector, "weight_pct": round(weight, 1)}
                for sector, weight in sorted(
                    sector_weights.items(), key=lambda pair: pair[1], reverse=True
                )
            ],
            "avg_relative_strength": _avg(rs_values),
            "avg_trend_strength": _avg(trend_values),
        },
        "best_holdings": sorted_holdings[:3],
        "worst_holdings": sorted_holdings[-3:][::-1],
        "overweight_flags": overweight,
        "underweight_flags": underweight_vs_model,
        "suggested_rotation": {
            "add_candidates": rotation_candidates,
            "trim_candidates": trim_candidates,
            "note": "Ranking insight only — not trade execution.",
        },
    }


def _clean_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for raw in symbols:
        symbol = raw.strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        cleaned.append(symbol)
    return cleaned


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _empty_copilot() -> dict[str, Any]:
    return {
        "portfolio_quality_score": 0,
        "holdings_count": 0,
        "exposure": {"sectors": [], "avg_relative_strength": None, "avg_trend_strength": None},
        "best_holdings": [],
        "worst_holdings": [],
        "overweight_flags": [],
        "underweight_flags": [],
        "suggested_rotation": {
            "add_candidates": [],
            "trim_candidates": [],
            "note": "Provide at least one ticker.",
        },
    }
