"""Portfolio ranking dashboard assembly."""

from __future__ import annotations

from typing import Any

from analysis.research_decision.ranking import (
    find_symbol_ranking,
    predict_universe_scores,
    predict_universe_scores_prior,
    rank_movers,
)
from analysis.research_decision.trend_labels import (
    classify_daily_trend,
    classify_forecast_trend,
    trend_label_display,
)
from models.prediction_service import LoadedModel


def _thesis_summary(row: dict[str, Any]) -> str:
    indicators = row.get("indicators") or {}
    daily = classify_daily_trend(indicators)
    forecast = classify_forecast_trend(
        prediction=row.get("prediction"),
        ranking_score=row.get("ranking_score"),
    )
    rs21 = indicators.get("rs_vs_spy_21d")
    rs_note = ""
    if rs21 is not None:
        rs_note = f" RS vs SPY {rs21:+.1%}."
    return (
        f"{trend_label_display(daily)} daily · "
        f"{trend_label_display(forecast)} 5d.{rs_note}"
    ).strip()


def _enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    indicators = row.get("indicators") or {}
    daily = classify_daily_trend(indicators)
    forecast = classify_forecast_trend(
        prediction=row.get("prediction"),
        ranking_score=row.get("ranking_score"),
    )
    rs21 = indicators.get("rs_vs_spy_21d")
    return {
        "symbol": row["symbol"],
        "rank": row["rank"],
        "percentile": row.get("percentile"),
        "ranking_score": row["ranking_score"],
        "trend": trend_label_display(forecast),
        "daily_trend": trend_label_display(daily),
        "relative_strength": float(rs21) if rs21 is not None else None,
        "thesis_summary": _thesis_summary(row),
        "rank_change": row.get("rank_change"),
        "score_change": row.get("score_change"),
        "prior_rank": row.get("prior_rank"),
    }


def build_portfolio_ranking_dashboard(
    loaded: LoadedModel,
    *,
    universe: str = "top20",
) -> dict[str, Any]:
    today_rows = predict_universe_scores(loaded, universe=universe)
    prior_rows = predict_universe_scores_prior(loaded, universe=universe)
    if not prior_rows:
        prior_rows = today_rows

    movers = rank_movers(today_rows, prior_rows)
    top10 = [_enrich_row(row) for row in today_rows[:10]]
    bottom10 = [_enrich_row(row) for row in today_rows[-10:][::-1]]
    upgrades = [_enrich_row(row) for row in movers["upgrades"]]
    downgrades = [_enrich_row(row) for row in movers["downgrades"]]

    as_of = today_rows[0]["date"] if today_rows else None
    return {
        "as_of_date": as_of,
        "universe_size": len(today_rows),
        "top10": top10,
        "bottom10": bottom10,
        "biggest_upgrades": upgrades,
        "biggest_downgrades": downgrades,
        "all_rankings": [_enrich_row(row) for row in today_rows],
    }


def lookup_symbol_in_dashboard(
    dashboard: dict[str, Any],
    symbol: str,
) -> dict[str, Any] | None:
    symbol_upper = symbol.strip().upper()
    for row in dashboard.get("all_rankings") or []:
        if row["symbol"] == symbol_upper:
            return row
    return None
