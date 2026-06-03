"""Read latest constructed portfolio from SQLite."""

from __future__ import annotations

import json

from ranking_pipeline.portfolio.api_models import (
    LatestPortfolioResponse,
    PortfolioContributor,
    PortfolioHolding,
    PortfolioRiskSummary,
)
from ranking_pipeline.portfolio.persistence import open_portfolio_store


def get_latest_portfolio() -> LatestPortfolioResponse:
    store = open_portfolio_store()
    data = store.get_latest_portfolio()
    if not data:
        raise LookupError(
            "No portfolio snapshot found. Run scripts/run_portfolio_daily.py after ranking."
        )

    snap = data["snapshot"]
    metrics_raw = data.get("metrics") or {}
    metrics_json = {}
    if metrics_raw.get("metrics_json"):
        metrics_json = json.loads(metrics_raw["metrics_json"])

    contributors = metrics_json.get("top_contributors") or []

    holdings = [
        PortfolioHolding(
            symbol=h["symbol"],
            weight=float(h["weight"]),
            final_score=h.get("final_score"),
            probability_outperform_spy=h.get("ml_probability"),
            expected_excess_return=h.get("expected_excess_return"),
        )
        for h in data.get("holdings", [])
    ]

    return LatestPortfolioResponse(
        portfolio_id=snap["portfolio_id"],
        ranking_run_id=snap["ranking_run_id"],
        as_of_date=snap["as_of_date"],
        sizing_mode=snap["sizing_mode"],
        holdings=holdings,
        risk=PortfolioRiskSummary(
            expected_return_5d=float(metrics_raw.get("expected_return_5d") or 0),
            expected_excess_5d=float(metrics_raw.get("expected_excess_5d") or 0),
            portfolio_volatility=metrics_raw.get("portfolio_volatility"),
            turnover=metrics_raw.get("turnover"),
            concentration_hhi=metrics_raw.get("concentration_hhi"),
        ),
        top_contributors=[
            PortfolioContributor(
                symbol=c["symbol"],
                weight=float(c["weight"]),
                expected_excess_return=float(c["expected_excess_return"]),
                contribution=float(c["contribution"]),
            )
            for c in contributors
        ],
    )
