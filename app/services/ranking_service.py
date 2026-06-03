"""Read precomputed ranking results from SQLite."""

from __future__ import annotations

from ranking_pipeline.api_models import (
    FeatureContribution,
    RankedStock,
    RankingRunMeta,
    TopRankingsResponse,
)
from ranking_pipeline.config import default_config
from ranking_pipeline.storage.sqlite import open_store


def get_top_rankings(
    *,
    limit: int = 20,
    run_id: str | None = None,
) -> TopRankingsResponse:
    cfg = default_config()
    store = open_store(cfg)
    rid = run_id or store.latest_run_id()
    if not rid:
        raise LookupError("No ranking runs found. Run scripts/run_ranking_daily.py first.")

    meta = store.get_run_meta(rid)
    if not meta:
        raise LookupError(f"Ranking run not found: {rid}")

    rows = store.get_ranking_results(rid, limit=limit)
    stocks = [
        RankedStock(
            symbol=r["symbol"],
            rank=r["rank"],
            final_score=r["final_score"],
            composite_score=r.get("composite_score"),
            probability_outperform_spy=r.get("ml_probability"),
            expected_excess_return=r.get("expected_excess_return"),
            contributions=[
                FeatureContribution(group=g, weighted_contribution=float(v))
                for g, v in (r.get("contributions") or {}).items()
            ],
        )
        for r in rows
    ]
    return TopRankingsResponse(
        run=RankingRunMeta(
            run_id=meta["run_id"],
            as_of_date=meta["as_of_date"],
            model_backend=meta["model_backend"],
            universe_snapshot_id=meta.get("universe_snapshot_id"),
            symbol_count=meta.get("symbol_count"),
        ),
        stocks=stocks,
    )
