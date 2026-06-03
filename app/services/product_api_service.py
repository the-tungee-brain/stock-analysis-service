"""Map precomputed snapshots to product API v1 contracts."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app.services.pipeline_status_reader import PipelineStatusReader
from app.services.portfolio_enriched_service import get_latest_portfolio_enriched
from app.services.ranking_service import get_top_rankings
from app.api.product.models import (
    API_VERSION,
    HoldingScoreContributionV1,
    PortfolioLatestResponseV1,
    PortfolioMetricsV1,
    PortfolioTopContributorV1,
    RankingItemV1,
    RankingsTopResponseV1,
    RiskLayerV1,
    SystemHealthResponseV1,
)
from ranking_pipeline.config import default_config
from ranking_pipeline.portfolio.persistence import open_portfolio_store
from ranking_pipeline.storage.sqlite import open_store


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_top_contributors_v1(
    raw: list | None,
) -> list[PortfolioTopContributorV1]:
    """Map persisted metrics_json contributors to v1 contract."""
    if not raw:
        return []
    out: list[PortfolioTopContributorV1] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol")
        if not symbol:
            continue
        try:
            out.append(
                PortfolioTopContributorV1(
                    symbol=str(symbol),
                    weight=float(row.get("weight") or 0.0),
                    expected_excess_return=float(row.get("expected_excess_return") or 0.0),
                    contribution=float(row.get("contribution") or 0.0),
                )
            )
        except (TypeError, ValueError):
            continue
    return out


def get_rankings_top_v1(*, limit: int = 20, run_id: str | None = None) -> RankingsTopResponseV1:
    """Serve latest or specified ranking run (precomputed only)."""
    cfg = default_config()
    store = open_store(cfg)
    rid = run_id or store.latest_run_id()
    if not rid:
        raise LookupError("No ranking runs available")

    meta = store.get_run_meta(rid)
    if not meta:
        raise LookupError(f"Ranking run not found: {rid}")

    rows = store.get_ranking_results(rid, limit=limit)
    items = [
        RankingItemV1(
            symbol=r["symbol"],
            rank=r["rank"],
            final_score=float(r["final_score"]),
            ml_probability=r.get("ml_probability"),
            expected_excess_return=r.get("expected_excess_return"),
        )
        for r in rows
    ]

    return RankingsTopResponseV1(
        api_version=API_VERSION,
        timestamp=meta.get("created_at") or _utc_timestamp(),
        run_id=rid,
        as_of_date=meta["as_of_date"],
        regime_id=meta.get("regime_id"),
        items=items,
    )


def get_portfolio_latest_v1() -> PortfolioLatestResponseV1:
    """Serve latest portfolio snapshot with metrics and optional risk layer."""
    enriched = get_latest_portfolio_enriched()
    pstore = open_portfolio_store()
    raw = pstore.get_portfolio(enriched.portfolio_id) or {}
    snap = raw.get("snapshot") or {}
    metrics_row = raw.get("metrics") or {}
    metrics_json: dict = {}
    if metrics_row.get("metrics_json"):
        metrics_json = json.loads(metrics_row["metrics_json"])

    risk_payload = metrics_json.get("risk_layer") or {}
    turnover = metrics_row.get("turnover")
    if turnover is None:
        turnover = risk_payload.get("turnover")

    holdings: list[HoldingScoreContributionV1] = []
    for h in raw.get("holdings", []):
        w = float(h["weight"])
        er = h.get("expected_excess_return") or 0.0
        holdings.append(
            HoldingScoreContributionV1(
                symbol=h["symbol"],
                weight=w,
                score_contribution=float(w * er) if er else 0.0,
                final_score=h.get("final_score"),
                expected_excess_return=h.get("expected_excess_return"),
            )
        )

    risk_layer = None
    if enriched.risk_layer:
        rl = enriched.risk_layer
        risk_layer = RiskLayerV1(
            portfolio_beta=rl.portfolio_beta,
            portfolio_volatility=rl.portfolio_volatility,
            target_volatility=rl.target_volatility,
            correlation_risk_score=rl.correlation_risk_score,
            sector_breakdown=rl.sector_breakdown or {},
            vol_scale_factor=rl.vol_scale_factor,
        )
    elif risk_payload:
        risk_layer = RiskLayerV1(
            portfolio_beta=risk_payload.get("portfolio_beta"),
            portfolio_volatility=risk_payload.get("portfolio_volatility"),
            target_volatility=risk_payload.get("target_volatility"),
            correlation_risk_score=risk_payload.get("correlation_risk_score"),
            sector_breakdown=risk_payload.get("sector_breakdown") or {},
            vol_scale_factor=risk_payload.get("vol_scale_factor"),
        )

    beta = None
    vol = None
    corr_score = None
    sectors: dict[str, float] = {}
    if risk_layer:
        beta = risk_layer.portfolio_beta
        vol = risk_layer.portfolio_volatility
        corr_score = risk_layer.correlation_risk_score
        sectors = risk_layer.sector_breakdown

    return PortfolioLatestResponseV1(
        api_version=API_VERSION,
        timestamp=snap.get("created_at") or _utc_timestamp(),
        portfolio_id=enriched.portfolio_id,
        ranking_run_id=enriched.ranking_run_id,
        as_of_date=enriched.as_of_date,
        sizing_mode=enriched.sizing_mode,
        holdings=holdings,
        metrics=PortfolioMetricsV1(
            expected_return_5d=float(enriched.risk.expected_return_5d),
            expected_excess_5d=float(enriched.risk.expected_excess_5d),
            volatility=vol or metrics_row.get("portfolio_volatility"),
            beta_vs_spy=beta,
            correlation_risk_score=corr_score,
            sector_breakdown=sectors,
            turnover_estimate=float(turnover) if turnover is not None else None,
            concentration_hhi=metrics_row.get("concentration_hhi"),
        ),
        risk_layer=risk_layer,
        top_contributors=_parse_top_contributors_v1(metrics_json.get("top_contributors")),
    )


def get_system_health_v1() -> SystemHealthResponseV1:
    """Operational health from SQLite run metadata."""
    snap = PipelineStatusReader().get_status()
    return SystemHealthResponseV1(
        api_version=API_VERSION,
        last_pipeline_run_time=snap.last_pipeline_run_time,
        universe_size=snap.universe_size,
        last_successful_ranking_run=snap.last_successful_ranking_run,
        last_successful_portfolio_run=snap.last_successful_portfolio_run,
        system_status=snap.system_status,
        last_ranking_run_at=snap.last_ranking_run_at,
        last_portfolio_run_at=snap.last_portfolio_run_at,
        regime_id=snap.regime_id,
    )
