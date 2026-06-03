"""Build tradable portfolio from precomputed ranking results."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd

from ranking_pipeline.config import default_config
from ranking_pipeline.datetime_utils import to_naive_utc_timestamp
from ranking_pipeline.features.parquet_store import load_ranking_features
from ranking_pipeline.portfolio.config import PortfolioConfig, default_portfolio_config
from ranking_pipeline.portfolio.constraints import apply_all_constraints
from ranking_pipeline.portfolio.metrics import PortfolioRiskMetrics, compute_portfolio_metrics
from ranking_pipeline.portfolio.persistence import PortfolioStore, open_portfolio_store
from ranking_pipeline.portfolio.rebalancer import compute_trades, daily_rebalance
from ranking_pipeline.portfolio.sizing import RankedCandidate, compute_target_weights
from ranking_pipeline.storage.sqlite import RankingStore, open_store


def _load_atr(symbol: str, as_of: pd.Timestamp) -> float | None:
    try:
        df = load_ranking_features(symbol)
        cutoff = to_naive_utc_timestamp(as_of)
        df = df[df.index <= cutoff]
        if df.empty or "atr_14" not in df.columns:
            return None
        val = df["atr_14"].iloc[-1]
        return float(val) if pd.notna(val) else None
    except FileNotFoundError:
        return None


def _ranking_rows_to_candidates(
    rows: list[dict],
    as_of: pd.Timestamp,
) -> list[RankedCandidate]:
    out: list[RankedCandidate] = []
    for r in rows:
        sym = r["symbol"]
        out.append(
            RankedCandidate(
                symbol=sym,
                final_score=float(r["final_score"]),
                expected_excess_return=r.get("expected_excess_return"),
                ml_probability=r.get("ml_probability"),
                atr_14=_load_atr(sym, as_of),
            )
        )
    return out


def construct_portfolio_from_run(
    ranking_run_id: str | None = None,
    *,
    portfolio_config: PortfolioConfig | None = None,
    ranking_store: RankingStore | None = None,
    portfolio_store: PortfolioStore | None = None,
    sector_by_symbol: dict[str, str] | None = None,
) -> dict:
    """
    Downstream portfolio build from SQLite ranking results only.

    Does not invoke ranking, feature, or ML pipelines.
    """
    pcfg = portfolio_config or default_portfolio_config()
    rstore = ranking_store or open_store(default_config())
    pstore = portfolio_store or open_portfolio_store(pcfg)

    run_id = ranking_run_id or rstore.latest_run_id()
    if not run_id:
        raise LookupError("No ranking run available")

    meta = rstore.get_run_meta(run_id)
    if not meta:
        raise LookupError(f"Ranking run not found: {run_id}")

    as_of = pd.Timestamp(meta["as_of_date"])
    rows = rstore.get_ranking_results(run_id, limit=pcfg.top_n)
    if not rows:
        raise ValueError("Ranking run has no results")

    candidates = _ranking_rows_to_candidates(rows, as_of)
    target = compute_target_weights(candidates, pcfg.sizing_mode)

    snapshot_id = meta.get("universe_snapshot_id")
    adv_map = (
        rstore.load_adv_by_symbols(snapshot_id, list(target.index))
        if snapshot_id
        else {}
    )

    prev_series = pd.Series(
        pstore.load_previous_weights(meta["as_of_date"]),
        dtype="float64",
    )

    constrained = apply_all_constraints(
        target,
        prev_series,
        config=pcfg.constraints,
        adv_by_symbol=adv_map,
        sector_by_symbol=sector_by_symbol,
    )

    final = daily_rebalance(
        constrained,
        prev_series,
        smoothing_alpha=pcfg.smoothing_alpha,
    )

    risk = compute_portfolio_metrics(final, candidates, prev_series)
    trades = compute_trades(prev_series, final)

    portfolio_id = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
        + "-pf-"
        + uuid.uuid4().hex[:8]
    )

    holdings = _holdings_payload(final, candidates)
    pstore.save_portfolio(
        portfolio_id=portfolio_id,
        ranking_run_id=run_id,
        as_of_date=meta["as_of_date"],
        sizing_mode=pcfg.sizing_mode.value,
        holdings=holdings,
        metrics=_metrics_payload(risk),
        trades=_trades_payload(trades),
    )

    return {
        "portfolio_id": portfolio_id,
        "ranking_run_id": run_id,
        "as_of_date": meta["as_of_date"],
        "weights": final.to_dict(),
        "metrics": _metrics_payload(risk),
        "trade_count": len([t for t in trades if t.side.value != "hold"]),
    }


def _holdings_payload(
    weights: pd.Series,
    candidates: list[RankedCandidate],
) -> list[dict]:
    by_sym = {c.symbol: c for c in candidates}
    out: list[dict] = []
    for sym, w in weights.items():
        c = by_sym.get(sym)
        out.append(
            {
                "symbol": sym,
                "weight": float(w),
                "final_score": c.final_score if c else None,
                "ml_probability": c.ml_probability if c else None,
                "expected_excess_return": c.expected_excess_return if c else None,
                "atr_14": c.atr_14 if c else None,
            }
        )
    return out


def _metrics_payload(risk: PortfolioRiskMetrics) -> dict:
    return {
        "expected_return_5d": risk.expected_return_5d,
        "expected_excess_5d": risk.expected_excess_5d,
        "portfolio_volatility": risk.portfolio_volatility_proxy,
        "turnover": risk.turnover,
        "concentration_hhi": risk.concentration_hhi,
        "top_contributors": risk.top_contributors,
    }


def _trades_payload(trades: list) -> list[dict]:
    return [
        {
            "symbol": t.symbol,
            "side": t.side.value,
            "weight_change": t.weight_change,
            "target_weight": t.target_weight,
            "previous_weight": t.previous_weight,
        }
        for t in trades
    ]
