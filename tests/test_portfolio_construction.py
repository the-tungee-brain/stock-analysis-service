"""Portfolio construction layer tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ranking_pipeline.portfolio.config import PortfolioConfig, PortfolioConstraints, SizingMode
from ranking_pipeline.portfolio.constraints import apply_all_constraints, cap_position_weights
from ranking_pipeline.portfolio.metrics import compute_portfolio_metrics
from ranking_pipeline.portfolio.persistence import PortfolioStore
from ranking_pipeline.portfolio.rebalancer import compute_trades, smooth_weights
from ranking_pipeline.portfolio.sizing import RankedCandidate, compute_target_weights


def _candidates(n: int = 5) -> list[RankedCandidate]:
    return [
        RankedCandidate(
            symbol=f"S{i}",
            final_score=float(n - i),
            expected_excess_return=0.01 * (i + 1),
            atr_14=1.0 + i * 0.5,
        )
        for i in range(n)
    ]


def test_equal_weight_sums_to_one():
    w = compute_target_weights(_candidates(4), SizingMode.EQUAL_WEIGHT)
    assert abs(w.sum() - 1.0) < 1e-9


def test_score_weighted_favors_high_score():
    cands = _candidates(3)
    w = compute_target_weights(cands, SizingMode.SCORE_WEIGHTED)
    assert w["S0"] > w["S2"]


def test_vol_adjusted_differs_from_equal():
    w_eq = compute_target_weights(_candidates(4), SizingMode.EQUAL_WEIGHT)
    w_vol = compute_target_weights(_candidates(4), SizingMode.VOLATILITY_ADJUSTED)
    assert not w_eq.equals(w_vol)


def test_max_position_cap():
    w = pd.Series({f"S{i}": 0.5 if i == 0 else 0.5 / 9 for i in range(10)})
    capped = cap_position_weights(w, max_weight=0.10)
    assert capped.max() <= 0.10 + 1e-6
    assert abs(capped.sum() - 1.0) < 1e-6


def test_smoothing_blends_prior():
    prev = pd.Series({"A": 0.5, "B": 0.5})
    tgt = pd.Series({"A": 1.0})
    blended = smooth_weights(tgt, prev, alpha=0.3)
    assert blended["A"] < 1.0
    assert blended["B"] > 0


def test_trades_detect_buy_sell():
    prev = pd.Series({"A": 0.5, "B": 0.5})
    cur = pd.Series({"A": 0.7, "B": 0.3})
    trades = compute_trades(prev, cur)
    sides = {t.symbol: t.side.value for t in trades}
    assert sides["A"] == "buy"
    assert sides["B"] == "sell"


def test_portfolio_metrics_contributors():
    cands = _candidates(3)
    w = compute_target_weights(cands, SizingMode.EQUAL_WEIGHT)
    m = compute_portfolio_metrics(w, cands, pd.Series(dtype=float))
    assert m.expected_excess_5d > 0
    assert len(m.top_contributors) > 0


def test_sqlite_portfolio_roundtrip(tmp_path: Path):
    db = tmp_path / "pf.db"
    store = PortfolioStore(db)
    store.save_portfolio(
        portfolio_id="pf-1",
        ranking_run_id="run-1",
        as_of_date="2026-06-01",
        sizing_mode="equal_weight",
        holdings=[{"symbol": "AAPL", "weight": 1.0, "final_score": 1.2}],
        metrics={
            "expected_return_5d": 0.01,
            "expected_excess_5d": 0.005,
            "portfolio_volatility": 2.0,
            "turnover": 1.0,
            "concentration_hhi": 1.0,
            "top_contributors": [],
        },
        trades=[],
    )
    latest = store.get_latest_portfolio()
    assert latest is not None
    assert latest["holdings"][0]["symbol"] == "AAPL"


def test_turnover_constraint_limits_change():
    prev = pd.Series({"A": 1.0})
    tgt = pd.Series({"B": 1.0})
    cfg = PortfolioConstraints(max_daily_turnover=0.2, max_position_weight=1.0)
    out = apply_all_constraints(tgt, prev, config=cfg)
    turnover = float((out.reindex(["A", "B"]).fillna(0) - prev.reindex(["A", "B"]).fillna(0)).abs().sum())
    assert turnover <= 0.2 + 1e-6
