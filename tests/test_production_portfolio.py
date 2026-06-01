"""Tests for production portfolio configuration."""

from __future__ import annotations

import pandas as pd

from backtest.production_portfolio import ProductionPortfolioConfig, run_production_portfolio_backtest
from backtest.ranking_portfolio import RankingStrategy
from models.labels import EXCESS_RETURN_COLUMN
from models.pattern_production import production_portfolio_config


def _predictions() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=20)
    symbols = [f"S{i}" for i in range(10)]
    for date in dates:
        for rank, symbol in enumerate(symbols):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "prob_1": 0.95 - rank * 0.05,
                    EXCESS_RETURN_COLUMN: 0.03 - rank * 0.002,
                }
            )
    return pd.DataFrame(rows)


def test_production_defaults_match_phase2_winner():
    cfg = production_portfolio_config()
    assert cfg.universe == "top20"
    assert cfg.top_n == 10
    assert cfg.rebalance_days == 5
    assert cfg.hold_days == 5
    ranking = cfg.to_ranking_config()
    assert ranking.strategy == RankingStrategy.LONG_TOP_N
    assert ranking.max_position_weight == 0.15


def test_production_portfolio_backtest_builds_concentration_report():
    preds = _predictions()
    labeled = {
        symbol: preds[preds["symbol"] == symbol].copy()
        for symbol in preds["symbol"].unique()
    }
    cfg = ProductionPortfolioConfig(universe="top20", top_n=3, rebalance_days=5, hold_days=5)
    # Monkeypatch symbol resolution for unit test
    result = run_production_portfolio_backtest(preds, labeled, cfg)
    assert result["summary"]["n_periods"] >= 1
    assert "symbol_exposure" in result["concentration"]
    assert "sector_exposure" in result["concentration"]
