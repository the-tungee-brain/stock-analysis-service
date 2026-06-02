"""Tests for productization layer."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.prediction_ledger.store import load_ledger, upsert_rows
from analysis.productization.research_brief import build_research_brief
from analysis.productization.verdict import verdict_from_score
from data.paths import LEDGER_DIR, ledger_parquet_path


def test_verdict_bands():
    assert verdict_from_score(75) == "Strong Buy"
    assert verdict_from_score(65) == "Buy"
    assert verdict_from_score(50) == "Neutral"
    assert verdict_from_score(40) == "Reduce"
    assert verdict_from_score(20) == "Avoid"


def test_research_brief_limits_reasons():
    brief = build_research_brief(
        research_decision={
            "research_quality_score": {"score": 72},
            "multi_timeframe": {
                "forecast_trend": "bullish",
                "forecast_trend_label": "Bullish",
                "conclusion": "Aligned uptrend.",
            },
            "contributors": {
                "positive": ["Strong RS", "Above SMA200", "Momentum", "Extra"],
                "negative": ["Near resistance", "Weak RS 126d", "Volume absent", "Extra bad"],
            },
            "ranking": {"expected_outcome": "Slightly outperform SPY"},
            "regime": {"current": {"regime_label": "Bull + High VIX", "vix_regime": "high"}},
        },
        ranking_score=0.72,
    )
    assert len(brief["reasons"]) <= 3
    assert len(brief["risk_factors"]) <= 3
    assert brief["verdict"]["label"] == "Strong Buy"
    assert brief["verdict"]["trend_verdict"] == "Bull"


def test_ledger_upsert(tmp_path, monkeypatch):
    monkeypatch.setattr("analysis.prediction_ledger.store.LEDGER_DIR", tmp_path)
    monkeypatch.setattr(
        "analysis.prediction_ledger.store.ledger_parquet_path",
        lambda: tmp_path / "predictions.parquet",
    )
    upsert_rows(
        [
            {
                "symbol": "MSFT",
                "as_of_date": "2024-06-01",
                "model_key": "test",
                "model_version": "2024-05-31",
                "ranking_score": 0.7,
                "rank": 3,
                "percentile": 80,
                "regime_label": "Bull + Medium VIX",
                "market_regime": "bull",
                "vix_regime": "medium",
                "expected_outcome": "Outperform",
                "resolved": False,
                "return_5d": None,
                "return_spy_5d": None,
                "excess_return_5d": None,
                "correct": None,
                "alpha_captured": None,
            }
        ]
    )
    frame = load_ledger(tmp_path / "predictions.parquet")
    assert len(frame) == 1
    assert frame.iloc[0]["symbol"] == "MSFT"
