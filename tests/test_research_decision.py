"""Tests for the research decision layer."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.research_decision.contributors import build_contributors, contributor_deltas
from analysis.research_decision.quality_score import compute_research_quality_score
from analysis.research_decision.ranking import ranking_explanation
from analysis.research_decision.trend_labels import (
    build_multi_timeframe_payload,
    classify_daily_trend,
    classify_forecast_trend,
    synthesize_multi_timeframe_conclusion,
)
from app.services.research_decision_service import build_research_decision_payload
from data.store import save_raw
from features.build_features import build_and_save_features
from models.pattern_production import (
    PRODUCTION_LABEL_SCHEME,
    PRODUCTION_TRAINING_UNIVERSE,
    production_model_config,
    production_train_metadata_kwargs,
)
from models.prediction_service import load_deployed_model
from models.train_and_save import TrainAndSaveConfig, train_and_save
from tests.test_pattern_train_and_save import _synthetic_ohlcv


@pytest.fixture
def loaded_tradeable_model(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr("data.paths.RAW_DIR", raw_dir)
    monkeypatch.setattr("data.paths.FEATURES_DIR", features_dir)

    for symbol in ("MSFT", "AAPL", "SPY"):
        save_raw(_synthetic_ohlcv(rows=600), symbol)
        build_and_save_features(symbol)

    train_and_save(
        TrainAndSaveConfig(
            symbols=("MSFT", "AAPL"),
            train_end_date=pd.Timestamp("2021-12-31"),
            artifact_dir=artifact_dir,
            model_config=production_model_config(),
            min_up_prob=0.55,
            universe=PRODUCTION_TRAINING_UNIVERSE,
            extra_metadata={
                key: value
                for key, value in production_train_metadata_kwargs(
                    universe=PRODUCTION_TRAINING_UNIVERSE
                ).items()
                if key not in {"min_up_prob", "universe"}
            },
        )
    )
    return load_deployed_model(artifact_dir)


def test_multi_timeframe_conclusion_pullback():
    conclusion = synthesize_multi_timeframe_conclusion("bullish", "bullish", "bearish")
    assert "pullback" in conclusion.lower()


def test_ranking_explanation_percentile():
    block = ranking_explanation(rank=7, universe_size=20, percentile=65)
    assert block["rank_display"] == "7 / 20"
    assert "65" in block["percentile_label"]
    assert block["expected_outcome"]


def test_contributors_positive_rs():
    result = build_contributors({"rs_vs_spy_21d": 0.05, "close_vs_sma200": 0.02})
    assert any("RS 21d" in item for item in result["positive"])


def test_signal_change_drivers():
    today = {"rs_vs_spy_21d": 0.01, "close_vs_sma20": -0.01, "ret_21d": -0.05}
    prior = {"rs_vs_spy_21d": 0.04, "close_vs_sma20": 0.02, "ret_21d": 0.03}
    drivers = contributor_deltas(today, prior)
    assert "Relative strength weakened" in drivers["negative"]
    assert "Price lost SMA20" in drivers["negative"]


def test_research_quality_score_range():
    score = compute_research_quality_score(
        ranking_score=0.72,
        daily_trend=classify_daily_trend({"close_vs_sma20": 0.02, "close_vs_sma200": 0.05, "ret_21d": 0.04}),
        weekly_trend="bullish",
        indicators={"rs_vs_spy_21d": 0.04},
        regime_market="bull",
        signal_bias="bullish",
        chart_intelligence_score=70,
    )
    assert 0 <= score["score"] <= 100


def test_build_research_decision_payload(loaded_tradeable_model, monkeypatch):
    monkeypatch.setattr(
        "analysis.research_decision.service.ensure_raw_ohlcv",
        lambda symbol: _synthetic_ohlcv(rows=600),
    )
    monkeypatch.setattr(
        "analysis.research_decision.features.ensure_raw_ohlcv",
        lambda symbol: _synthetic_ohlcv(rows=600),
    )
    payload = build_research_decision_payload("MSFT", loaded_tradeable_model)
    assert payload is not None
    assert payload.symbol == "MSFT"
    assert payload.multi_timeframe is not None
    assert payload.research_quality_score is not None


def test_forecast_trend_from_score():
    assert classify_forecast_trend(prediction=1, ranking_score=0.7, min_up_prob=0.55) == "bullish"
    assert classify_forecast_trend(prediction=0, ranking_score=0.4, min_up_prob=0.55) == "bearish"


def test_multi_timeframe_payload_labels():
    payload = build_multi_timeframe_payload(
        weekly="bullish",
        daily="bullish",
        forecast="bearish",
    )
    assert payload["weekly_trend_label"] == "Bullish"
    assert payload["conclusion"]
