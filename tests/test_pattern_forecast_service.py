"""Tests for pattern forecast attachment to symbol intelligence."""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.pattern_forecast_service import (
    build_pattern_trend_forecast,
    pattern_forecast_from_prediction,
)
from data.store import save_raw
from features.build_features import build_and_save_features
from models.pattern_production import (
    PRODUCTION_LABEL_SCHEME,
    PRODUCTION_MODEL_KEY,
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

    save_raw(_synthetic_ohlcv(rows=600), "MSFT")
    build_and_save_features("MSFT")
    train_and_save(
        TrainAndSaveConfig(
            symbols=("MSFT",),
            train_end_date=pd.Timestamp("2021-12-31"),
            artifact_dir=artifact_dir,
            model_config=production_model_config(),
            min_up_prob=0.65,
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


def test_pattern_forecast_from_prediction_binary():
    forecast = pattern_forecast_from_prediction(
        {
            "date": "2024-06-01",
            "label_scheme": PRODUCTION_LABEL_SCHEME.value,
            "prediction": 1,
            "probabilities": {"0": 0.2, "1": 0.8},
            "up_prob": 0.8,
            "ranking_score": 0.8,
            "trade_signal": True,
            "in_training_universe": True,
            "indicators": {"rs_vs_spy_21d": 0.03},
            "model_train_end_date": "2024-05-31",
            "model_key": PRODUCTION_MODEL_KEY,
            "model_label": "Relative strength + trend",
            "training_universe": PRODUCTION_TRAINING_UNIVERSE,
            "n_features": 11,
            "feature_groups": ["relative_strength", "trend"],
            "portfolio_strategy": {
                "strategy_type": "ranking",
                "portfolio_universe": "top20",
                "top_n": 10,
                "rebalance_days": 5,
                "hold_days": 5,
                "max_position_weight": 0.15,
            },
        }
    )

    payload = forecast.model_dump(mode="json", by_alias=True)
    assert payload["asOfDate"] == "2024-06-01"
    assert payload["labelScheme"] == PRODUCTION_LABEL_SCHEME.value
    assert payload["upProb"] == 0.8
    assert payload["rankingScore"] == 0.8
    assert payload["modelKey"] == PRODUCTION_MODEL_KEY
    assert payload["portfolioStrategy"]["strategyType"] == "ranking"
    assert payload["portfolioStrategy"]["topN"] == 10


def test_build_pattern_trend_forecast_returns_none_without_model():
    assert build_pattern_trend_forecast("MSFT", None) is None


def test_build_pattern_trend_forecast_for_trained_symbol(loaded_tradeable_model):
    forecast = build_pattern_trend_forecast("MSFT", loaded_tradeable_model)
    assert forecast is not None
    assert forecast.label_scheme == PRODUCTION_LABEL_SCHEME.value
    assert forecast.in_training_universe is True
