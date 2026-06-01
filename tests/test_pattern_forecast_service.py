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
from models.labels import LabelScheme
from models.pattern_production import production_model_config
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

    save_raw(_synthetic_ohlcv(), "MSFT")
    build_and_save_features("MSFT")
    train_and_save(
        TrainAndSaveConfig(
            symbols=("MSFT",),
            train_end_date=pd.Timestamp("2021-06-30"),
            artifact_dir=artifact_dir,
            model_config=production_model_config(),
            min_up_prob=0.65,
            universe="tradeable_v1",
        )
    )
    return load_deployed_model(artifact_dir)


def test_pattern_forecast_from_prediction_binary():
    forecast = pattern_forecast_from_prediction(
        {
            "date": "2024-06-01",
            "label_scheme": LabelScheme.BINARY_UPDOWN.value,
            "prediction": 1,
            "probabilities": {"0": 0.2, "1": 0.8},
            "up_prob": 0.8,
            "trade_signal": True,
            "in_training_universe": True,
            "indicators": {"rsi_14": 55.0},
            "model_train_end_date": "2024-05-31",
        }
    )

    payload = forecast.model_dump(mode="json", by_alias=True)
    assert payload["asOfDate"] == "2024-06-01"
    assert payload["labelScheme"] == "binary_updown"
    assert payload["upProb"] == 0.8
    assert payload["tradeSignal"] is True
    assert payload["inTrainingUniverse"] is True


def test_build_pattern_trend_forecast_returns_none_without_model():
    assert build_pattern_trend_forecast("MSFT", None) is None


def test_build_pattern_trend_forecast_for_trained_symbol(loaded_tradeable_model):
    forecast = build_pattern_trend_forecast("MSFT", loaded_tradeable_model)
    assert forecast is not None
    assert forecast.label_scheme == "binary_updown"
    assert forecast.in_training_universe is True
