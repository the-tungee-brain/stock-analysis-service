"""End-to-end tests for artifact training and the prediction API."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from data.store import save_raw
from features.build_features import build_and_save_features
from models.train_and_save import TrainAndSaveConfig, train_and_save
from models.xgb_model import XGBModelConfig
from tests.test_pattern_train_and_save import _synthetic_ohlcv


@pytest.fixture
def trained_api_client(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    artifact_dir = tmp_path / "artifacts"
    monkeypatch.setattr("data.paths.RAW_DIR", raw_dir)
    monkeypatch.setattr("data.paths.FEATURES_DIR", features_dir)

    save_raw(_synthetic_ohlcv(rows=600), "AAPL")
    build_and_save_features("AAPL")

    train_and_save(
        TrainAndSaveConfig(
            symbols=("AAPL",),
            train_end_date=pd.Timestamp("2021-12-31"),
            artifact_dir=artifact_dir,
            model_config=XGBModelConfig(n_estimators=10, max_depth=2, random_state=0),
        )
    )

    app = create_app(artifact_dir=artifact_dir)
    with TestClient(app) as client:
        yield client


def test_health_returns_model_metadata(trained_api_client):
    response = trained_api_client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model"]["train_end_date"] <= "2021-12-31"
    assert payload["model"]["n_features"] > 0
    assert "AAPL" in payload["model"]["symbols"]


def test_predict_returns_prediction_payload(trained_api_client):
    response = trained_api_client.get("/predict", params={"symbol": "AAPL"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert payload["date"]
    assert payload["prediction"] in (-1, 0, 1)
    assert set(payload["probabilities"].keys()) == {"-1", "0", "1"}
    assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-6
    for key in ("rs_vs_spy_21d", "close_vs_sma20", "ret_21d"):
        assert key in payload["indicators"]


def test_predict_unknown_symbol_returns_404(trained_api_client):
    response = trained_api_client.get("/predict", params={"symbol": "ZZZZ"})
    assert response.status_code == 404
