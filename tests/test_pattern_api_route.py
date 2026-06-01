"""Tests for pattern prediction routes on the main FastAPI app."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.main import app
from data.store import save_raw
from features.build_features import build_and_save_features
from models.train_and_save import TrainAndSaveConfig, train_and_save
from tests.test_pattern_train_and_save import _synthetic_ohlcv


def _fake_user():
    return {"sub": "test-user", "email": "test@example.com"}


@pytest.fixture(autouse=True)
def auth_override():
    app.dependency_overrides[get_current_user] = _fake_user
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def pattern_client(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    monkeypatch.setattr("data.paths.RAW_DIR", raw_dir)
    monkeypatch.setattr("data.paths.FEATURES_DIR", features_dir)
    monkeypatch.setattr("models.artifact_store.DEFAULT_ARTIFACT_DIR", artifact_dir)

    save_raw(_synthetic_ohlcv(), "AAPL")
    build_and_save_features("AAPL")
    train_and_save(
        TrainAndSaveConfig(
            symbols=("AAPL",),
            train_end_date=pd.Timestamp("2021-06-30"),
            artifact_dir=artifact_dir,
        )
    )

    from models.prediction_service import load_deployed_model

    app.state.pattern_loaded_model = load_deployed_model(artifact_dir)
    yield TestClient(app)
    app.state.pattern_loaded_model = None


def test_pattern_health_requires_loaded_model():
    app.state.pattern_loaded_model = None
    client = TestClient(app)
    response = client.get("/api/v1/pattern/health")
    assert response.status_code == 503


def test_pattern_predict_returns_payload(pattern_client):
    response = pattern_client.get("/api/v1/pattern/predict", params={"symbol": "AAPL"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert payload["prediction"] in (-1, 0, 1)
    assert set(payload["probabilities"].keys()) == {"-1", "0", "1"}
    assert "upProb" in payload or "up_prob" in payload
    assert "inTrainingUniverse" in payload or "in_training_universe" in payload
