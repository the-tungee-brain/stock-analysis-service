"""Tests for pattern prediction routes on the main FastAPI app."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.adapters.cache.pattern_analysis_cache import PatternAnalysisCache
from app.auth.dependencies import get_current_user
from app.core.llm_config import settings
from app.main import app
from app.models.user_models import AppUserItem
from app.services.pattern_analysis_service import PatternAnalysisService
from data.store import save_raw
from features.build_features import build_and_save_features
from models.pattern_production import production_model_config
from models.train_and_save import TrainAndSaveConfig, train_and_save
from tests.conftest import seed_pattern_benchmarks
from tests.test_pattern_train_and_save import _synthetic_ohlcv


def _fake_user() -> AppUserItem:
    return AppUserItem(
        id="user-1",
        identity_sub="test-user",
        identity_provider="google",
        email="test@example.com",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.fixture(autouse=True)
def auth_override():
    async def _override_current_user() -> AppUserItem:
        return _fake_user()

    app.dependency_overrides[get_current_user] = _override_current_user
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture(autouse=True)
def jwt_override(monkeypatch):
    monkeypatch.setattr(
        "app.auth.dependencies.verify_jwt",
        lambda token: {"sub": "test-user"},
    )


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


@pytest.fixture(autouse=True)
def pro_user(monkeypatch):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset({"test-user"}))


@pytest.fixture
def pattern_client(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    monkeypatch.setattr("data.paths.RAW_DIR", raw_dir)
    monkeypatch.setattr("data.paths.FEATURES_DIR", features_dir)
    monkeypatch.setattr("models.artifact_store.DEFAULT_ARTIFACT_DIR", artifact_dir)

    seed_pattern_benchmarks(rows=600)
    save_raw(_synthetic_ohlcv(rows=600), "AAPL")
    build_and_save_features("AAPL")
    train_and_save(
        TrainAndSaveConfig(
            symbols=("AAPL",),
            train_end_date=pd.Timestamp("2021-12-31"),
            artifact_dir=artifact_dir,
            model_config=production_model_config(),
            universe="top20",
            extra_metadata={
                "model_key": "C",
                "model_label": "Relative strength + trend",
                "feature_groups": ["relative_strength", "trend"],
                "strategy_type": "ranking",
                "portfolio_universe": "top20",
                "top_n": 10,
                "rebalance_days": 5,
                "hold_days": 5,
                "max_position_weight": 0.15,
            },
        )
    )

    from models.prediction_service import load_deployed_model

    app.state.pattern_loaded_model = load_deployed_model(artifact_dir)
    yield TestClient(app)
    app.state.pattern_loaded_model = None
    if hasattr(app.state, "pattern_analysis_service"):
        delattr(app.state, "pattern_analysis_service")


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value


@pytest.fixture
def pattern_cache_service():
    service = PatternAnalysisService(
        cache=PatternAnalysisCache(redis_client=_FakeRedis(), ttl_seconds=60),
        enabled=True,
    )
    app.state.pattern_analysis_service = service
    yield service
    if hasattr(app.state, "pattern_analysis_service"):
        delattr(app.state, "pattern_analysis_service")


def test_pattern_health_requires_loaded_model(auth_headers):
    app.state.pattern_loaded_model = None
    client = TestClient(app)
    response = client.get("/api/v1/pattern/health", headers=auth_headers)
    assert response.status_code == 503


def test_pattern_predict_returns_payload(pattern_client, auth_headers):
    response = pattern_client.get(
        "/api/v1/pattern/predict",
        params={"symbol": "AAPL"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert payload["prediction"] in (0, 1)
    assert set(payload["probabilities"].keys()) == {"0", "1"}
    assert payload["upProb"] is not None
    assert payload["rankingScore"] is not None
    assert payload["modelKey"] == "C"
    assert payload["portfolioStrategy"]["strategyType"] == "ranking"
    assert "inTrainingUniverse" in payload


def test_pattern_predict_requires_pro(pattern_client, monkeypatch, auth_headers):
    monkeypatch.setattr(settings, "PAID_USER_IDS", frozenset())
    response = pattern_client.get(
        "/api/v1/pattern/predict",
        params={"symbol": "AAPL"},
        headers=auth_headers,
    )
    assert response.status_code == 403


def test_pattern_intelligence_returns_payload(pattern_client, auth_headers):
    response = pattern_client.get(
        "/api/v1/pattern/intelligence",
        params={"symbol": "AAPL"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "AAPL"
    assert "trendContext" in payload
    assert "scores" in payload
    assert payload["scores"]["confirmationScore"] is not None
    assert "explanation" in payload
    assert payload["coreModel"] is not None
    assert payload["scores"]["alignmentState"] in {"confirmed", "conflict", "model_only"}
    assert "primary" in payload["explanation"]["disclaimer"].lower()
    summary = payload["chartIntelligence"]["summary"]
    assert summary["outlook"]["label"]
    assert summary["outlook"]["expectation"]
    assert summary["keyLevel"]["display"]
    assert summary["whyThisOutlook"]
    assert summary["thesis"]
    assert "interpretation" not in payload


def test_pattern_intelligence_cache_hit_avoids_rebuild(
    pattern_client,
    auth_headers,
    pattern_cache_service,
    monkeypatch,
):
    first = pattern_client.get(
        "/api/v1/pattern/intelligence",
        params={"symbol": "AAPL"},
        headers=auth_headers,
    )
    assert first.status_code == 200

    def _fail_build(*args, **kwargs):
        raise AssertionError("pattern intelligence should be served from cache")

    monkeypatch.setattr(
        "app.services.pattern_analysis_service.build_pattern_intelligence",
        _fail_build,
    )

    second = pattern_client.get(
        "/api/v1/pattern/intelligence",
        params={"symbol": "AAPL"},
        headers=auth_headers,
    )
    assert second.status_code == 200
    assert second.json() == first.json()


def test_pattern_predict_after_intelligence_reuses_cached_prediction(
    pattern_client,
    auth_headers,
    pattern_cache_service,
    monkeypatch,
):
    response = pattern_client.get(
        "/api/v1/pattern/intelligence",
        params={"symbol": "AAPL"},
        headers=auth_headers,
    )
    assert response.status_code == 200

    def _fail_predict(*args, **kwargs):
        raise AssertionError("prediction should be served from cache")

    monkeypatch.setattr(
        "app.services.pattern_analysis_service.predict_for_symbol",
        _fail_predict,
    )

    predict = pattern_client.get(
        "/api/v1/pattern/predict",
        params={"symbol": "AAPL"},
        headers=auth_headers,
    )
    assert predict.status_code == 200
    payload = predict.json()
    assert payload["symbol"] == "AAPL"
    assert payload["prediction"] in (0, 1)


def test_pattern_intelligence_after_predict_reuses_cached_analysis(
    pattern_client,
    auth_headers,
    pattern_cache_service,
    monkeypatch,
):
    predict = pattern_client.get(
        "/api/v1/pattern/predict",
        params={"symbol": "AAPL"},
        headers=auth_headers,
    )
    assert predict.status_code == 200

    def _fail_predict(*args, **kwargs):
        raise AssertionError("prediction should be served from cache")

    def _fail_build(*args, **kwargs):
        raise AssertionError("pattern intelligence should be served from cache")

    monkeypatch.setattr(
        "app.services.pattern_analysis_service.predict_for_symbol",
        _fail_predict,
    )
    monkeypatch.setattr(
        "app.services.pattern_analysis_service.build_pattern_intelligence",
        _fail_build,
    )

    intelligence = pattern_client.get(
        "/api/v1/pattern/intelligence",
        params={"symbol": "AAPL"},
        headers=auth_headers,
    )
    assert intelligence.status_code == 200
    assert intelligence.json()["symbol"] == "AAPL"
