from __future__ import annotations

import pandas as pd

from app.adapters.cache.pattern_analysis_cache import PatternAnalysisCache
from app.services import pattern_analysis_service as service_module
from app.services.pattern_analysis_service import (
    PatternAnalysisService,
    build_pattern_analysis_cache_key,
    model_fingerprint,
)
from models.prediction_service import LoadedModel


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value


class _FailingRedis:
    def get(self, key: str) -> str | None:
        raise RuntimeError("redis unavailable")

    def setex(self, key: str, ttl: int, value: str) -> None:
        raise RuntimeError("redis unavailable")


class _PatternResult:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def to_dict(self) -> dict:
        return self._payload


def _frame(as_of: str = "2026-06-04") -> pd.DataFrame:
    index = pd.DatetimeIndex([as_of], name="date")
    return pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1_000_000],
        },
        index=index,
    )


def _loaded_model(label: str = "Relative strength + trend") -> LoadedModel:
    return LoadedModel(
        model=object(),
        metadata={
            "model_key": "C",
            "model_label": label,
            "train_start_date": "2020-01-01",
            "train_end_date": "2025-12-31",
            "label_scheme": "binary",
            "min_up_prob": 0.65,
        },
        feature_columns=["rs_vs_spy_21d"],
    )


def _patch_live_build(monkeypatch, *, calls: dict[str, int]) -> None:
    raw = _frame()

    monkeypatch.setattr(service_module, "ensure_raw_ohlcv", lambda symbol: raw)
    monkeypatch.setattr(service_module, "ensure_benchmark_ohlcv", lambda: None)
    monkeypatch.setattr(service_module, "load_benchmark_ohlcv", lambda symbol: raw)

    def _predict(symbol: str, loaded: LoadedModel) -> dict:
        calls["predict"] = calls.get("predict", 0) + 1
        return {
            "symbol": symbol,
            "date": "2026-06-04",
            "prediction": 1,
            "up_prob": 0.72,
            "ranking_score": 0.72,
            "model_key": "C",
            "model_label": "Relative strength + trend",
            "in_training_universe": True,
            "indicators": {"rs_vs_spy_21d": 0.05},
        }

    def _intelligence(
        symbol: str,
        *,
        loaded_model: LoadedModel | None = None,
        raw: pd.DataFrame | None = None,
        prediction_payload: dict | None = None,
    ) -> _PatternResult:
        calls["intelligence"] = calls.get("intelligence", 0) + 1
        return _PatternResult(
            {
                "symbol": symbol,
                "as_of_date": "2026-06-04",
                "core_model": prediction_payload,
                "trend_context": {"as_of_date": "2026-06-04"},
            }
        )

    monkeypatch.setattr(service_module, "predict_for_symbol", _predict)
    monkeypatch.setattr(service_module, "build_pattern_intelligence", _intelligence)


def test_cache_key_changes_when_inputs_change():
    base = {
        "symbol": "AAPL",
        "raw_as_of": "2026-06-04",
        "model_fingerprint": "model-a",
        "spy_as_of": "2026-06-04",
        "vix_as_of": "2026-06-04",
    }
    base_key = build_pattern_analysis_cache_key(**base)

    for field, value in {
        "raw_as_of": "2026-06-05",
        "model_fingerprint": "model-b",
        "spy_as_of": "2026-06-05",
        "vix_as_of": "2026-06-05",
    }.items():
        changed = {**base, field: value}
        assert build_pattern_analysis_cache_key(**changed) != base_key


def test_model_fingerprint_changes_with_model_metadata():
    assert model_fingerprint(_loaded_model("Model A")) != model_fingerprint(
        _loaded_model("Model B")
    )


def test_corrupt_cache_entry_rebuilds(monkeypatch):
    redis_client = _FakeRedis()
    cache = PatternAnalysisCache(redis_client=redis_client, ttl_seconds=60)
    service = PatternAnalysisService(cache=cache, enabled=True)
    redis_client.store[
        cache.redis_key(
            build_pattern_analysis_cache_key(
                symbol="AAPL",
                raw_as_of="2026-06-04",
                model_fingerprint=model_fingerprint(_loaded_model()),
                spy_as_of="2026-06-04",
                vix_as_of="2026-06-04",
            )
        )
    ] = "{not-json"
    calls: dict[str, int] = {}
    _patch_live_build(monkeypatch, calls=calls)

    snapshot = service.get_or_build("AAPL", _loaded_model())

    assert snapshot.prediction_payload["symbol"] == "AAPL"
    assert calls == {"predict": 1, "intelligence": 1}


def test_redis_failure_falls_back_to_live_build(monkeypatch):
    cache = PatternAnalysisCache(redis_client=_FailingRedis(), ttl_seconds=60)
    service = PatternAnalysisService(cache=cache, enabled=True)
    calls: dict[str, int] = {}
    _patch_live_build(monkeypatch, calls=calls)

    snapshot = service.get_or_build("AAPL", _loaded_model())

    assert snapshot.prediction_payload["symbol"] == "AAPL"
    assert calls == {"predict": 1, "intelligence": 1}


def test_cache_hit_reuses_prediction_and_intelligence(monkeypatch):
    cache = PatternAnalysisCache(redis_client=_FakeRedis(), ttl_seconds=60)
    service = PatternAnalysisService(cache=cache, enabled=True)
    calls: dict[str, int] = {}
    _patch_live_build(monkeypatch, calls=calls)

    first = service.get_or_build("AAPL", _loaded_model())
    second = service.get_or_build("AAPL", _loaded_model())

    assert first.prediction_payload == second.prediction_payload
    assert first.pattern_intelligence == second.pattern_intelligence
    assert calls == {"predict": 1, "intelligence": 1}
