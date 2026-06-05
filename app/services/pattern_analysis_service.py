from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd

from analysis.pattern_intelligence import build_pattern_intelligence
from app.adapters.cache.pattern_analysis_cache import PatternAnalysisCache
from app.core.latency_observability import observe_dependency, record_dependency_latency
from data.benchmarks import BENCHMARK_SYMBOL, VIX_SYMBOL, ensure_benchmark_ohlcv
from data.loader import load_symbol
from models.prediction_service import LoadedModel, ensure_raw_ohlcv, predict_for_symbol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatternAnalysisSnapshot:
    cache_key: str
    prediction_payload: dict[str, Any]
    pattern_intelligence: dict[str, Any]


class PatternAnalysisService:
    def __init__(
        self,
        *,
        cache: PatternAnalysisCache | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.cache = cache
        self.enabled = _cache_enabled() if enabled is None else enabled

    def get_or_build(
        self,
        symbol: str,
        loaded_model: LoadedModel,
    ) -> PatternAnalysisSnapshot:
        symbol_upper = symbol.strip().upper()
        raw = ensure_raw_ohlcv(symbol_upper)
        ensure_benchmark_ohlcv()
        spy = load_symbol(BENCHMARK_SYMBOL)
        vix = load_symbol(VIX_SYMBOL)
        cache_key = build_pattern_analysis_cache_key(
            symbol=symbol_upper,
            raw_as_of=_latest_as_of(raw),
            model_fingerprint=model_fingerprint(loaded_model),
            spy_as_of=_latest_as_of(spy),
            vix_as_of=_latest_as_of(vix),
        )

        cached = self._get_cached(cache_key)
        if cached is not None:
            return PatternAnalysisSnapshot(
                cache_key=cache_key,
                prediction_payload=dict(cached["prediction_payload"]),
                pattern_intelligence=dict(cached["pattern_intelligence"]),
            )

        with observe_dependency("pattern_analysis_build"):
            prediction_payload = predict_for_symbol(symbol_upper, loaded_model)
            intelligence = build_pattern_intelligence(
                symbol_upper,
                loaded_model=loaded_model,
                raw=raw,
                prediction_payload=prediction_payload,
            ).to_dict()

        snapshot = PatternAnalysisSnapshot(
            cache_key=cache_key,
            prediction_payload=prediction_payload,
            pattern_intelligence=intelligence,
        )
        self._put_cached(snapshot)
        return snapshot

    def get_or_build_prediction_payload(
        self,
        symbol: str,
        loaded_model: LoadedModel,
    ) -> dict[str, Any]:
        try:
            return self.get_or_build(symbol, loaded_model).prediction_payload
        except OSError:
            logger.warning(
                "Pattern analysis build failed during prediction; falling back to prediction only",
                exc_info=True,
            )
            with observe_dependency("pattern_prediction_build"):
                return predict_for_symbol(symbol, loaded_model)

    def _get_cached(self, cache_key: str) -> dict[str, Any] | None:
        if not self.enabled or self.cache is None:
            record_dependency_latency(
                "pattern_analysis_cache",
                0.0,
                cache_status="miss",
            )
            return None
        try:
            cached = self.cache.get(cache_key)
        except Exception:
            logger.warning("Pattern analysis cache read failed", exc_info=True)
            record_dependency_latency(
                "pattern_analysis_cache",
                0.0,
                cache_status="miss",
                error=True,
            )
            return None
        if cached is None:
            record_dependency_latency(
                "pattern_analysis_cache",
                0.0,
                cache_status="miss",
            )
            return None
        record_dependency_latency(
            "pattern_analysis_cache",
            0.0,
            cache_status="hit",
        )
        return cached

    def _put_cached(self, snapshot: PatternAnalysisSnapshot) -> None:
        if not self.enabled or self.cache is None:
            return
        try:
            self.cache.put(
                snapshot.cache_key,
                {
                    "prediction_payload": snapshot.prediction_payload,
                    "pattern_intelligence": snapshot.pattern_intelligence,
                },
            )
        except Exception:
            logger.warning("Pattern analysis cache write failed", exc_info=True)


def build_pattern_analysis_cache_key(
    *,
    symbol: str,
    raw_as_of: str,
    model_fingerprint: str,
    spy_as_of: str,
    vix_as_of: str,
) -> str:
    return ":".join(
        (
            symbol.strip().upper(),
            raw_as_of,
            model_fingerprint,
            spy_as_of,
            vix_as_of,
        )
    )


def model_fingerprint(loaded_model: LoadedModel) -> str:
    metadata = loaded_model.metadata
    fingerprint_payload = {
        "feature_columns": list(loaded_model.feature_columns),
        "model_key": metadata.get("model_key"),
        "model_label": metadata.get("model_label"),
        "train_end_date": metadata.get("train_end_date"),
        "train_start_date": metadata.get("train_start_date"),
        "label_scheme": metadata.get("label_scheme"),
        "min_up_prob": metadata.get("min_up_prob"),
    }
    encoded = json.dumps(fingerprint_payload, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _latest_as_of(frame: pd.DataFrame) -> str:
    if frame.empty:
        raise ValueError("Cannot build pattern analysis cache key from empty OHLCV data")
    return pd.Timestamp(frame.index[-1]).strftime("%Y-%m-%d")


def _cache_enabled() -> bool:
    value = os.getenv("PATTERN_ANALYSIS_CACHE_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}
