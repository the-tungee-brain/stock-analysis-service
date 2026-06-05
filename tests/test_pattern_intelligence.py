"""Tests for Pattern Intelligence layer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.pattern_intelligence.candlestick_engine import (
    PATTERN_CATALOG,
    active_patterns_on_date,
    scan_candlestick_patterns,
)
from analysis.pattern_intelligence.explanation import build_pattern_explanation
from analysis.pattern_intelligence.scoring import build_pattern_scores
from analysis.pattern_intelligence.service import build_pattern_intelligence
from app.services.pattern_intelligence_service import pattern_intelligence_from_dict
from models.pattern_production import (
    PRODUCTION_TRAINING_UNIVERSE,
    production_model_config,
    production_train_metadata_kwargs,
)
from models.prediction_service import load_deployed_model
from models.train_and_save import TrainAndSaveConfig, train_and_save
from tests.conftest import seed_pattern_benchmarks
from tests.test_pattern_train_and_save import _synthetic_ohlcv


def _hammer_bar(index: pd.DatetimeIndex) -> pd.DataFrame:
    close = pd.Series([100.0, 99.0, 98.5, 98.0, 99.5], index=index[:5])
    open_ = pd.Series([100.0, 99.2, 98.7, 98.3, 99.3], index=index[:5])
    high = open_ + 0.3
    low = pd.Series([99.8, 97.0, 96.5, 96.0, 98.5], index=index[:5])
    volume = pd.Series([1_000_000] * 5, index=index[:5])
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index[:5],
    )


def test_scan_covers_all_catalog_patterns():
    rows = _synthetic_ohlcv(rows=400)
    scan = scan_candlestick_patterns(rows)
    for pattern_id in PATTERN_CATALOG:
        assert f"hit_{pattern_id}" in scan.columns
        assert f"strength_{pattern_id}" in scan.columns


def test_hammer_detection_on_synthetic_sequence():
    index = pd.date_range("2024-01-01", periods=5, freq="B", name="date")
    df = _hammer_bar(index)
    scan = scan_candlestick_patterns(df)
    hits = active_patterns_on_date(scan, index[-1])
    assert any(hit.pattern_id == "hammer" for hit in hits)


def test_scoring_defers_to_model_alignment():
    context = build_trend_context_from_frame(_synthetic_ohlcv(rows=400))
    scores = build_pattern_scores(
        pattern=None,
        context=context,
        model_prediction=1,
        model_up_prob=0.7,
    )
    assert scores.confidence == "model_only"
    assert scores.alignment_state == "model_only"
    assert 0.0 <= scores.confirmation_score <= 1.0


def test_explanation_mentions_core_model():
    context = build_trend_context_from_frame(_synthetic_ohlcv(rows=400))
    scores = build_pattern_scores(pattern=None, context=context, model_prediction=1)
    text = build_pattern_explanation(
        symbol="MSFT",
        pattern=None,
        context=context,
        scores=scores,
        history=None,
        model_label="Relative strength + trend",
        model_prediction=1,
        ranking_score=0.62,
    )
    assert "Model C" in text["model_context"]
    assert "primary alpha" in text["disclaimer"].lower() or "primary" in text["disclaimer"]


def test_pattern_intelligence_service_roundtrip(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "artifacts"
    raw_dir = tmp_path / "raw"
    features_dir = tmp_path / "features"
    monkeypatch.setattr("data.paths.RAW_DIR", raw_dir)
    monkeypatch.setattr("data.paths.FEATURES_DIR", features_dir)
    monkeypatch.setattr("models.artifact_store.DEFAULT_ARTIFACT_DIR", artifact_dir)

    from data.store import save_raw
    from features.build_features import build_and_save_features

    seed_pattern_benchmarks(rows=600)
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
    loaded = load_deployed_model(artifact_dir)
    result = build_pattern_intelligence("MSFT", loaded_model=loaded)
    api_model = pattern_intelligence_from_dict(result.to_dict())
    dumped = api_model.model_dump(mode="json", by_alias=True)
    assert dumped["symbol"] == "MSFT"
    assert "trendContext" in dumped
    assert "scores" in dumped
    assert "explanation" in dumped
    assert dumped["scores"]["alignmentState"] in {"confirmed", "conflict", "model_only"}
    assert "setupOutcome" in dumped or dumped.get("setupOutcome") is None
    assert "chartIntelligence" in dumped
    summary = dumped["chartIntelligence"]["summary"]
    assert summary["outlook"]["label"]
    assert summary["thesis"]


def test_derive_alignment_state():
    from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit
    from analysis.pattern_intelligence.scoring import derive_alignment_state

    pattern = CandlestickPatternHit(
        pattern_id="hammer",
        label="Hammer",
        direction="bullish",
        strength=0.8,
        as_of_date="2024-01-01",
        bar_index=0,
    )
    assert derive_alignment_state(pattern=pattern, model_prediction=1, model_alignment=0.85) == "confirmed"
    assert derive_alignment_state(pattern=pattern, model_prediction=0, model_alignment=0.25) == "conflict"
    assert derive_alignment_state(pattern=None, model_prediction=1, model_alignment=0.5) == "model_only"


def build_trend_context_from_frame(df: pd.DataFrame):
    from analysis.pattern_intelligence.trend_context import TrendContext

    close = float(df["close"].iloc[-1])
    return TrendContext(
        as_of_date=pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d"),
        close=close,
        sma_50=close * 0.98,
        sma_200=close * 0.95,
        above_sma_50=True,
        above_sma_200=True,
        rs_vs_spy_21d=0.02,
        rs_vs_spy_63d=0.04,
        rs_vs_spy_126d=0.01,
        vol_ratio_20d=1.2,
        vol_zscore_20d=0.5,
    )
