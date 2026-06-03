"""Tests for Chart Intelligence analyst summary."""

from __future__ import annotations

import pandas as pd

from analysis.pattern_intelligence.analyst_summary import build_analyst_summary
from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit
from analysis.pattern_intelligence.chart_analysis import (
    analyze_trend_structure,
    analyze_volume,
    find_support_resistance_zones,
)
from analysis.pattern_intelligence.scoring import build_pattern_scores
from tests.test_pattern_intelligence import build_trend_context_from_frame
from tests.test_pattern_train_and_save import _synthetic_ohlcv


def _build_summary(**kwargs):
    ohlcv = kwargs.pop("ohlcv", _synthetic_ohlcv(rows=400))
    symbol = kwargs.pop("symbol", "MSFT")
    pattern = kwargs.pop("pattern", None)
    model_prediction = kwargs.pop("model_prediction", 1)
    ranking_score = kwargs.pop("ranking_score", 0.55)

    structure = analyze_trend_structure(ohlcv)
    supports, resistances = find_support_resistance_zones(ohlcv)
    volume_ctx = analyze_volume(ohlcv, structure)
    context = build_trend_context_from_frame(ohlcv)
    scores = build_pattern_scores(
        pattern=pattern,
        context=context,
        model_prediction=model_prediction,
        ranking_score=ranking_score,
    )

    return build_analyst_summary(
        symbol=symbol,
        pattern=pattern,
        structure=structure,
        supports=supports,
        resistances=resistances,
        context=context,
        volume_ctx=volume_ctx,
        scores=scores,
        model_prediction=model_prediction,
        ranking_score=ranking_score,
    )


def test_summary_has_four_sections():
    summary = _build_summary()
    assert summary["outlook"]["label"]
    assert summary["outlook"]["expectation"]
    assert summary["key_level"]["display"]
    assert summary["key_level"]["implication"]
    assert 1 <= len(summary["why_this_outlook"]) <= 6
    assert summary["thesis"]
    assert summary["disclaimer"]


def test_outlook_excludes_model_probabilities():
    summary = _build_summary(ranking_score=0.55)
    assert summary["outlook"]["probability"] is None
    assert summary["outlook"]["probability_display"] is None
    assert "model c" not in summary["outlook"]["expectation"].lower()
    assert "%" not in summary["outlook"]["expectation"]
    assert summary["outlook"]["model_context"] is None


def test_evidence_uses_distinct_plain_language():
    summary = _build_summary(ranking_score=0.62)
    texts = {b["text"] for b in summary["why_this_outlook"]}
    joined = " ".join(texts).lower()
    assert "trend" in joined
    assert "leadership" in joined or "market" in joined
    assert not any("Outperforming SPY" in t for t in texts)
    assert not any("Uptrend intact" in t for t in texts)


def test_slight_bearish_inside_uptrend_expectation():
    pattern = CandlestickPatternHit(
        pattern_id="bearish_engulfing",
        label="Bearish Engulfing",
        direction="bearish",
        strength=0.75,
        as_of_date="2024-06-01",
        bar_index=0,
    )
    summary = _build_summary(
        pattern=pattern,
        model_prediction=0,
        ranking_score=0.453,
    )
    assert "bearish" in summary["outlook"]["label"].lower()
    assert "5 sessions" in summary["outlook"]["expectation"]


def test_bullish_thesis_prefers_resistance_key_level():
    summary = _build_summary(ranking_score=0.62, model_prediction=1)
    if summary["key_level"]["level_type"] == "resistance":
        assert "Resistance" in summary["key_level"]["display"]
        assert "break above" in summary["key_level"]["implication"].lower()


def test_evidence_bullets_use_plain_language():
    summary = _build_summary(ranking_score=0.62)
    texts = [b["text"].lower() for b in summary["why_this_outlook"]]
    joined = " ".join(texts)
    assert "sma200" not in joined
    assert "%" not in joined
    assert len(summary["why_this_outlook"]) <= 5


def test_thesis_avoids_restatement_of_evidence():
    summary = _build_summary(ranking_score=0.62)
    thesis = summary["thesis"].lower()
    outlook = summary["outlook"]["expectation"].lower()
    assert "outperform" not in thesis
    assert "strong uptrend" not in thesis
    assert thesis.count("expect modest upside") <= outlook.count("expect modest upside")


def test_key_level_rejects_stale_resistance_far_below_price():
    from analysis.pattern_intelligence.analyst_summary import _key_level_block
    from analysis.pattern_intelligence.chart_analysis import PriceZone

    close = 200.0
    key = _key_level_block(
        thesis_side="bullish",
        supports=[],
        resistances=[
            PriceZone(
                price_low=48.0,
                price_high=52.0,
                label="Resistance: $50.00",
                zone_type="resistance",
                touches=3,
                strength=0.8,
            ),
            PriceZone(
                price_low=205.0,
                price_high=210.0,
                label="Resistance: $207.00",
                zone_type="resistance",
                touches=2,
                strength=0.6,
            ),
        ],
        close=close,
    )
    assert key.get("available") is True
    assert key["price"] == 205.0
    assert "50" not in key["display"]


def test_thesis_mentions_breakout_scenario():
    summary = _build_summary(ranking_score=0.62)
    thesis = summary["thesis"].lower()
    assert "break" in thesis or "weaken" in thesis or "strengthen" in thesis


def test_benchmark_summary():
    summary = _build_summary(symbol="SPY", ranking_score=None, model_prediction=None)
    assert summary["outlook"]["is_benchmark"] is True
    assert summary["outlook"]["probability"] is None
    assert summary["outlook"]["benchmark_notice"]
