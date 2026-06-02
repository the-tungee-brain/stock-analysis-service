"""Tests for Chart Intelligence overlays and analyst summary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.pattern_intelligence.candlestick_engine import (
    active_patterns_on_date,
    scan_candlestick_patterns,
)
from analysis.pattern_intelligence.chart_analysis import (
    analyze_moving_averages,
    analyze_trend_structure,
    find_support_resistance_zones,
)
from analysis.pattern_intelligence.chart_intelligence import build_chart_intelligence
from analysis.pattern_intelligence.scoring import build_pattern_scores
from analysis.pattern_intelligence.service import build_pattern_intelligence
from tests.test_pattern_intelligence import build_trend_context_from_frame
from tests.test_pattern_train_and_save import _synthetic_ohlcv


def test_trend_structure_detects_swings():
    ohlcv = _synthetic_ohlcv(rows=400)
    structure = analyze_trend_structure(ohlcv)
    assert structure.bias in {"uptrend", "downtrend", "mixed"}
    assert structure.summary
    assert len(structure.swing_points) >= 1


def test_support_resistance_zones_return_levels():
    ohlcv = _synthetic_ohlcv(rows=400)
    supports, resistances = find_support_resistance_zones(ohlcv)
    assert isinstance(supports, list)
    assert isinstance(resistances, list)


def test_moving_average_context_includes_sma20():
    ohlcv = _synthetic_ohlcv(rows=400)
    ma = analyze_moving_averages(ohlcv)
    assert ma.sma_20 is not None
    assert ma.sma_50 is not None
    assert ma.sma_200 is not None
    assert "sma20" in ma.sma_series


def test_chart_intelligence_payload_shape():
    ohlcv = _synthetic_ohlcv(rows=400)
    as_of = pd.Timestamp(ohlcv.index[-1])
    scan = scan_candlestick_patterns(ohlcv)
    active = active_patterns_on_date(scan, as_of)
    primary = active[0] if active else None
    context = build_trend_context_from_frame(ohlcv)
    scores = build_pattern_scores(
        pattern=primary,
        context=context,
        model_prediction=1,
        ranking_score=0.62,
    )

    payload = build_chart_intelligence(
        symbol="MSFT",
        ohlcv=ohlcv,
        pattern=primary,
        active_patterns=active,
        context=context,
        scores=scores,
        model_prediction=1,
        ranking_score=0.62,
    )

    assert "trendlines" in payload
    assert "support_zones" in payload
    assert "resistance_zones" in payload
    assert "annotations" in payload
    assert "highlighted_candles" in payload
    assert "pattern_metadata" in payload
    assert payload["summary"]["outlook"]["label"]
    assert payload["summary"]["thesis"]
    assert "scorecard" not in payload
    assert "narrative" not in payload


def test_chart_intelligence_includes_breakout_and_fib_channel_keys():
    ohlcv = _synthetic_ohlcv(rows=400)
    as_of = pd.Timestamp(ohlcv.index[-1])
    scan = scan_candlestick_patterns(ohlcv)
    active = active_patterns_on_date(scan, as_of)
    primary = active[0] if active else None
    context = build_trend_context_from_frame(ohlcv)
    scores = build_pattern_scores(
        pattern=primary,
        context=context,
        model_prediction=1,
        ranking_score=0.62,
    )

    payload = build_chart_intelligence(
        symbol="MSFT",
        ohlcv=ohlcv,
        pattern=primary,
        active_patterns=active,
        context=context,
        scores=scores,
        model_prediction=1,
        ranking_score=0.62,
    )

    assert "breakout_events" in payload
    assert "fib_channel" in payload
    assert isinstance(payload["breakout_events"], list)


def test_detect_failed_breakout_on_synthetic_pierce_and_reject():
    from analysis.pattern_intelligence.chart_analysis import (
        PriceZone,
        detect_breakout_events,
    )

    index = pd.date_range("2024-01-01", periods=40, freq="B")
    close = np.full(40, 100.0)
    high = close + 1.0
    low = close - 1.0
    close[25] = 106.0
    high[25] = 108.0
    close[26] = 103.0
    high[26] = 104.0
    close[27] = 101.0
    high[27] = 102.0
    ohlcv = pd.DataFrame(
        {
            "open": close - 0.5,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(40, 1_000_000.0),
        },
        index=index,
    )
    resistance = PriceZone(
        price_low=104.5,
        price_high=105.0,
        label="Resistance: $105.00",
        zone_type="resistance",
        touches=2,
        strength=0.7,
    )
    events = detect_breakout_events(ohlcv, [], [resistance], recent_only_bars=40)
    kinds = {event.kind for event in events}
    assert "failed_breakout" in kinds


def test_build_fib_channel_returns_parallel_lines():
    from analysis.pattern_intelligence.chart_analysis import analyze_trend_structure, build_fib_channel

    ohlcv = _synthetic_ohlcv(rows=400)
    structure = analyze_trend_structure(ohlcv)
    channel = build_fib_channel(ohlcv, structure)
    if channel is None:
        return
    assert channel["lines"]
    assert len(channel["lines"]) >= 3
    assert all(line["points"] for line in channel["lines"])


def test_service_includes_chart_intelligence_summary():
    ohlcv = _synthetic_ohlcv(rows=400)
    result = build_pattern_intelligence("MSFT", raw=ohlcv)
    assert result.chart_intelligence
    assert result.chart_intelligence["summary"]["outlook"]["expectation"]
    assert result.chart_intelligence["summary"]["why_this_outlook"]


def test_pattern_metadata_omits_qualification_checks():
    ohlcv = _synthetic_ohlcv(rows=400)
    as_of = pd.Timestamp(ohlcv.index[-1])
    scan = scan_candlestick_patterns(ohlcv)
    active = active_patterns_on_date(scan, as_of)
    if not active:
        return
    context = build_trend_context_from_frame(ohlcv)
    scores = build_pattern_scores(pattern=active[0], context=context, model_prediction=1)
    payload = build_chart_intelligence(
        symbol="MSFT",
        ohlcv=ohlcv,
        pattern=active[0],
        active_patterns=active,
        context=context,
        scores=scores,
        model_prediction=1,
        ranking_score=0.62,
    )
    meta = payload["pattern_metadata"][0]
    assert "quality_score" in meta
    assert "qualification_checks" not in meta
