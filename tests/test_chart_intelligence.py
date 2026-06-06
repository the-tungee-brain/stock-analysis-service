"""Tests for Chart Intelligence overlays and analyst summary."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.pattern_intelligence.candlestick_engine import (
    active_patterns_on_date,
    scan_candlestick_patterns,
)
from analysis.pattern_intelligence.chart_analysis import (
    _LevelCandidate,
    _cluster_level_candidates,
    PriceZone,
    analyze_moving_averages,
    analyze_trend_structure,
    find_support_resistance_zones,
)
from analysis.pattern_intelligence.chart_intelligence import (
    _selected_levels,
    _zone_dict,
    build_chart_intelligence,
)
from analysis.pattern_intelligence.scoring import build_pattern_scores
from analysis.pattern_intelligence.service import build_pattern_intelligence
from app.models.intelligence_models import ChartIntelligence
from tests.conftest import seed_pattern_benchmarks
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
    assert "selected_levels" in payload
    assert "annotations" in payload
    assert "highlighted_candles" in payload
    assert "pattern_metadata" in payload
    assert payload["summary"]["outlook"]["label"]
    assert payload["summary"]["thesis"]
    assert "scorecard" not in payload
    assert "narrative" not in payload


def test_support_resistance_zones_include_metadata_and_sources():
    ohlcv = _synthetic_ohlcv(rows=400)
    supports, resistances = find_support_resistance_zones(ohlcv)
    zones = supports + resistances
    assert zones
    assert any("priorHighLow" in zone.sources for zone in zones)
    assert any("movingAverage" in zone.sources for zone in zones)
    for zone in zones:
        assert zone.midpoint is not None
        assert zone.timeframe in {"shortTerm", "intermediate", "longTerm"}
        assert zone.level_role in {
            "actionable",
            "nearbyContext",
            "majorHistorical",
        }
        assert zone.actionable_for is not None
        assert zone.actionable_for["chartContext"] is True


def test_chart_intelligence_selected_levels_are_additive():
    ohlcv = _synthetic_ohlcv(rows=400)
    as_of = pd.Timestamp(ohlcv.index[-1])
    scan = scan_candlestick_patterns(ohlcv)
    active = active_patterns_on_date(scan, as_of)
    primary = active[0] if active else None
    context = build_trend_context_from_frame(ohlcv)
    scores = build_pattern_scores(pattern=primary, context=context)

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

    selected = payload["selected_levels"]
    assert {
        "nearest_support",
        "nearestSupport",
        "nearest_resistance",
        "nearestResistance",
        "actionable_support",
        "actionableSupport",
        "actionable_resistance",
        "actionableResistance",
        "major_support",
        "majorSupport",
        "major_resistance",
        "majorResistance",
    }.issubset(set(selected))
    for zone in payload["support_zones"] + payload["resistance_zones"]:
        assert "price_low" in zone
        assert "price_high" in zone
        assert "priceLow" in zone
        assert "priceHigh" in zone
        assert "strength" in zone
        assert "touches" in zone
        assert "type" in zone
        assert "level_role" in zone
        assert "levelRole" in zone
        assert "actionable_for" in zone
        assert "actionableFor" in zone
        assert "zone_state" in zone
        assert "zoneState" in zone
        assert "display_level" in zone
        assert "displayLevel" in zone
        assert "breakout_level" in zone
        assert "breakoutLevel" in zone


def test_inside_resistance_uses_upper_edge_as_breakout_level():
    zone = PriceZone(
        price_low=464.81,
        price_high=478.35,
        label="Resistance: $471.58",
        zone_type="resistance",
        touches=2,
        strength=0.6,
        midpoint=471.58,
        level_role="nearbyContext",
        actionable_for={
            "chartContext": True,
            "tradeStop": False,
            "tradeTarget": False,
            "breakoutTrigger": False,
        },
        zone_state="insideZone",
        display_level=478.35,
        breakout_level=478.35,
    )

    payload = _zone_dict(zone)
    selected = _selected_levels(supports=[], resistances=[payload])

    assert payload["price_low"] == 464.81
    assert selected["nearest_resistance"]["breakout_level"] == 478.35
    assert selected["nearest_resistance"]["breakout_level"] > 466.0


def test_generator_marks_far_historical_support_as_context_only():
    index = pd.date_range("2024-01-01", periods=260, freq="B")
    close = np.linspace(205.0, 500.0, len(index))
    low = close - 4.0
    high = close + 4.0
    low[20] = 198.0
    close[20] = 202.0
    high[20] = 206.0
    ohlcv = pd.DataFrame(
        {
            "open": close - 1.0,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(len(index), 1_000_000.0),
        },
        index=index,
    )

    supports, _ = find_support_resistance_zones(
        ohlcv,
        max_zones=10,
        recent_swing_bars=0,
    )
    historical = [zone for zone in supports if zone.midpoint and zone.midpoint < 225]

    assert historical
    assert all(zone.level_role == "majorHistorical" for zone in historical)
    assert all(not (zone.actionable_for or {}).get("tradeStop") for zone in historical)


def test_round_number_only_level_is_not_actionable():
    supports = _cluster_level_candidates(
        [
            _LevelCandidate(
                price=500.0,
                zone_type="support",
                source="roundNumber",
                bar_index=99,
            )
        ],
        zone_type="support",
        tolerance_pct=0.012,
        max_zones=4,
        close=503.0,
        atr=2.0,
        total_bars=100,
    )

    assert supports
    assert supports[0].level_role != "actionable"
    assert supports[0].actionable_for is not None
    assert supports[0].actionable_for["tradeStop"] is False

    ma_round_cluster = _cluster_level_candidates(
        [
            _LevelCandidate(
                price=500.0,
                zone_type="support",
                source="roundNumber",
                bar_index=99,
            ),
            _LevelCandidate(
                price=500.5,
                zone_type="support",
                source="movingAverage",
                bar_index=99,
            ),
        ],
        zone_type="support",
        tolerance_pct=0.012,
        max_zones=4,
        close=503.0,
        atr=2.0,
        total_bars=100,
    )

    assert ma_round_cluster
    assert ma_round_cluster[0].level_role != "actionable"
    assert ma_round_cluster[0].actionable_for is not None
    assert ma_round_cluster[0].actionable_for["tradeStop"] is False


def test_actionable_support_and_resistance_must_respect_side():
    supports = _cluster_level_candidates(
        [
            _LevelCandidate(
                price=501.0,
                zone_type="support",
                source="swing",
                bar_index=98,
                touches=2,
            )
        ],
        zone_type="support",
        tolerance_pct=0.012,
        max_zones=4,
        close=500.0,
        atr=2.0,
        total_bars=100,
    )
    resistances = _cluster_level_candidates(
        [
            _LevelCandidate(
                price=499.0,
                zone_type="resistance",
                source="swing",
                bar_index=98,
                touches=2,
            )
        ],
        zone_type="resistance",
        tolerance_pct=0.012,
        max_zones=4,
        close=500.0,
        atr=2.0,
        total_bars=100,
    )

    assert supports[0].level_role != "actionable"
    assert supports[0].actionable_for is not None
    assert supports[0].actionable_for["tradeStop"] is False
    assert resistances[0].level_role != "actionable"
    assert resistances[0].actionable_for is not None
    assert resistances[0].actionable_for["tradeTarget"] is False
    assert resistances[0].actionable_for["breakoutTrigger"] is False


def test_chart_intelligence_selected_levels_null_decodes():
    ohlcv = _synthetic_ohlcv(rows=400)
    as_of = pd.Timestamp(ohlcv.index[-1])
    scan = scan_candlestick_patterns(ohlcv)
    active = active_patterns_on_date(scan, as_of)
    primary = active[0] if active else None
    context = build_trend_context_from_frame(ohlcv)
    scores = build_pattern_scores(pattern=primary, context=context)
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
    payload["selected_levels"] = None

    decoded = ChartIntelligence(**payload)

    assert decoded.selected_levels is None


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


def test_service_includes_chart_intelligence_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path / "raw")
    seed_pattern_benchmarks(rows=400)
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
