"""Chart Intelligence: visual overlays and concise analyst summary."""

from __future__ import annotations

from typing import Any

from analysis.pattern_intelligence.analyst_summary import build_analyst_summary
from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit
from analysis.pattern_intelligence.chart_analysis import (
    analyze_moving_averages,
    analyze_trend_structure,
    analyze_volume,
    breakout_event_dict,
    build_fib_channel,
    detect_breakout_events,
    find_support_resistance_zones,
)
from analysis.pattern_intelligence.chart_pattern_meta import build_pattern_metadata
from analysis.pattern_intelligence.scoring import PatternScoreBreakdown
from analysis.pattern_intelligence.trend_context import TrendContext


def build_chart_intelligence(
    *,
    symbol: str,
    ohlcv,
    pattern: CandlestickPatternHit | None,
    active_patterns: list[CandlestickPatternHit],
    context: TrendContext,
    scores: PatternScoreBreakdown,
    model_prediction: int | None,
    ranking_score: float | None,
) -> dict[str, Any]:
    structure = analyze_trend_structure(ohlcv)
    supports, resistances = find_support_resistance_zones(ohlcv)
    volume_ctx = analyze_volume(ohlcv, structure)
    ma_ctx = analyze_moving_averages(ohlcv)

    pattern_metadata = [
        _chart_pattern_metadata(ohlcv, hit, structure_bias=structure.bias)
        for hit in active_patterns[:3]
    ]

    summary = build_analyst_summary(
        symbol=symbol,
        pattern=pattern,
        structure=structure,
        supports=supports,
        resistances=resistances,
        context=context,
        volume_ctx=volume_ctx,
        ma_ctx=ma_ctx,
        scores=scores,
        model_prediction=model_prediction,
        ranking_score=ranking_score,
    )

    overlays = build_visual_overlays(
        ohlcv=ohlcv,
        structure=structure,
        supports=supports,
        resistances=resistances,
        ma_ctx=ma_ctx,
        pattern_metadata=pattern_metadata,
    )

    fib_channel = build_fib_channel(ohlcv, structure)
    fib_lines = (fib_channel or {}).get("lines") or []
    if fib_lines:
        trendlines = list(overlays["trendlines"])
        trendlines.extend(fib_lines)
        overlays = {**overlays, "trendlines": trendlines}

    breakout_events = detect_breakout_events(ohlcv, supports, resistances)
    annotations = list(overlays["annotations"])
    for event in breakout_events:
        direction = (
            "bearish"
            if event.kind in {"failed_breakout", "confirmed_breakdown"}
            else "bullish"
        )
        annotations.append(
            {
                "type": "breakout",
                "breakout_kind": event.kind,
                "bar_index": event.bar_index,
                "date": event.date,
                "price": event.price,
                "label": event.label,
                "direction": direction,
                "position": "aboveBar"
                if event.kind in {"failed_breakout", "confirmed_breakout"}
                else "belowBar",
            }
        )

    return {
        "trendlines": overlays["trendlines"],
        "support_zones": overlays["support_zones"],
        "resistance_zones": overlays["resistance_zones"],
        "annotations": annotations,
        "highlighted_candles": overlays["highlighted_candles"],
        "breakout_events": [breakout_event_dict(event) for event in breakout_events],
        "fib_channel": fib_channel,
        "pattern_metadata": pattern_metadata,
        "summary": summary,
    }


def _chart_pattern_metadata(
    ohlcv,
    pattern: CandlestickPatternHit,
    *,
    structure_bias: str,
) -> dict[str, Any]:
    """Pattern overlay metadata without qualification checklist (chart-only)."""
    meta = build_pattern_metadata(ohlcv, pattern, structure_bias=structure_bias)
    return {
        "pattern_id": meta["pattern_id"],
        "label": meta["label"],
        "direction": meta["direction"],
        "confidence": meta["confidence"],
        "quality_score": meta["quality_score"],
        "candle_indexes": meta["candle_indexes"],
        "start_date": meta["start_date"],
        "end_date": meta["end_date"],
        "annotations": meta.get("annotations", []),
        "highlighted_candles": meta.get("highlighted_candles", []),
    }


def build_visual_overlays(
    *,
    ohlcv,
    structure,
    supports,
    resistances,
    ma_ctx,
    pattern_metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    trendlines: list[dict[str, Any]] = []
    if structure.trendline:
        trendlines.append(structure.trendline)

    for key, label, style in [
        ("sma20", "SMA 20", "sma20"),
        ("sma50", "SMA 50", "sma50"),
        ("sma200", "SMA 200", "sma200"),
    ]:
        series = ma_ctx.sma_series.get(key)
        if not series:
            continue
        trendlines.append(
            {
                "label": label,
                "style": style,
                "points": series,
            }
        )

    support_zones = [_zone_dict(zone) for zone in supports]
    resistance_zones = [_zone_dict(zone) for zone in resistances]

    annotations: list[dict[str, Any]] = []
    highlighted: list[dict[str, Any]] = []
    for swing in structure.swing_points[-6:]:
        annotations.append(
            {
                "type": "swing",
                "swing_type": swing.kind,
                "bar_index": swing.bar_index,
                "date": swing.date,
                "price": swing.price,
                "label": "Swing high" if swing.kind == "high" else "Swing low",
            }
        )

    for meta in pattern_metadata:
        annotations.extend(meta.get("annotations", []))
        highlighted.extend(meta.get("highlighted_candles", []))

    return {
        "trendlines": trendlines,
        "support_zones": support_zones,
        "resistance_zones": resistance_zones,
        "annotations": annotations,
        "highlighted_candles": highlighted,
    }


def _zone_dict(zone) -> dict[str, Any]:
    return {
        "price_low": zone.price_low,
        "price_high": zone.price_high,
        "label": zone.label,
        "zone_type": zone.zone_type,
        "touches": zone.touches,
        "strength": zone.strength,
    }
