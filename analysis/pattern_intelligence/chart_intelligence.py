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
    reference_price = _latest_close(ohlcv)
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
        "reference_price": reference_price,
        "referencePrice": reference_price,
        "selected_levels": _selected_levels(
            supports=overlays["support_zones"],
            resistances=overlays["resistance_zones"],
            reference_price=reference_price,
        ),
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
    actionable_for = zone.actionable_for or {
        "chartContext": True,
        "tradeStop": False,
        "tradeTarget": False,
        "breakoutTrigger": False,
    }
    return {
        "price_low": zone.price_low,
        "price_high": zone.price_high,
        "priceLow": zone.price_low,
        "priceHigh": zone.price_high,
        "midpoint": zone.midpoint,
        "label": zone.label,
        "zone_type": zone.zone_type,
        "type": zone.zone_type,
        "touches": zone.touches,
        "strength": zone.strength,
        "timeframe": zone.timeframe,
        "source": zone.sources[0] if zone.sources else "swing",
        "sources": list(zone.sources),
        "recency_bars": zone.recency_bars,
        "recencyBars": zone.recency_bars,
        "distance_pct_from_current": zone.distance_pct_from_current,
        "distancePctFromCurrent": zone.distance_pct_from_current,
        "atr_distance": zone.atr_distance,
        "atrDistance": zone.atr_distance,
        "level_role": zone.level_role,
        "levelRole": zone.level_role,
        "zone_state": zone.zone_state,
        "zoneState": zone.zone_state,
        "display_level": zone.display_level,
        "displayLevel": zone.display_level,
        "breakout_level": zone.breakout_level,
        "breakoutLevel": zone.breakout_level,
        "actionable_for": actionable_for,
        "actionableFor": actionable_for,
    }


def _latest_close(ohlcv) -> float | None:
    try:
        value = float(ohlcv["close"].iloc[-1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return value if value > 0 else None


def _selected_levels(
    *,
    supports: list[dict[str, Any]],
    resistances: list[dict[str, Any]],
    reference_price: float | None = None,
) -> dict[str, Any]:
    support_levels = sorted(
        [
            level
            for level in supports
            if _is_active_support(level, reference_price=reference_price)
        ],
        key=lambda level: _level_display_price(level) or float("-inf"),
        reverse=True,
    )
    resistance_levels = sorted(
        [
            level
            for level in resistances
            if _is_active_resistance(level, reference_price=reference_price)
        ],
        key=lambda level: _level_display_price(level) or float("inf"),
    )
    nearest_support = support_levels[0] if support_levels else None
    next_support = support_levels[1] if len(support_levels) > 1 else None
    nearest_resistance = resistance_levels[0] if resistance_levels else None
    next_resistance = resistance_levels[1] if len(resistance_levels) > 1 else None
    actionable_support = _first_level(
        support_levels,
        role="actionable",
        actionable_key="tradeStop",
        reference_price=reference_price,
        side="support",
    )
    actionable_resistance = _first_level(
        resistance_levels,
        role="actionable",
        actionable_key="tradeTarget",
        reference_price=reference_price,
        side="resistance",
    )
    major_support = _first_level(supports, role="majorHistorical")
    major_resistance = _first_level(resistances, role="majorHistorical")
    return {
        "reference_price": reference_price,
        "referencePrice": reference_price,
        "nearest_support": nearest_support,
        "nearestSupport": nearest_support,
        "next_support": next_support,
        "nextSupport": next_support,
        "nearest_resistance": nearest_resistance,
        "nearestResistance": nearest_resistance,
        "next_resistance": next_resistance,
        "nextResistance": next_resistance,
        "actionable_support": actionable_support,
        "actionableSupport": actionable_support,
        "actionable_resistance": actionable_resistance,
        "actionableResistance": actionable_resistance,
        "major_support": major_support,
        "majorSupport": major_support,
        "major_resistance": major_resistance,
        "majorResistance": major_resistance,
    }


def _first_level(
    levels: list[dict[str, Any]],
    *,
    role: str | None = None,
    actionable_key: str | None = None,
    allowed_states: set[str] | None = None,
    reference_price: float | None = None,
    side: str | None = None,
) -> dict[str, Any] | None:
    for level in levels:
        if role is not None and level.get("level_role") != role:
            continue
        if allowed_states is not None and level.get("zone_state") not in allowed_states:
            continue
        if side == "support" and not _is_active_support(
            level,
            reference_price=reference_price,
        ):
            continue
        if side == "resistance" and not _is_active_resistance(
            level,
            reference_price=reference_price,
        ):
            continue
        if actionable_key is not None:
            actionable_for = level.get("actionable_for") or {}
            if not actionable_for.get(actionable_key):
                continue
        return level
    return None


def _level_display_price(level: dict[str, Any]) -> float | None:
    for key in ("display_level", "displayLevel", "midpoint"):
        value = level.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    low = level.get("price_low", level.get("priceLow"))
    high = level.get("price_high", level.get("priceHigh"))
    if isinstance(low, (int, float)) and isinstance(high, (int, float)):
        return (float(low) + float(high)) / 2
    return None


def _is_active_support(
    level: dict[str, Any],
    *,
    reference_price: float | None,
) -> bool:
    if level.get("zone_state") != "belowPrice":
        return False
    price = _level_display_price(level)
    return (
        price is not None
        and reference_price is not None
        and price < reference_price
    )


def _is_active_resistance(
    level: dict[str, Any],
    *,
    reference_price: float | None,
) -> bool:
    if level.get("zone_state") != "abovePrice":
        return False
    price = _level_display_price(level)
    breakout = level.get("breakout_level", level.get("breakoutLevel"))
    breakout_price = float(breakout) if isinstance(breakout, (int, float)) else price
    return (
        price is not None
        and reference_price is not None
        and price > reference_price
        and breakout_price > reference_price
    )
