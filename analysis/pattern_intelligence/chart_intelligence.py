"""Chart Intelligence: scorecard, narrative, and visual overlay assembly."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from analysis.pattern_intelligence.benchmarks import is_model_benchmark_symbol
from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit
from analysis.pattern_intelligence.chart_analysis import (
    BreakoutEvent,
    MovingAverageContext,
    PriceZone,
    TrendStructure,
    VolumeContext,
    analyze_moving_averages,
    analyze_trend_structure,
    analyze_volume,
    detect_breakout_events,
    find_support_resistance_zones,
)
from analysis.pattern_intelligence.chart_pattern_meta import build_pattern_metadata
from analysis.pattern_intelligence.chart_replay import build_pattern_replay
from analysis.pattern_intelligence.scoring import PatternScoreBreakdown
from analysis.pattern_intelligence.trend_context import TrendContext

ScoreBias = Literal["bullish", "neutral", "bearish"]


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
    volume_ctx = analyze_volume(
        ohlcv,
        structure,
        pattern_bar_index=pattern.bar_index if pattern else None,
    )
    ma_ctx = analyze_moving_averages(ohlcv)
    breakouts = detect_breakout_events(ohlcv, supports, resistances)

    pattern_metadata = [
        build_pattern_metadata(
            ohlcv,
            hit,
            structure_bias=structure.bias,
            volume_note=volume_ctx.pattern_volume_note,
            volume_confirmed=volume_ctx.pattern_volume_confirmed,
        )
        for hit in active_patterns[:3]
    ]

    pattern_replay = (
        build_pattern_replay(
            ohlcv,
            pattern.pattern_id,
            current_bar_index=pattern.bar_index,
        )
        if pattern is not None
        else None
    )

    scorecard = build_market_structure_scorecard(
        symbol=symbol,
        structure=structure,
        supports=supports,
        resistances=resistances,
        context=context,
        volume_ctx=volume_ctx,
        ma_ctx=ma_ctx,
        pattern=pattern,
        scores=scores,
        model_prediction=model_prediction,
        ranking_score=ranking_score,
    )

    trade_thesis = build_trade_thesis_panel(
        symbol=symbol,
        pattern=pattern,
        structure=structure,
        supports=supports,
        resistances=resistances,
        context=context,
        volume_ctx=volume_ctx,
        ma_ctx=ma_ctx,
        pattern_metadata=pattern_metadata[0] if pattern_metadata else None,
        pattern_replay=pattern_replay,
        scores=scores,
        model_prediction=model_prediction,
        ranking_score=ranking_score,
    )

    chart_score = compute_chart_intelligence_score(
        structure=structure,
        context=context,
        supports=supports,
        resistances=resistances,
        volume_ctx=volume_ctx,
        pattern=pattern,
        pattern_quality=(
            pattern_metadata[0]["quality_score"] if pattern_metadata else None
        ),
        scores=scores,
        symbol=symbol,
        model_prediction=model_prediction,
        ranking_score=ranking_score,
    )

    decision_hierarchy = build_decision_hierarchy(is_benchmark=is_model_benchmark_symbol(symbol))

    overlays = build_visual_overlays(
        ohlcv=ohlcv,
        structure=structure,
        supports=supports,
        resistances=resistances,
        ma_ctx=ma_ctx,
        pattern_metadata=pattern_metadata,
        breakouts=breakouts,
        volume_ctx=volume_ctx,
    )

    narrative = build_trader_narrative(
        symbol=symbol,
        pattern=pattern,
        structure=structure,
        supports=supports,
        resistances=resistances,
        context=context,
        volume_ctx=volume_ctx,
        ma_ctx=ma_ctx,
        scorecard=scorecard,
        trade_thesis=trade_thesis,
        model_prediction=model_prediction,
        ranking_score=ranking_score,
    )

    return {
        "trendlines": overlays["trendlines"],
        "support_zones": overlays["support_zones"],
        "resistance_zones": overlays["resistance_zones"],
        "annotations": overlays["annotations"],
        "highlighted_candles": overlays["highlighted_candles"],
        "structure_labels": list(structure.structure_labels),
        "pattern_metadata": pattern_metadata,
        "breakouts": [_breakout_dict(event) for event in breakouts],
        "volume_markers": overlays["volume_markers"],
        "structure": _structure_dict(structure),
        "moving_averages": _ma_dict(ma_ctx),
        "volume": _volume_dict(volume_ctx),
        "support_resistance_summary": _sr_summary(supports, resistances, context.close),
        "relative_strength": _rs_dict(context),
        "narrative": narrative,
        "scorecard": scorecard,
        "trade_thesis": trade_thesis,
        "decision_hierarchy": decision_hierarchy,
        "chart_intelligence_score": chart_score,
        "pattern_replay": pattern_replay,
    }


def build_market_structure_scorecard(
    *,
    symbol: str,
    structure: TrendStructure,
    supports: list[PriceZone],
    resistances: list[PriceZone],
    context: TrendContext,
    volume_ctx: VolumeContext,
    ma_ctx: MovingAverageContext,
    pattern: CandlestickPatternHit | None,
    scores: PatternScoreBreakdown,
    model_prediction: int | None,
    ranking_score: float | None,
) -> dict[str, Any]:
    is_benchmark = is_model_benchmark_symbol(symbol)

    trend = _bias_from_structure(structure)
    sr = _bias_from_zones(context.close, supports, resistances)
    rs = _bias_from_rs(context, is_benchmark)
    volume = _bias_from_volume(volume_ctx)
    pattern_bias = _bias_from_pattern(pattern, structure)
    model_c = _bias_from_model(model_prediction, ranking_score, is_benchmark)

    rows = [
        _score_row("Market regime", _bias_from_ma_regime(ma_ctx), ma_ctx.summary),
        _score_row("Trend structure", trend, structure.summary),
        _score_row("Relative strength", rs, _rs_dict(context)["summary"]),
        _score_row("Support / resistance", sr, _sr_summary(supports, resistances, context.close)),
        _score_row("Volume", volume, volume_ctx.summary),
        _score_row(
            "Pattern",
            _pattern_context_bias(pattern, structure),
            _pattern_context_detail(pattern, structure, pattern_bias),
        ),
    ]
    if not is_benchmark:
        rows.append(
            _score_row(
                "Model C",
                model_c,
                _model_summary(model_prediction, ranking_score),
            )
        )

    thesis = _composite_thesis(rows, is_benchmark)
    return {
        "rows": rows,
        "thesis": thesis,
        "priority_order": [
            "market_regime",
            "trend_structure",
            "relative_strength",
            "support_resistance",
            "volume",
            "candlestick_patterns",
            "model_c",
        ],
    }


def build_trader_narrative(
    *,
    symbol: str,
    pattern: CandlestickPatternHit | None,
    structure: TrendStructure,
    supports: list[PriceZone],
    resistances: list[PriceZone],
    context: TrendContext,
    volume_ctx: VolumeContext,
    ma_ctx: MovingAverageContext,
    scorecard: dict[str, Any],
    trade_thesis: dict[str, Any],
    model_prediction: int | None,
    ranking_score: float | None,
) -> dict[str, Any]:
    action = scorecard["thesis"]["action"]
    return {
        "summary": trade_thesis["final_thesis"],
        "action": action,
        "headline": trade_thesis["headline"],
        "disclaimer": (
            "Chart intelligence is contextual technical analysis, not investment advice. "
            "Market structure and trend drive the thesis; patterns provide context only."
        ),
    }


def build_trade_thesis_panel(
    *,
    symbol: str,
    pattern: CandlestickPatternHit | None,
    structure: TrendStructure,
    supports: list[PriceZone],
    resistances: list[PriceZone],
    context: TrendContext,
    volume_ctx: VolumeContext,
    ma_ctx: MovingAverageContext,
    pattern_metadata: dict[str, Any] | None,
    pattern_replay: dict[str, Any] | None,
    scores: PatternScoreBreakdown,
    model_prediction: int | None,
    ranking_score: float | None,
) -> dict[str, Any]:
    is_benchmark = is_model_benchmark_symbol(symbol)
    bull_case: list[str] = []
    bear_case: list[str] = []

    if ma_ctx.above_sma_200:
        bull_case.append("Above SMA200 — long-term trend supportive.")
    elif ma_ctx.above_sma_200 is False:
        bear_case.append("Below SMA200 — long-term trend remains heavy.")

    if structure.bias == "uptrend":
        bull_case.append("Higher highs and higher lows — uptrend structure intact.")
    elif structure.bias == "downtrend":
        bear_case.append("Lower highs and lower lows — downtrend structure in control.")

    if structure.exhaustion:
        bear_case.append("Momentum slowing at recent swing highs/lows.")

    if not is_benchmark and context.rs_vs_spy_63d is not None:
        pct = context.rs_vs_spy_63d * 100
        if pct > 1.5:
            bull_case.append(f"RS outperforming SPY ({pct:+.1f}% over 63d).")
        elif pct < -1.5:
            bear_case.append(f"RS lagging SPY ({pct:+.1f}% over 63d).")

    if resistances and context.close >= resistances[0].price_low * 0.98:
        bear_case.append(
            f"Resistance nearby ({resistances[0].label.split(':')[0]})."
        )
    if supports and context.close <= supports[0].price_high * 1.02:
        bull_case.append(f"Support holding near ${supports[0].price_low:.2f}.")

    if volume_ctx.breakout_confirmed:
        bull_case.append("Recent breakout confirmed by volume.")
    if volume_ctx.pattern_volume_absent and pattern is not None:
        bear_case.append(
            f"{pattern.label} detected but volume confirmation absent."
        )

    if pattern is not None:
        if pattern.direction == "bearish" and structure.bias == "uptrend":
            bear_case.append(f"{pattern.label} formed — appears counter-trend.")
        elif pattern.direction == "bullish" and structure.bias == "downtrend":
            bull_case.append(f"{pattern.label} formed against bearish structure.")
        elif pattern.direction == "bearish":
            bear_case.append(f"{pattern.label} aligns with bearish structure.")
        elif pattern.direction == "bullish":
            bull_case.append(f"{pattern.label} aligns with bullish structure.")

    if structure.bias == "uptrend" and pattern is not None and pattern.direction == "bearish":
        final = (
            "Trend remains bullish. Bearish pattern appears counter-trend — "
            "probability favors continuation unless structure breaks."
        )
        headline = "Primary thesis: bullish (pattern is context only)"
    elif structure.bias == "downtrend" and pattern is not None and pattern.direction == "bullish":
        final = (
            "Trend remains bearish. Bullish pattern is counter-trend until "
            "structure repairs with higher highs and higher lows."
        )
        headline = "Primary thesis: bearish (pattern is context only)"
    elif structure.bias == "uptrend":
        final = "Trend remains bullish. Favor holding/adds while structure holds."
        headline = "Primary thesis: bullish"
    elif structure.bias == "downtrend":
        final = "Trend remains bearish. Favor reducing risk or waiting for repair."
        headline = "Primary thesis: bearish"
    else:
        final = "Mixed structure — wait for trend resolution before sizing up."
        headline = "Primary thesis: neutral / wait"

    replay_note = None
    if pattern_replay and pattern_replay.get("occurrences", 0) >= 3:
        replay_note = pattern_replay.get("summary")

    return {
        "headline": headline,
        "bull_case": bull_case[:5],
        "bear_case": bear_case[:5],
        "final_thesis": final,
        "pattern_proof": pattern_metadata.get("proof_mode") if pattern_metadata else None,
        "pattern_replay": pattern_replay,
        "replay_note": replay_note,
    }


def compute_chart_intelligence_score(
    *,
    structure: TrendStructure,
    context: TrendContext,
    supports: list[PriceZone],
    resistances: list[PriceZone],
    volume_ctx: VolumeContext,
    pattern: CandlestickPatternHit | None,
    pattern_quality: int | None,
    scores: PatternScoreBreakdown,
    symbol: str,
    model_prediction: int | None,
    ranking_score: float | None,
) -> dict[str, Any]:
    is_benchmark = is_model_benchmark_symbol(symbol)

    trend_component = _component_from_structure(structure)
    rs_component = _component_from_rs(context, is_benchmark)
    sr_component = _component_from_zones(context.close, supports, resistances)
    volume_component = _component_from_volume(volume_ctx)
    pattern_component = pattern_quality if pattern_quality is not None else 50
    model_component = _component_from_model(model_prediction, ranking_score, is_benchmark)

    total = (
        trend_component * 0.35
        + rs_component * 0.25
        + sr_component * 0.15
        + volume_component * 0.10
        + pattern_component * 0.10
        + model_component * 0.05
    )
    score = int(round(np.clip(total, 0, 100)))

    return {
        "score": score,
        "label": _score_label(score),
        "weights": {
            "trend_structure": 0.35,
            "relative_strength": 0.25,
            "support_resistance": 0.15,
            "volume": 0.10,
            "pattern_quality": 0.10,
            "model_alignment": 0.05,
        },
        "components": {
            "trend_structure": round(trend_component, 1),
            "relative_strength": round(rs_component, 1),
            "support_resistance": round(sr_component, 1),
            "volume": round(volume_component, 1),
            "pattern_quality": round(float(pattern_component), 1),
            "model_alignment": round(model_component, 1),
        },
    }


def build_decision_hierarchy(*, is_benchmark: bool) -> dict[str, Any]:
    layers = [
        {"rank": 1, "key": "market_regime", "label": "Market regime", "note": "Long-term trend and moving-average context."},
        {"rank": 2, "key": "trend_structure", "label": "Trend structure", "note": "HH/HL/LH/LL sequence and trendlines."},
        {"rank": 3, "key": "relative_strength", "label": "Relative strength", "note": "Performance vs SPY."},
        {"rank": 4, "key": "support_resistance", "label": "Support / resistance", "note": "Major levels and breakouts."},
        {"rank": 5, "key": "volume", "label": "Volume", "note": "Confirmation or lack of participation."},
        {"rank": 6, "key": "candlestick_patterns", "label": "Candlestick patterns", "note": "Context only — never overrides strong trend alone."},
    ]
    if not is_benchmark:
        layers.append(
            {"rank": 7, "key": "model_c", "label": "Model C", "note": "5-day ranking overlay."}
        )
    return {
        "layers": layers,
        "rule": "Patterns never override strong trend structure alone.",
    }


def build_visual_overlays(
    *,
    ohlcv,
    structure: TrendStructure,
    supports: list[PriceZone],
    resistances: list[PriceZone],
    ma_ctx: MovingAverageContext,
    pattern_metadata: list[dict[str, Any]],
    breakouts: list[BreakoutEvent],
    volume_ctx: VolumeContext,
) -> dict[str, Any]:
    start_date = pd.Timestamp(ohlcv.index[0]).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(ohlcv.index[-1]).strftime("%Y-%m-%d")

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
        trendlines.append({"label": label, "style": style, "points": series})

    support_zones = [
        {**_zone_dict(zone), "start_date": start_date, "end_date": end_date}
        for zone in supports
    ]
    resistance_zones = [
        {**_zone_dict(zone), "start_date": start_date, "end_date": end_date}
        for zone in resistances
    ]

    annotations: list[dict[str, Any]] = []
    highlighted: list[dict[str, Any]] = []

    for label_point in structure.structure_labels:
        annotations.append(
            {
                "type": "structure_label",
                "label": label_point["label"],
                "kind": label_point["kind"],
                "bar_index": label_point["bar_index"],
                "date": label_point["date"],
                "price": label_point["price"],
                "position": "aboveBar" if label_point["kind"] == "high" else "belowBar",
            }
        )

    for event in breakouts:
        annotations.append(
            {
                "type": "breakout",
                "event_type": event.event_type,
                "bar_index": event.bar_index,
                "date": event.date,
                "price": event.price,
                "label": event.label,
                "volume_confirmed": event.volume_confirmed,
                "position": "aboveBar",
            }
        )

    volume_markers: list[dict[str, Any]] = []
    if volume_ctx.breakout_confirmed:
        volume_markers.append(
            {
                "type": "volume_confirmed_breakout",
                "label": "Breakout confirmed by volume",
                "bar_index": len(ohlcv) - 1,
                "date": end_date,
            }
        )
    if volume_ctx.pattern_volume_confirmed:
        volume_markers.append(
            {
                "type": "pattern_volume_confirmed",
                "label": "Pattern confirmed by volume",
            }
        )
    if volume_ctx.pattern_volume_absent:
        volume_markers.append(
            {
                "type": "pattern_volume_absent",
                "label": volume_ctx.pattern_volume_note
                or "Pattern lacks volume confirmation",
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
        "volume_markers": volume_markers,
    }


def _structure_dict(structure: TrendStructure) -> dict[str, Any]:
    return {
        "bias": structure.bias,
        "summary": structure.summary,
        "higher_highs": structure.higher_highs,
        "higher_lows": structure.higher_lows,
        "lower_highs": structure.lower_highs,
        "lower_lows": structure.lower_lows,
        "trend_break": structure.trend_break,
        "exhaustion": structure.exhaustion,
        "acceleration": structure.acceleration,
        "structure_labels": list(structure.structure_labels),
    }


def _ma_dict(ma_ctx: MovingAverageContext) -> dict[str, Any]:
    return {
        "sma20": ma_ctx.sma_20,
        "sma50": ma_ctx.sma_50,
        "sma200": ma_ctx.sma_200,
        "above_sma20": ma_ctx.above_sma_20,
        "above_sma50": ma_ctx.above_sma_50,
        "above_sma200": ma_ctx.above_sma_200,
        "dist_sma20_pct": ma_ctx.dist_sma_20_pct,
        "dist_sma50_pct": ma_ctx.dist_sma_50_pct,
        "dist_sma200_pct": ma_ctx.dist_sma_200_pct,
        "golden_cross": ma_ctx.golden_cross,
        "death_cross": ma_ctx.death_cross,
        "summary": ma_ctx.summary,
    }


def _volume_dict(volume_ctx: VolumeContext) -> dict[str, Any]:
    return {
        "label": volume_ctx.label,
        "summary": volume_ctx.summary,
        "vol_ratio_20d": volume_ctx.vol_ratio_20d,
        "vol_zscore_20d": volume_ctx.vol_zscore_20d,
        "spike": volume_ctx.spike,
        "accumulation": volume_ctx.accumulation,
        "distribution": volume_ctx.distribution,
        "breakout_confirmed": volume_ctx.breakout_confirmed,
        "weak_move": volume_ctx.weak_move,
        "pattern_volume_confirmed": volume_ctx.pattern_volume_confirmed,
        "pattern_volume_absent": volume_ctx.pattern_volume_absent,
        "pattern_volume_note": volume_ctx.pattern_volume_note,
    }


def _rs_dict(context: TrendContext) -> dict[str, Any]:
    rs63 = context.rs_vs_spy_63d
    summary = "Relative strength context unavailable."
    trend = "neutral"
    if rs63 is not None:
        pct = rs63 * 100
        if pct > 2:
            summary = f"RS vs SPY: {pct:+.1f}% (63d). Stock continues outperforming benchmark."
            trend = "leading"
        elif pct < -2:
            summary = f"RS vs SPY: {pct:+.1f}% (63d). Stock is lagging the benchmark."
            trend = "lagging"
        else:
            summary = f"RS vs SPY: {pct:+.1f}% (63d). Performance is roughly in line with SPY."
            trend = "inline"
    return {
        "rs_vs_spy_21d": context.rs_vs_spy_21d,
        "rs_vs_spy_63d": context.rs_vs_spy_63d,
        "rs_vs_spy_126d": context.rs_vs_spy_126d,
        "trend": trend,
        "summary": summary,
    }


def _sr_summary(supports: list[PriceZone], resistances: list[PriceZone], close: float) -> str:
    if supports and resistances:
        return (
            f"Support: ${supports[0].price_low:.2f}. "
            f"Resistance: ${resistances[0].price_high:.2f}. "
            f"Current price sits between major levels."
        )
    if supports:
        return f"Support: ${supports[0].price_low:.2f}."
    if resistances:
        return f"Resistance: ${resistances[0].price_high:.2f}."
    return "No dominant support/resistance zones identified."


def _zone_dict(zone: PriceZone) -> dict[str, Any]:
    return {
        "price_low": zone.price_low,
        "price_high": zone.price_high,
        "label": zone.label,
        "zone_type": zone.zone_type,
        "touches": zone.touches,
        "strength": zone.strength,
        "strength_score": zone.strength_score,
        "is_major": zone.is_major,
    }


def _breakout_dict(event: BreakoutEvent) -> dict[str, Any]:
    return {
        "event_type": event.event_type,
        "bar_index": event.bar_index,
        "date": event.date,
        "price": event.price,
        "label": event.label,
        "volume_confirmed": event.volume_confirmed,
        "zone_label": event.zone_label,
    }


def _score_row(name: str, bias: ScoreBias, detail: str) -> dict[str, Any]:
    return {"name": name, "bias": bias, "detail": detail}


def _bias_from_structure(structure: TrendStructure) -> ScoreBias:
    if structure.bias == "uptrend" and not structure.trend_break:
        return "bullish"
    if structure.bias == "downtrend" and not structure.trend_break:
        return "bearish"
    if structure.trend_break:
        return "neutral"
    return "neutral"


def _bias_from_zones(
    close: float,
    supports: list[PriceZone],
    resistances: list[PriceZone],
) -> ScoreBias:
    if supports and close <= supports[0].price_high * 1.01:
        return "bullish"
    if resistances and close >= resistances[0].price_low * 0.99:
        return "bearish"
    return "neutral"


def _bias_from_rs(context: TrendContext, is_benchmark: bool) -> ScoreBias:
    if is_benchmark:
        return "neutral"
    rs = context.rs_vs_spy_63d
    if rs is None:
        return "neutral"
    if rs > 0.02:
        return "bullish"
    if rs < -0.02:
        return "bearish"
    return "neutral"


def _bias_from_volume(volume_ctx: VolumeContext) -> ScoreBias:
    if volume_ctx.breakout_confirmed or volume_ctx.accumulation:
        return "bullish"
    if volume_ctx.distribution or volume_ctx.weak_move:
        return "bearish"
    return "neutral"


def _bias_from_pattern(
    pattern: CandlestickPatternHit | None,
    structure: TrendStructure,
) -> ScoreBias:
    if pattern is None:
        return "neutral"
    if pattern.direction == "bullish" and structure.bias != "downtrend":
        return "bullish"
    if pattern.direction == "bearish" and structure.bias != "uptrend":
        return "bearish"
    return "neutral"


def _bias_from_model(
    model_prediction: int | None,
    ranking_score: float | None,
    is_benchmark: bool,
) -> ScoreBias:
    if is_benchmark:
        return "neutral"
    prob = ranking_score
    if prob is None and model_prediction is not None:
        prob = 0.62 if model_prediction == 1 else 0.38
    if prob is None:
        return "neutral"
    if prob >= 0.58:
        return "bullish"
    if prob <= 0.42:
        return "bearish"
    return "neutral"


def _model_summary(model_prediction: int | None, ranking_score: float | None) -> str:
    if ranking_score is not None:
        return f"Model C ranking probability {ranking_score:.0%}."
    if model_prediction == 1:
        return "Model C leans bullish for the next 5 sessions."
    if model_prediction == 0 or model_prediction == -1:
        return "Model C leans cautious for the next 5 sessions."
    return "Model C unavailable."


def _bias_from_ma_regime(ma_ctx: MovingAverageContext) -> ScoreBias:
    if ma_ctx.above_sma_200 and ma_ctx.above_sma_50:
        return "bullish"
    if ma_ctx.above_sma_200 is False and ma_ctx.above_sma_50 is False:
        return "bearish"
    return "neutral"


def _pattern_context_bias(
    pattern: CandlestickPatternHit | None,
    structure: TrendStructure,
) -> ScoreBias:
    if pattern is None:
        return "neutral"
    if pattern.direction == "bearish" and structure.bias == "uptrend":
        return "neutral"
    if pattern.direction == "bullish" and structure.bias == "downtrend":
        return "neutral"
    return _bias_from_pattern(pattern, structure)


def _pattern_context_detail(
    pattern: CandlestickPatternHit | None,
    structure: TrendStructure,
    pattern_bias: ScoreBias,
) -> str:
    if pattern is None:
        return "No active pattern."
    if pattern.direction == "bearish" and structure.bias == "uptrend":
        return f"{pattern.label} — counter-trend context only."
    if pattern.direction == "bullish" and structure.bias == "downtrend":
        return f"{pattern.label} — counter-trend until structure repairs."
    return f"{pattern.label} ({pattern_bias})."


def _component_from_structure(structure: TrendStructure) -> float:
    if structure.bias == "uptrend" and not structure.trend_break:
        return 82 if structure.acceleration else 75
    if structure.bias == "downtrend" and not structure.trend_break:
        return 18
    if structure.trend_break:
        return 45
    return 50


def _component_from_rs(context: TrendContext, is_benchmark: bool) -> float:
    if is_benchmark:
        return 50
    rs = context.rs_vs_spy_63d
    if rs is None:
        return 50
    return float(np.clip(50 + rs * 500, 10, 90))


def _component_from_zones(
    close: float,
    supports: list[PriceZone],
    resistances: list[PriceZone],
) -> float:
    score = 50.0
    if supports and close <= supports[0].price_high * 1.02:
        score += 12
    if resistances and close >= resistances[0].price_low * 0.98:
        score -= 12
    if supports:
        score += min(8, supports[0].strength_score * 0.08)
    return float(np.clip(score, 0, 100))


def _component_from_volume(volume_ctx: VolumeContext) -> float:
    if volume_ctx.breakout_confirmed or volume_ctx.accumulation:
        return 78
    if volume_ctx.pattern_volume_confirmed:
        return 70
    if volume_ctx.pattern_volume_absent or volume_ctx.weak_move:
        return 35
    if volume_ctx.distribution:
        return 30
    return 50


def _component_from_model(
    model_prediction: int | None,
    ranking_score: float | None,
    is_benchmark: bool,
) -> float:
    if is_benchmark:
        return 50
    prob = ranking_score
    if prob is None and model_prediction is not None:
        prob = 0.62 if model_prediction == 1 else 0.38
    if prob is None:
        return 50
    return float(np.clip(prob * 100, 0, 100))


def _score_label(score: int) -> str:
    if score >= 70:
        return "Strong chart quality"
    if score >= 55:
        return "Constructive"
    if score >= 45:
        return "Mixed"
    return "Weak / caution"


def _composite_thesis(rows: list[dict[str, Any]], is_benchmark: bool) -> dict[str, str]:
    weights = {
        "Market regime": 3,
        "Trend structure": 4,
        "Relative strength": 3,
        "Support / resistance": 2,
        "Volume": 1,
        "Pattern": 1,
        "Model C": 2 if not is_benchmark else 0,
    }
    score = 0.0
    total = 0.0
    for row in rows:
        w = weights.get(row["name"], 1)
        total += w
        if row["bias"] == "bullish":
            score += w
        elif row["bias"] == "bearish":
            score -= w

    ratio = score / total if total else 0.0
    if ratio >= 0.35:
        return {
            "headline": "Primary thesis: bullish",
            "action": "hold_or_add",
            "detail": "Structure, trend, and confirmation layers lean bullish.",
        }
    if ratio <= -0.35:
        return {
            "headline": "Primary thesis: bearish",
            "action": "reduce_or_avoid",
            "detail": "Structure and confirmation layers lean bearish.",
        }
    return {
        "headline": "Primary thesis: neutral / mixed",
        "action": "hold_with_caution",
        "detail": "Signals conflict — size risk conservatively until structure resolves.",
    }
