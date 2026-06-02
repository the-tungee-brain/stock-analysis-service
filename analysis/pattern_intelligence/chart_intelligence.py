"""Chart Intelligence: scorecard, narrative, and visual overlay assembly."""

from __future__ import annotations

from typing import Any, Literal

from analysis.pattern_intelligence.benchmarks import is_model_benchmark_symbol
from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit
from analysis.pattern_intelligence.chart_analysis import (
    MovingAverageContext,
    PriceZone,
    TrendStructure,
    VolumeContext,
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
    volume_ctx = analyze_volume(ohlcv, structure)
    ma_ctx = analyze_moving_averages(ohlcv)

    pattern_metadata = [
        build_pattern_metadata(ohlcv, hit, structure_bias=structure.bias)
        for hit in active_patterns[:3]
    ]

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
        "structure": _structure_dict(structure),
        "moving_averages": _ma_dict(ma_ctx),
        "volume": _volume_dict(volume_ctx),
        "support_resistance_summary": _sr_summary(supports, resistances, context.close),
        "relative_strength": _rs_dict(context),
        "narrative": narrative,
        "scorecard": scorecard,
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
        _score_row("Trend", trend, structure.summary),
        _score_row("Support / resistance", sr, _sr_summary(supports, resistances, context.close)),
        _score_row("Relative strength", rs, _rs_dict(context)["summary"]),
        _score_row("Volume", volume, volume_ctx.summary),
        _score_row("Pattern", pattern_bias, pattern.label if pattern else "No active pattern"),
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
            "market_structure",
            "trend",
            "relative_strength",
            "support_resistance",
            "volume",
            "pattern",
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
    model_prediction: int | None,
    ranking_score: float | None,
) -> dict[str, Any]:
    is_benchmark = is_model_benchmark_symbol(symbol)
    parts: list[str] = []

    if structure.bias == "uptrend":
        parts.append("The broader trend remains intact with higher highs and higher lows.")
    elif structure.bias == "downtrend":
        parts.append("Price structure remains bearish with lower highs and lower lows.")
    else:
        parts.append("Market structure is mixed without a dominant trend sequence.")

    if supports and resistances:
        parts.append(
            f"Price sits between support near ${supports[0].price_low:.2f} "
            f"and resistance near ${resistances[0].price_high:.2f}."
        )
    elif supports:
        parts.append(f"Nearest support sits near ${supports[0].price_low:.2f}.")
    elif resistances:
        parts.append(f"Nearest resistance sits near ${resistances[0].price_high:.2f}.")

    if not is_benchmark and context.rs_vs_spy_63d is not None:
        pct = context.rs_vs_spy_63d * 100
        if pct > 1:
            parts.append(
                f"Relative strength vs SPY remains positive at {pct:+.1f}% (63d) — "
                "the name continues to lead the benchmark."
            )
        elif pct < -1:
            parts.append(
                f"Relative strength vs SPY is negative at {pct:+.1f}% (63d) — "
                "the stock is lagging the market."
            )

    parts.append(ma_ctx.summary)
    parts.append(volume_ctx.summary)

    if pattern is not None:
        if pattern.direction == "bearish" and structure.bias == "uptrend":
            parts.append(
                f"A {pattern.label.lower()} formed, but price remains above key moving averages "
                "while relative strength continues to lead. Near-term pullback risk exists, "
                "but the primary thesis remains bullish."
            )
        elif pattern.direction == "bullish" and structure.bias == "downtrend":
            parts.append(
                f"A {pattern.label.lower()} appeared against a still-bearish structure — "
                "treat it as counter-trend until support and trend repair confirm."
            )
        else:
            parts.append(
                f"The active {pattern.label.lower()} aligns with the prevailing "
                f"{structure.bias} structure and adds contextual confirmation."
            )
    else:
        parts.append("No dominant candlestick pattern is active — lean on structure and trend.")

    if not is_benchmark and ranking_score is not None:
        parts.append(
            f"Model C ranking probability is {ranking_score:.0%} — "
            f"{'supportive' if ranking_score >= 0.55 else 'cautious'} for the 5-day outlook."
        )

    full = " ".join(parts)
    action = scorecard["thesis"]["action"]
    return {
        "summary": full,
        "action": action,
        "headline": scorecard["thesis"]["headline"],
        "disclaimer": (
            "Chart intelligence is contextual technical analysis, not investment advice. "
            "Patterns frame risk; market structure and trend drive the thesis."
        ),
    }


def build_visual_overlays(
    *,
    ohlcv,
    structure: TrendStructure,
    supports: list[PriceZone],
    resistances: list[PriceZone],
    ma_ctx: MovingAverageContext,
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


def _composite_thesis(rows: list[dict[str, Any]], is_benchmark: bool) -> dict[str, str]:
    weights = {
        "Trend": 3,
        "Support / resistance": 2,
        "Relative strength": 2,
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
