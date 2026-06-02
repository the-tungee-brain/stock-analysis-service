"""Concise analyst summary for Chart Intelligence (outlook-first, no redundancy)."""

from __future__ import annotations

from typing import Any, Literal

from analysis.pattern_intelligence.benchmarks import BENCHMARK_NOTICE, is_model_benchmark_symbol
from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit
from analysis.pattern_intelligence.chart_analysis import (
    MovingAverageContext,
    PriceZone,
    TrendStructure,
    VolumeContext,
)
from analysis.pattern_intelligence.scoring import PatternScoreBreakdown
from analysis.pattern_intelligence.trend_context import TrendContext

ThesisSide = Literal["bullish", "bearish", "neutral"]
BulletTone = Literal["support", "caution"]


def build_analyst_summary(
    *,
    symbol: str,
    pattern: CandlestickPatternHit | None,
    structure: TrendStructure,
    supports: list[PriceZone],
    resistances: list[PriceZone],
    context: TrendContext,
    volume_ctx: VolumeContext,
    ma_ctx: MovingAverageContext,
    scores: PatternScoreBreakdown,
    model_prediction: int | None,
    ranking_score: float | None,
) -> dict[str, Any]:
    is_benchmark = is_model_benchmark_symbol(symbol)
    probability = None if is_benchmark else _resolve_probability(ranking_score, model_prediction)
    outlook_label, tone = _conviction_tier(probability, is_benchmark)
    thesis_side = _composite_thesis_side(
        structure=structure,
        context=context,
        volume_ctx=volume_ctx,
        pattern=pattern,
        probability=probability,
        model_prediction=model_prediction,
        is_benchmark=is_benchmark,
    )

    return {
        "outlook": _outlook_block(
            label=outlook_label,
            tone=tone,
            probability=probability,
            thesis_side=thesis_side,
            structure=structure,
            pattern=pattern,
            is_benchmark=is_benchmark,
        ),
        "key_level": _key_level_block(
            thesis_side=thesis_side,
            supports=supports,
            resistances=resistances,
            close=context.close,
        ),
        "why_this_outlook": _evidence_bullets(
            symbol=symbol,
            pattern=pattern,
            structure=structure,
            context=context,
            volume_ctx=volume_ctx,
            ma_ctx=ma_ctx,
            scores=scores,
            supports=supports,
            resistances=resistances,
            thesis_side=thesis_side,
            is_benchmark=is_benchmark,
        ),
        "thesis": _thesis_narrative(
            symbol=symbol,
            pattern=pattern,
            structure=structure,
            context=context,
            volume_ctx=volume_ctx,
            thesis_side=thesis_side,
            outlook_label=outlook_label,
            supports=supports,
            resistances=resistances,
            is_benchmark=is_benchmark,
        ),
        "disclaimer": (
            "Chart intelligence is contextual technical analysis, not investment advice. "
            "Model C remains the primary 5-day signal; patterns and structure provide context."
        ),
    }


def _resolve_probability(
    ranking_score: float | None,
    model_prediction: int | None,
) -> float | None:
    if ranking_score is not None:
        return ranking_score
    if model_prediction is None:
        return None
    return 0.62 if model_prediction == 1 else 0.38


def _conviction_tier(
    probability: float | None,
    is_benchmark: bool,
) -> tuple[str, str]:
    if is_benchmark:
        return "Benchmark", "unavailable"
    if probability is None:
        return "Unavailable", "unavailable"
    if probability >= 0.65:
        return "Strong Bullish", "strong_bullish"
    if probability >= 0.57:
        return "Bullish", "bullish"
    if probability >= 0.52:
        return "Slight Bullish", "slight_bullish"
    if probability >= 0.48:
        return "Neutral", "neutral"
    if probability >= 0.43:
        return "Slight Bearish", "slight_bearish"
    if probability >= 0.35:
        return "Bearish", "bearish"
    return "Strong Bearish", "strong_bearish"


def _composite_thesis_side(
    *,
    structure: TrendStructure,
    context: TrendContext,
    volume_ctx: VolumeContext,
    pattern: CandlestickPatternHit | None,
    probability: float | None,
    model_prediction: int | None,
    is_benchmark: bool,
) -> ThesisSide:
    score = 0.0
    if structure.bias == "uptrend" and not structure.trend_break:
        score += 1.5
    elif structure.bias == "downtrend" and not structure.trend_break:
        score -= 1.5
    elif structure.trend_break:
        score *= 0.5

    rs = context.rs_vs_spy_63d
    if not is_benchmark and rs is not None:
        if rs > 0.02:
            score += 0.8
        elif rs < -0.02:
            score -= 0.8

    if volume_ctx.breakout_confirmed or volume_ctx.accumulation:
        score += 0.4
    elif volume_ctx.distribution or volume_ctx.weak_move:
        score -= 0.4

    if not is_benchmark:
        prob = probability
        if prob is None and model_prediction is not None:
            prob = 0.62 if model_prediction == 1 else 0.38
        if prob is not None:
            if prob >= 0.55:
                score += 0.9
            elif prob <= 0.45:
                score -= 0.9

    if pattern is not None:
        if pattern.direction == "bullish" and structure.bias != "downtrend":
            score += 0.3
        elif pattern.direction == "bearish" and structure.bias != "uptrend":
            score -= 0.3

    if score >= 0.6:
        return "bullish"
    if score <= -0.6:
        return "bearish"
    return "neutral"


def _outlook_block(
    *,
    label: str,
    tone: str,
    probability: float | None,
    thesis_side: ThesisSide,
    structure: TrendStructure,
    pattern: CandlestickPatternHit | None,
    is_benchmark: bool,
) -> dict[str, Any]:
    expectation = _expectation_sentence(
        label=label,
        thesis_side=thesis_side,
        structure=structure,
        pattern=pattern,
        is_benchmark=is_benchmark,
    )
    model_line = None
    if not is_benchmark and probability is not None:
        edge = _model_edge_phrase(label, probability)
        model_line = f"Model C suggests {edge} over the next 5 sessions."

    return {
        "label": label,
        "tone": tone,
        "probability": probability,
        "probability_display": (
            f"{int(round(probability * 100))}%" if probability is not None else None
        ),
        "expectation": expectation,
        "model_context": model_line,
        "is_benchmark": is_benchmark,
        "benchmark_notice": BENCHMARK_NOTICE if is_benchmark else None,
    }


def _expectation_sentence(
    *,
    label: str,
    thesis_side: ThesisSide,
    structure: TrendStructure,
    pattern: CandlestickPatternHit | None,
    is_benchmark: bool,
) -> str:
    if is_benchmark:
        if structure.bias == "uptrend":
            return (
                "Expect the benchmark to hold its uptrend over the next several sessions, "
                "with pattern context flagging tactical shifts only."
            )
        if structure.bias == "downtrend":
            return (
                "Expect continued benchmark weakness over the next several sessions; "
                "defensive positioning may be warranted."
            )
        return (
            "Expect mixed benchmark trading over the next several sessions until "
            "structure resolves more clearly."
        )

    label_lower = label.lower()
    if "bullish" in label_lower:
        if structure.bias == "uptrend" and structure.trend_break:
            return (
                "Expect modest upside or sideways trading over the next 5 sessions. "
                "The broader uptrend remains intact, but nearby resistance may limit gains."
            )
        if structure.bias == "uptrend":
            return (
                "Expect modest upside over the next 5 sessions as the uptrend stays intact, "
                "though nearby resistance could slow progress."
            )
        if pattern is not None and pattern.direction == "bullish":
            return (
                "Expect a constructive near-term bounce over the next 5 sessions, "
                "though the longer-term trend still needs confirmation."
            )
        return (
            "Expect a mildly positive bias over the next 5 sessions, with upside "
            "dependent on follow-through above nearby levels."
        )

    if "bearish" in label_lower:
        if structure.bias == "downtrend":
            return (
                "Expect continued pressure over the next 5 sessions as the downtrend "
                "remains in control; rallies may struggle near overhead resistance."
            )
        if structure.bias == "uptrend":
            return (
                "Expect near-term softness or consolidation over the next 5 sessions "
                "inside an otherwise intact longer-term uptrend."
            )
        return (
            "Expect downside risk or choppy trade over the next 5 sessions until "
            "price stabilizes above support."
        )

    if thesis_side == "bullish":
        return (
            "Expect range-bound to slightly higher trade over the next 5 sessions; "
            "confirmation above resistance would improve the outlook."
        )
    if thesis_side == "bearish":
        return (
            "Expect range-bound to slightly lower trade over the next 5 sessions; "
            "a break below support would increase downside risk."
        )
    return (
        "Expect sideways trading over the next 5 sessions until price commits "
        "to a clear break above resistance or below support."
    )


def _model_edge_phrase(label: str, probability: float) -> str:
    label_lower = label.lower()
    if "strong bullish" in label_lower:
        return "a strong bullish edge"
    if "slight bullish" in label_lower:
        return "a modest bullish edge"
    if "bullish" in label_lower:
        return "a bullish edge"
    if "strong bearish" in label_lower:
        return "a strong bearish edge"
    if "slight bearish" in label_lower:
        return "a modest bearish edge"
    if "bearish" in label_lower:
        return "a bearish edge"
    if probability >= 0.52:
        return "a slight bullish edge"
    if probability <= 0.48:
        return "a slight bearish edge"
    return "no clear directional edge"


def _key_level_block(
    *,
    thesis_side: ThesisSide,
    supports: list[PriceZone],
    resistances: list[PriceZone],
    close: float,
) -> dict[str, Any]:
    if thesis_side == "bullish" and resistances:
        zone = resistances[0]
        price = zone.price_high
        level_type = "resistance"
        implication = (
            "If price breaks above this level, bullish momentum is likely to strengthen. "
            "Failure near resistance may lead to consolidation."
        )
    elif thesis_side == "bearish" and supports:
        zone = supports[0]
        price = zone.price_low
        level_type = "support"
        implication = (
            "If price breaks below this level, bearish pressure is likely to accelerate. "
            "Holding above support would keep the downside thesis in check."
        )
    elif resistances and supports:
        dist_res = abs(resistances[0].price_low - close)
        dist_sup = abs(close - supports[0].price_high)
        if dist_res <= dist_sup:
            zone = resistances[0]
            price = zone.price_high
            level_type = "resistance"
        else:
            zone = supports[0]
            price = zone.price_low
            level_type = "support"
        implication = (
            f"This {level_type} is the nearest major level — a decisive break would set "
            "the next directional move for the 5-day outlook."
        )
    elif resistances:
        zone = resistances[0]
        price = zone.price_high
        level_type = "resistance"
        implication = (
            "A sustained move above this resistance would shift the outlook more constructive; "
            "rejection here favors consolidation."
        )
    elif supports:
        zone = supports[0]
        price = zone.price_low
        level_type = "support"
        implication = (
            "Holding this support keeps the current thesis intact; "
            "a break below would increase near-term downside risk."
        )
    else:
        return {
            "label": "Key level unavailable",
            "price": None,
            "level_type": None,
            "display": "No dominant support or resistance identified.",
            "implication": (
                "Without a clear nearby level, focus on trend direction and Model C "
                "for the 5-day read."
            ),
        }

    return {
        "label": f"${price:.2f} {level_type.title()}",
        "price": round(price, 2),
        "level_type": level_type,
        "display": f"Key Level: ${price:.2f} {level_type.title()}",
        "implication": implication,
    }


def _evidence_bullets(
    *,
    symbol: str,
    pattern: CandlestickPatternHit | None,
    structure: TrendStructure,
    context: TrendContext,
    volume_ctx: VolumeContext,
    ma_ctx: MovingAverageContext,
    scores: PatternScoreBreakdown,
    supports: list[PriceZone],
    resistances: list[PriceZone],
    thesis_side: ThesisSide,
    is_benchmark: bool,
) -> list[dict[str, Any]]:
    bullets: list[tuple[str, BulletTone]] = []

    if structure.bias == "uptrend" and not structure.trend_break:
        bullets.append(("Long-term uptrend intact", "support"))
    elif structure.bias == "downtrend" and not structure.trend_break:
        bullets.append(("Downtrend structure remains in control", "support"))
    elif structure.trend_break:
        bullets.append(("Recent trend break — structure is in transition", "caution"))
    else:
        bullets.append(("Mixed market structure — no dominant trend", "caution"))

    if not is_benchmark:
        rs = context.rs_vs_spy_63d
        if rs is not None and rs > 0.02:
            bullets.append(("Outperforming the broader market", "support"))
        elif rs is not None and rs < -0.02:
            bullets.append(("Lagging the broader market", "caution"))
        elif scores.relative_strength >= 0.55:
            bullets.append(("Relative strength remains supportive", "support"))

    if volume_ctx.accumulation or volume_ctx.breakout_confirmed:
        bullets.append(("Volume supports accumulation", "support"))
    elif volume_ctx.distribution:
        bullets.append(("Volume suggests distribution", "caution"))
    elif volume_ctx.weak_move:
        bullets.append(("Recent move lacks volume confirmation", "caution"))

    if pattern is not None and pattern.direction != "neutral":
        if pattern.direction == "bullish":
            bullets.append((f"{pattern.label} confirms buyer interest", "support"))
        else:
            bullets.append((f"{pattern.label} signals near-term selling pressure", "caution"))

    if ma_ctx.above_sma_200 is True and ma_ctx.dist_sma_200_pct is not None:
        if ma_ctx.dist_sma_200_pct > 0.15:
            bullets.append(("Price remains far above its long-term trend", "support"))
        elif ma_ctx.above_sma_200:
            bullets.append(("Price holds above its long-term trend", "support"))
    elif ma_ctx.above_sma_200 is False:
        bullets.append(("Price trades below its long-term trend", "caution"))

    if thesis_side == "bullish" and resistances:
        bullets.append(("Resistance overhead", "caution"))
    elif thesis_side == "bearish" and supports:
        bullets.append(("Support nearby — watch for a breakdown", "caution"))
    elif thesis_side == "neutral" and supports and resistances:
        bullets.append(("Trading between support and resistance", "caution"))

    if structure.exhaustion:
        bullets.append(("Signs of trend exhaustion", "caution"))
    if structure.acceleration and structure.bias == "uptrend":
        bullets.append(("Uptrend showing acceleration", "support"))

    deduped: list[tuple[str, BulletTone]] = []
    seen: set[str] = set()
    for text, tone in bullets:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((text, tone))

    return [
        {"text": text, "tone": tone}
        for text, tone in deduped[:6]
    ]


def _thesis_narrative(
    *,
    symbol: str,
    pattern: CandlestickPatternHit | None,
    structure: TrendStructure,
    context: TrendContext,
    volume_ctx: VolumeContext,
    thesis_side: ThesisSide,
    outlook_label: str,
    supports: list[PriceZone],
    resistances: list[PriceZone],
    is_benchmark: bool,
) -> str:
    sym = symbol.upper()
    sentences: list[str] = []

    if is_benchmark:
        if structure.bias == "uptrend":
            sentences.append(
                f"{sym} remains in an uptrend as the market benchmark."
            )
        elif structure.bias == "downtrend":
            sentences.append(
                f"{sym} remains under pressure as the benchmark trends lower."
            )
        else:
            sentences.append(
                f"{sym} shows mixed benchmark structure without a clear directional edge."
            )
        if pattern is not None:
            sentences.append(
                f"A recent {pattern.label.lower()} adds tactical context but does not "
                "override the prevailing benchmark trend."
            )
        else:
            sentences.append(
                "No active candlestick pattern — lean on trend and regime for positioning context."
            )
        return " ".join(sentences[:3])

    if structure.bias == "uptrend":
        lead = f"{sym} remains in a strong uptrend"
        rs = context.rs_vs_spy_63d
        if rs is not None and rs > 0.02:
            lead += " and continues to outperform the broader market"
        sentences.append(f"{lead}.")
    elif structure.bias == "downtrend":
        sentences.append(
            f"{sym} remains in a bearish structure with lower highs and lower lows."
        )
    else:
        sentences.append(
            f"{sym} is consolidating without a dominant trend sequence."
        )

    outlook_fragment = _outlook_fragment(outlook_label, thesis_side)
    vol_fragment = ""
    if volume_ctx.accumulation or volume_ctx.breakout_confirmed:
        vol_fragment = " Recent buying pressure supports"
    elif volume_ctx.distribution:
        vol_fragment = " Distribution volume weighs on"
    elif volume_ctx.weak_move:
        vol_fragment = " Thin volume limits conviction in"

    if outlook_fragment:
        if vol_fragment:
            sentences.append(
                f"{vol_fragment.strip()} {outlook_fragment}."
            )
        else:
            sentences.append(f"{outlook_fragment.capitalize()}.")

    if thesis_side == "bullish" and resistances:
        sentences.append(
            "A breakout above nearby resistance would strengthen the continuation thesis; "
            "failure there may lead to consolidation."
        )
    elif thesis_side == "bearish" and supports:
        sentences.append(
            "A break below nearby support would confirm further downside; "
            "holding the level would keep the bear case in check."
        )
    elif pattern is not None and pattern.direction == "bearish" and structure.bias == "uptrend":
        sentences.append(
            f"The {pattern.label.lower()} flags near-term pullback risk inside the broader uptrend."
        )

    return " ".join(sentences[:3])


def _outlook_fragment(label: str, thesis_side: ThesisSide) -> str:
    label_lower = label.lower()
    if "bullish" in label_lower:
        return "a mildly bullish 5-day outlook, although nearby resistance could slow upside progress"
    if "bearish" in label_lower:
        return "a cautious 5-day outlook with downside risk if support fails"
    if thesis_side == "bullish":
        return "the setup favors modest upside if resistance gives way"
    if thesis_side == "bearish":
        return "the setup favors caution until support is reclaimed"
    return "the 5-day outlook is balanced until price commits directionally"
