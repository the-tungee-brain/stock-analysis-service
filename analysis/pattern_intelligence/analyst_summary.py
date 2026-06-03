"""Concise analyst summary for Chart Intelligence (outlook-first, no redundancy)."""

from __future__ import annotations

from typing import Any, Literal

from analysis.pattern_intelligence.benchmarks import BENCHMARK_NOTICE, is_model_benchmark_symbol
from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit
from analysis.pattern_intelligence.chart_analysis import (
    PriceZone,
    TrendStructure,
    VolumeContext,
)
from analysis.pattern_intelligence.scoring import PatternScoreBreakdown
from analysis.pattern_intelligence.trend_context import TrendContext

ThesisSide = Literal["bullish", "bearish", "neutral"]
BulletTone = Literal["support", "caution"]

# Max distance from current price for an actionable key level (fraction of spot).
KEY_LEVEL_MAX_DISTANCE_PCT = 0.15
EVIDENCE_MAX_BULLETS = 5


def build_analyst_summary(
    *,
    symbol: str,
    pattern: CandlestickPatternHit | None,
    structure: TrendStructure,
    supports: list[PriceZone],
    resistances: list[PriceZone],
    context: TrendContext,
    volume_ctx: VolumeContext,
    scores: PatternScoreBreakdown,
    model_prediction: int | None,
    ranking_score: float | None,
) -> dict[str, Any]:
    is_benchmark = is_model_benchmark_symbol(symbol)
    thesis_side = _composite_thesis_side(
        structure=structure,
        context=context,
        volume_ctx=volume_ctx,
        pattern=pattern,
        is_benchmark=is_benchmark,
    )
    outlook_label, tone = _qualitative_outlook_label(
        structure=structure,
        thesis_side=thesis_side,
        volume_ctx=volume_ctx,
        pattern=pattern,
        is_benchmark=is_benchmark,
    )

    key_level = _key_level_block(
        thesis_side=thesis_side,
        supports=supports,
        resistances=resistances,
        close=context.close,
    )

    return {
        "outlook": _outlook_block(
            label=outlook_label,
            tone=tone,
            thesis_side=thesis_side,
            structure=structure,
            pattern=pattern,
            volume_ctx=volume_ctx,
            is_benchmark=is_benchmark,
        ),
        "key_level": key_level,
        "why_this_outlook": _evidence_bullets(
            pattern=pattern,
            structure=structure,
            context=context,
            volume_ctx=volume_ctx,
            scores=scores,
            thesis_side=thesis_side,
            is_benchmark=is_benchmark,
        ),
        "thesis": _thesis_narrative(
            pattern=pattern,
            structure=structure,
            volume_ctx=volume_ctx,
            thesis_side=thesis_side,
            outlook_label=outlook_label,
            key_level=key_level,
            is_benchmark=is_benchmark,
        ),
        "disclaimer": (
            "Chart intelligence is qualitative technical context, not investment advice. "
            "See Trend Analysis for model scores and probabilities."
        ),
    }


def _composite_thesis_side(
    *,
    structure: TrendStructure,
    context: TrendContext,
    volume_ctx: VolumeContext,
    pattern: CandlestickPatternHit | None,
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


def _qualitative_outlook_label(
    *,
    structure: TrendStructure,
    thesis_side: ThesisSide,
    volume_ctx: VolumeContext,
    pattern: CandlestickPatternHit | None,
    is_benchmark: bool,
) -> tuple[str, str]:
    """Outlook label from price structure only — not Model C probabilities."""
    if is_benchmark:
        return "Benchmark", "unavailable"

    if thesis_side == "bullish":
        if structure.exhaustion:
            return "Slight Bullish", "slight_bullish"
        if structure.acceleration or (
            volume_ctx.breakout_confirmed and structure.bias == "uptrend"
        ):
            return "Bullish", "bullish"
        return "Slight Bullish", "slight_bullish"

    if thesis_side == "bearish":
        if structure.bias == "downtrend" and not structure.trend_break:
            return "Bearish", "bearish"
        return "Slight Bearish", "slight_bearish"

    if structure.trend_break:
        return "Neutral", "neutral"
    if pattern is not None and pattern.direction == "bullish":
        return "Slight Bullish", "slight_bullish"
    if pattern is not None and pattern.direction == "bearish":
        return "Slight Bearish", "slight_bearish"
    return "Neutral", "neutral"


def _outlook_block(
    *,
    label: str,
    tone: str,
    thesis_side: ThesisSide,
    structure: TrendStructure,
    pattern: CandlestickPatternHit | None,
    volume_ctx: VolumeContext,
    is_benchmark: bool,
) -> dict[str, Any]:
    expectation = _expectation_sentence(
        label=label,
        thesis_side=thesis_side,
        structure=structure,
        pattern=pattern,
        volume_ctx=volume_ctx,
        is_benchmark=is_benchmark,
    )

    return {
        "label": label,
        "tone": tone,
        "probability": None,
        "probability_display": None,
        "expectation": expectation,
        "model_context": None,
        "is_benchmark": is_benchmark,
        "benchmark_notice": BENCHMARK_NOTICE if is_benchmark else None,
    }


def _expectation_sentence(
    *,
    label: str,
    thesis_side: ThesisSide,
    structure: TrendStructure,
    pattern: CandlestickPatternHit | None,
    volume_ctx: VolumeContext,
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
        if structure.bias == "uptrend" and not structure.trend_break:
            return (
                "Expect modest upside over the next 5 sessions while the uptrend remains intact."
            )
        if pattern is not None and pattern.direction == "bullish":
            return (
                "Expect a constructive move higher over the next 5 sessions if buyers follow through."
            )
        return "Expect a mildly positive bias over the next 5 sessions."

    if "bearish" in label_lower:
        if structure.bias == "uptrend":
            return (
                "Expect near-term softness or consolidation over the next 5 sessions."
            )
        if structure.bias == "downtrend":
            return "Expect continued pressure over the next 5 sessions."
        return "Expect downside risk or choppy trade over the next 5 sessions."

    if thesis_side == "bullish":
        return "Expect range-bound to slightly higher trade over the next 5 sessions."
    if thesis_side == "bearish":
        return "Expect range-bound to slightly lower trade over the next 5 sessions."
    return "Expect sideways trade until price commits to a clear direction."


def _nearby_resistances(
    close: float,
    zones: list[PriceZone],
    *,
    max_distance_pct: float = KEY_LEVEL_MAX_DISTANCE_PCT,
) -> list[PriceZone]:
    nearby: list[PriceZone] = []
    for zone in zones:
        if zone.price_low <= close * 1.002:
            continue
        distance_pct = (zone.price_low - close) / max(close, 1e-9)
        if distance_pct <= max_distance_pct:
            nearby.append(zone)
    nearby.sort(key=lambda z: z.price_low - close)
    return nearby


def _nearby_supports(
    close: float,
    zones: list[PriceZone],
    *,
    max_distance_pct: float = KEY_LEVEL_MAX_DISTANCE_PCT,
) -> list[PriceZone]:
    nearby: list[PriceZone] = []
    for zone in zones:
        if zone.price_high >= close * 0.998:
            continue
        distance_pct = (close - zone.price_high) / max(close, 1e-9)
        if distance_pct <= max_distance_pct:
            nearby.append(zone)
    nearby.sort(key=lambda z: close - z.price_high)
    return nearby


def _unavailable_key_level(close: float) -> dict[str, Any]:
    return {
        "label": "No nearby level",
        "price": None,
        "level_type": None,
        "display": "No actionable level near current price",
        "implication": (
            f"No support or resistance sits within a reasonable distance of ${close:,.2f} — "
            "watch trend and Model C for the next move."
        ),
        "available": False,
    }


def _key_level_from_zone(
    *,
    zone: PriceZone,
    level_type: Literal["support", "resistance"],
    watch_price: float,
    implication: str,
) -> dict[str, Any]:
    return {
        "label": f"${watch_price:.2f} {level_type.title()}",
        "price": round(watch_price, 2),
        "level_type": level_type,
        "display": f"Key Level: ${watch_price:.2f} {level_type.title()}",
        "implication": implication,
        "available": True,
    }


def _key_level_block(
    *,
    thesis_side: ThesisSide,
    supports: list[PriceZone],
    resistances: list[PriceZone],
    close: float,
) -> dict[str, Any]:
    nearby_res = _nearby_resistances(close, resistances)
    nearby_sup = _nearby_supports(close, supports)

    if thesis_side == "bullish":
        if nearby_res:
            zone = nearby_res[0]
            watch = zone.price_low
            return _key_level_from_zone(
                zone=zone,
                level_type="resistance",
                watch_price=watch,
                implication=(
                    f"Watch ${watch:,.2f} resistance above current price — a breakout would "
                    "likely extend the bullish move; repeated rejection may signal consolidation."
                ),
            )
        return _unavailable_key_level(close)

    if thesis_side == "bearish":
        if nearby_sup:
            zone = nearby_sup[0]
            watch = zone.price_high
            return _key_level_from_zone(
                zone=zone,
                level_type="support",
                watch_price=watch,
                implication=(
                    f"Watch ${watch:,.2f} support below current price — a break would "
                    "likely accelerate downside; holding the level keeps the bear case contained."
                ),
            )
        return _unavailable_key_level(close)

    candidates: list[tuple[float, PriceZone, Literal["support", "resistance"], float]] = []
    for zone in nearby_res:
        watch = zone.price_low
        dist = watch - close
        candidates.append((dist, zone, "resistance", watch))
    for zone in nearby_sup:
        watch = zone.price_high
        dist = close - watch
        candidates.append((dist, zone, "support", watch))

    if not candidates:
        return _unavailable_key_level(close)

    _, zone, level_type, watch = min(candidates, key=lambda item: item[0])
    if level_type == "resistance":
        implication = (
            f"Nearest overhead level at ${watch:,.2f} — clearing it would favor upside; "
            "failure there keeps the range intact."
        )
    else:
        implication = (
            f"Nearest support at ${watch:,.2f} — losing it would open downside; "
            "holding it preserves the neutral-to-firm bias."
        )

    return _key_level_from_zone(
        zone=zone,
        level_type=level_type,
        watch_price=watch,
        implication=implication,
    )


def _evidence_bullets(
    *,
    pattern: CandlestickPatternHit | None,
    structure: TrendStructure,
    context: TrendContext,
    volume_ctx: VolumeContext,
    scores: PatternScoreBreakdown,
    thesis_side: ThesisSide,
    is_benchmark: bool,
) -> list[dict[str, Any]]:
    """At most one bullet per factor; avoid overlap with key level and outlook."""
    slots: dict[str, tuple[str, BulletTone]] = {}

    if structure.bias == "uptrend" and not structure.trend_break:
        slots["trend"] = ("Strong trend", "support")
    elif structure.bias == "downtrend" and not structure.trend_break:
        slots["trend"] = ("Weak trend structure", "caution")
    elif structure.trend_break:
        slots["trend"] = ("Trend recently broke down", "caution")
    else:
        slots["trend"] = ("Choppy, range-bound structure", "caution")

    if not is_benchmark:
        rs = context.rs_vs_spy_63d
        if rs is not None and rs > 0.02:
            slots["rs"] = ("Market leadership", "support")
        elif rs is not None and rs < -0.02:
            slots["rs"] = ("Losing market leadership", "caution")
        elif scores.relative_strength >= 0.55:
            slots["rs"] = ("Holding leadership vs the market", "support")
        elif scores.relative_strength <= 0.4:
            slots["rs"] = ("Leadership fading vs the market", "caution")

    if volume_ctx.accumulation or volume_ctx.breakout_confirmed:
        slots["volume"] = ("Volume confirms demand", "support")
    elif volume_ctx.distribution:
        slots["volume"] = ("Volume signals supply pressure", "caution")
    elif volume_ctx.weak_move:
        slots["volume"] = ("Rally lacks volume support", "caution")

    if pattern is not None and pattern.direction != "neutral":
        slots["pattern"] = _pattern_evidence_line(pattern, structure)

    if structure.exhaustion:
        slots["exhaustion"] = ("Momentum is tiring", "caution")
    elif structure.acceleration and structure.bias == "uptrend" and thesis_side == "bullish":
        slots["momentum"] = ("Momentum improving", "support")

    order = ["trend", "rs", "volume", "pattern", "momentum", "exhaustion"]
    bullets: list[dict[str, Any]] = []
    for key in order:
        if key not in slots:
            continue
        text, tone = slots[key]
        bullets.append({"text": text, "tone": tone})
        if len(bullets) >= EVIDENCE_MAX_BULLETS:
            break

    return bullets


def _thesis_narrative(
    *,
    pattern: CandlestickPatternHit | None,
    structure: TrendStructure,
    volume_ctx: VolumeContext,
    thesis_side: ThesisSide,
    outlook_label: str,
    key_level: dict[str, Any],
    is_benchmark: bool,
) -> str:
    if is_benchmark:
        if structure.bias == "uptrend":
            likely = "The benchmark will likely grind higher or hold firm over the next several sessions."
        elif structure.bias == "downtrend":
            likely = "The benchmark will likely stay under pressure over the next several sessions."
        else:
            likely = "The benchmark will likely chop sideways until structure resolves."
        strengthen = "Sustained strength in breadth and trend would confirm the bullish regime."
        weaken = "A sharp risk-off leg would weaken the constructive read."
        if pattern is not None and pattern.direction == "bearish":
            weaken = (
                f"A follow-through {pattern.label.lower()} would add tactical caution "
                "without necessarily reversing the primary trend."
            )
        return f"{likely} {strengthen} {weaken}"

    lead = _thesis_lead_phrase(thesis_side, volume_ctx, outlook_label)
    strengthen, weaken = _scenario_branches(key_level, thesis_side, pattern, structure)

    return f"{lead} {strengthen} {weaken}"


def _pattern_evidence_line(
    pattern: CandlestickPatternHit,
    structure: TrendStructure,
) -> tuple[str, BulletTone]:
    pattern_id = pattern.pattern_id.lower()
    if pattern.direction == "bullish":
        if pattern_id in {"bullish_engulfing", "hammer", "morning_star", "piercing_line"}:
            return "Buyers regained control after pullback", "support"
        if pattern_id == "three_white_soldiers":
            return "Sustained buying across recent sessions", "support"
        if structure.bias == "downtrend":
            return "Early sign of buyer interest against the trend", "support"
        return "Buyers showing renewed interest", "support"

    if pattern_id in {"bearish_engulfing", "shooting_star", "evening_star"}:
        return "Sellers pressed after a recent rally", "caution"
    if pattern_id == "three_black_crows":
        return "Persistent selling pressure", "caution"
    if structure.bias == "uptrend":
        return "Pullback pattern — watch for follow-through", "caution"
    return "Selling pressure emerging", "caution"


def _thesis_lead_phrase(
    thesis_side: ThesisSide,
    volume_ctx: VolumeContext,
    outlook_label: str,
) -> str:
    """Forward-looking setup line — does not repeat the Outlook expectation sentence."""
    label_lower = outlook_label.lower()
    if thesis_side == "bullish":
        if volume_ctx.accumulation or volume_ctx.breakout_confirmed:
            return "Buyers remain in control, though the next leg still needs follow-through."
        if "bullish" in label_lower:
            return "Momentum still favors the bulls, but the move may be gradual."
        return "The setup still leans higher if buyers hold the bid."

    if thesis_side == "bearish":
        if volume_ctx.distribution:
            return "Sellers remain in control until demand re-emerges."
        return "Downside pressure is more likely than a sustained rebound."

    return "The range is still in play until one side takes control."


def _scenario_branches(
    key_level: dict[str, Any],
    thesis_side: ThesisSide,
    pattern: CandlestickPatternHit | None,
    structure: TrendStructure,
) -> tuple[str, str]:
    price = key_level.get("price")
    level_type = key_level.get("level_type")
    available = key_level.get("available", price is not None)

    if available and price is not None and level_type == "resistance":
        strengthen = (
            f"A breakout above ${price:,.2f} resistance would strengthen the bullish continuation case."
        )
        weaken = (
            f"Repeated rejection below ${price:,.2f} would weaken the outlook and favor consolidation."
        )
        return strengthen, weaken

    if available and price is not None and level_type == "support":
        strengthen = (
            f"Holding ${price:,.2f} support would keep the bearish thesis contained or invite a bounce."
        )
        weaken = (
            f"A decisive break below ${price:,.2f} would strengthen the downside scenario."
        )
        return strengthen, weaken

    if pattern is not None and pattern.direction == "bearish" and structure.bias == "uptrend":
        strengthen = "Clearing recent highs would restore the bullish path."
        weaken = (
            f"Follow-through on the {pattern.label.lower()} would weaken the near-term bullish case."
        )
        return strengthen, weaken

    if thesis_side == "bullish":
        return (
            "A move through recent highs would strengthen the bullish case.",
            "Loss of momentum would weaken it.",
        )
    if thesis_side == "bearish":
        return (
            "A failed bounce would reinforce the bearish case.",
            "A sustained push above recent highs would weaken it.",
        )
    return (
        "A decisive breakout would set the next directional leg.",
        "A failed breakout would keep the range intact.",
    )
