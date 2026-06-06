from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.trading_bias_models import (
    AlignmentState,
    CatalystAlignment,
    TradingBiasAction,
    TradingBiasAlignment,
    TradingBiasLevels,
    TradingBiasResponse,
    VolumeAlignment,
)
from ranking_pipeline.regime.constants import (
    REGIME_HIGH_VOL_CHOP,
    REGIME_RISK_OFF,
    REGIME_RISK_ON_CHOP,
    REGIME_RISK_ON_TREND,
)

WEIGHT_PATTERN_TREND = 0.30
WEIGHT_MARKET_REGIME = 0.20
WEIGHT_RELATIVE_STRENGTH = 0.20
WEIGHT_VOLUME = 0.15
WEIGHT_LEVELS = 0.10
WEIGHT_CATALYST = 0.05


@dataclass(frozen=True)
class RankingContext:
    ml_probability: float | None = None
    expected_excess_return: float | None = None
    final_score: float | None = None
    rank: int | None = None
    universe_count: int | None = None


@dataclass(frozen=True)
class TradingBiasInputs:
    symbol: str
    pattern_intelligence: dict[str, Any] | None = None
    prediction_payload: dict[str, Any] | None = None
    regime_id: str | None = None
    ranking: RankingContext | None = None
    events: list[Any] = field(default_factory=list)
    data_gaps: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Component:
    score: float
    alignment: str
    bullish: tuple[str, ...] = ()
    bearish: tuple[str, ...] = ()


def evaluate_trading_bias(inputs: TradingBiasInputs) -> TradingBiasResponse:
    symbol = inputs.symbol.strip().upper()
    data_gaps = _dedupe(inputs.data_gaps)

    if inputs.pattern_intelligence is None:
        if "Pattern analysis unavailable" not in data_gaps:
            data_gaps.append("Pattern analysis unavailable")
        return TradingBiasResponse(
            symbol=symbol,
            bias="Neutral",
            confidence="Low",
            action="Watch",
            bullish_factors=[],
            bearish_factors=[],
            invalidation=None,
            levels=TradingBiasLevels(),
            alignment=TradingBiasAlignment(
                market_regime=_market_regime_alignment(inputs.regime_id).alignment,
                relative_strength="mixed",
                pattern_trend="mixed",
                volume="neutral",
                catalyst=_catalyst_component(inputs.events).alignment,  # type: ignore[arg-type]
            ),
            data_gaps=data_gaps,
        )

    pattern = _pattern_component(inputs.pattern_intelligence)
    market = _market_regime_alignment(inputs.regime_id)
    rs = _relative_strength_component(
        inputs.pattern_intelligence,
        inputs.prediction_payload,
        inputs.ranking,
    )
    volume = _volume_component(inputs.pattern_intelligence)
    levels_component = _levels_component(inputs.pattern_intelligence)
    catalyst = _catalyst_component(inputs.events)

    weighted_score = (
        pattern.score * WEIGHT_PATTERN_TREND
        + market.score * WEIGHT_MARKET_REGIME
        + rs.score * WEIGHT_RELATIVE_STRENGTH
        + volume.score * WEIGHT_VOLUME
        + levels_component.score * WEIGHT_LEVELS
        + catalyst.score * WEIGHT_CATALYST
    )

    bias = _bias_from_score(weighted_score)
    confidence = _confidence_from_score(
        score=weighted_score,
        bias=bias,
        pattern=pattern,
        market=market,
        rs=rs,
        data_gaps=data_gaps,
    )
    action = _action_for_bias(
        bias=bias,
        confidence=confidence,
        market=market,
        levels_component=levels_component,
    )
    levels = _extract_levels(inputs.pattern_intelligence, bias=bias)
    invalidation = _invalidation_text(bias=bias, levels=levels)

    bullish = _top_factors(
        pattern.bullish + levels_component.bullish + volume.bullish
        + market.bullish + rs.bullish + catalyst.bullish
    )
    bearish = _top_factors(
        pattern.bearish + levels_component.bearish + volume.bearish
        + market.bearish + rs.bearish + catalyst.bearish
    )

    return TradingBiasResponse(
        symbol=symbol,
        bias=bias,
        confidence=confidence,
        action=action,
        bullish_factors=bullish,
        bearish_factors=bearish,
        invalidation=invalidation,
        levels=levels,
        alignment=TradingBiasAlignment(
            market_regime=market.alignment,  # type: ignore[arg-type]
            relative_strength=rs.alignment,  # type: ignore[arg-type]
            pattern_trend=pattern.alignment,  # type: ignore[arg-type]
            volume=volume.alignment,  # type: ignore[arg-type]
            catalyst=catalyst.alignment,  # type: ignore[arg-type]
        ),
        data_gaps=data_gaps,
    )


def _pattern_component(payload: dict[str, Any]) -> _Component:
    trend = _get(payload, "trend_context", "trendContext") or {}
    scores = _get(payload, "scores") or {}
    chart = _get(payload, "chart_intelligence", "chartIntelligence") or {}
    summary = _get(chart, "summary") or {}
    outlook = _get(summary, "outlook") or {}
    primary = _get(payload, "primary_pattern", "primaryPattern") or {}

    trend_bias = (_get(trend, "trend_bias", "trendBias") or "").lower()
    outlook_label = str(_get(outlook, "label") or "").lower()
    trend_strength = _float(_get(scores, "trend_strength", "trendStrength"))
    score = 0.0
    bullish: list[str] = []
    bearish: list[str] = []

    if trend_bias == "uptrend":
        score += 0.55
        bullish.append("Price structure is in an uptrend")
    elif trend_bias == "downtrend":
        score -= 0.55
        bearish.append("Price structure is in a downtrend")

    if "bullish" in outlook_label:
        score += 0.35
        bullish.append("Chart intelligence leans bullish")
    elif "bearish" in outlook_label:
        score -= 0.35
        bearish.append("Chart intelligence leans bearish")
    elif outlook_label == "neutral":
        bullish.append("Chart outlook is neutral rather than deteriorating")

    direction = str(_get(primary, "direction") or "").lower()
    label = _get(primary, "label") or "Active pattern"
    if direction == "bullish":
        score += 0.15
        bullish.append(f"{label} pattern supports buyers")
    elif direction == "bearish":
        score -= 0.15
        bearish.append(f"{label} pattern warns of seller control")

    if trend_strength is not None:
        if trend_strength >= 0.68:
            score += 0.15
            bullish.append("Trend score is constructive")
        elif trend_strength <= 0.38:
            score -= 0.15
            bearish.append("Trend score is weak")

    return _component_from_score(score, bullish=bullish, bearish=bearish)


def _market_regime_alignment(regime_id: str | None) -> _Component:
    rid = (regime_id or "").lower()
    if rid == REGIME_RISK_ON_TREND:
        return _Component(
            score=0.8,
            alignment="aligned",
            bullish=("Market regime is risk-on trend",),
        )
    if rid == REGIME_RISK_ON_CHOP or rid == "":
        return _Component(
            score=0.0,
            alignment="mixed",
            bullish=("Market regime is not blocking risk",),
        )
    if rid in {REGIME_RISK_OFF, REGIME_HIGH_VOL_CHOP}:
        return _Component(
            score=-0.8,
            alignment="against",
            bearish=(f"Market regime is unfavorable ({rid})",),
        )
    return _Component(score=0.0, alignment="mixed")


def _relative_strength_component(
    payload: dict[str, Any],
    prediction_payload: dict[str, Any] | None,
    ranking: RankingContext | None,
) -> _Component:
    trend = _get(payload, "trend_context", "trendContext") or {}
    rs63 = _float(_get(trend, "rs_vs_spy_63d", "rsVsSpy63d"))
    rs21 = _float(_get(trend, "rs_vs_spy_21d", "rsVsSpy21d"))
    rs = rs63 if rs63 is not None else rs21

    ml_probability = (
        _float((prediction_payload or {}).get("up_prob"))
        or _float((prediction_payload or {}).get("ranking_score"))
        or (ranking.ml_probability if ranking else None)
    )
    expected_excess = (
        _float((prediction_payload or {}).get("expected_excess_return"))
        or (ranking.expected_excess_return if ranking else None)
    )
    rank = ranking.rank if ranking else None
    universe_count = ranking.universe_count if ranking else None

    score = 0.0
    bullish: list[str] = []
    bearish: list[str] = []

    if rs is not None:
        if rs >= 0.03:
            score += 0.45
            bullish.append("Relative strength versus SPY is positive")
        elif rs <= -0.03:
            score -= 0.45
            bearish.append("Relative strength versus SPY is negative")

    if ml_probability is not None:
        if ml_probability >= 0.62:
            score += 0.35
            bullish.append("Model C favors SPY outperformance")
        elif ml_probability <= 0.42:
            score -= 0.35
            bearish.append("Model C favors SPY underperformance")

    if expected_excess is not None:
        if expected_excess >= 0.01:
            score += 0.15
            bullish.append("Expected excess return is positive")
        elif expected_excess <= -0.01:
            score -= 0.15
            bearish.append("Expected excess return is negative")

    if rank is not None and universe_count:
        percentile = 1.0 - ((max(1, rank) - 1) / max(1, universe_count))
        if percentile >= 0.80:
            score += 0.15
            bullish.append("Ranking percentile is strong")
        elif percentile <= 0.35:
            score -= 0.15
            bearish.append("Ranking percentile is weak")

    return _component_from_score(score, bullish=bullish, bearish=bearish)


def _volume_component(payload: dict[str, Any]) -> _Component:
    trend = _get(payload, "trend_context", "trendContext") or {}
    scores = _get(payload, "scores") or {}
    chart = _get(payload, "chart_intelligence", "chartIntelligence") or {}
    summary = _get(chart, "summary") or {}
    bullets = _get(summary, "why_this_outlook", "whyThisOutlook") or []

    vol_ratio = _float(_get(trend, "vol_ratio_20d", "volRatio20d"))
    volume_score = _float(_get(scores, "volume_confirmation", "volumeConfirmation"))
    text = " ".join(
        str(_get(item, "text") or "") for item in bullets if isinstance(item, dict)
    ).lower()

    score = 0.0
    bullish: list[str] = []
    bearish: list[str] = []
    if vol_ratio is not None:
        if vol_ratio >= 1.35:
            score += 0.45
            bullish.append(f"Volume is elevated at {vol_ratio:.1f}x the 20-day average")
        elif vol_ratio <= 0.75:
            score -= 0.25
            bearish.append("Recent move lacks volume confirmation")

    if volume_score is not None:
        if volume_score >= 0.70:
            score += 0.35
            bullish.append("Volume confirmation score is constructive")
        elif volume_score <= 0.40:
            score -= 0.35
            bearish.append("Volume confirmation score is weak")

    if "distribution" in text or "weak volume" in text:
        score -= 0.6
        bearish.append("Chart summary flags weak volume or distribution")
    if "breakout volume" in text or "accumulation" in text:
        score += 0.35
        bullish.append("Chart summary flags accumulation or breakout volume")

    if score >= 0.3:
        alignment: VolumeAlignment = "confirmed"
    elif score <= -0.3:
        alignment = "warning"
    else:
        alignment = "neutral"
    return _Component(
        score=max(-1.0, min(1.0, score)),
        alignment=alignment,
        bullish=tuple(_dedupe(bullish)),
        bearish=tuple(_dedupe(bearish)),
    )


def _levels_component(payload: dict[str, Any]) -> _Component:
    chart = _get(payload, "chart_intelligence", "chartIntelligence") or {}
    events = _get(chart, "breakout_events", "breakoutEvents") or []
    score = 0.0
    bullish: list[str] = []
    bearish: list[str] = []

    for event in events[-3:]:
        if not isinstance(event, dict):
            continue
        kind = str(_get(event, "kind", "breakout_kind", "breakoutKind") or "").lower()
        if kind == "confirmed_breakout":
            score += 0.6
            bullish.append("Confirmed breakout above resistance")
        elif kind == "failed_breakdown":
            score += 0.35
            bullish.append("Failed breakdown suggests buyers defended support")
        elif kind == "failed_breakout":
            score -= 0.6
            bearish.append("Failed breakout warns of overhead supply")
        elif kind == "confirmed_breakdown":
            score -= 0.6
            bearish.append("Confirmed breakdown below support")

    return _component_from_score(score, bullish=bullish, bearish=bearish)


def _catalyst_component(events: list[Any]) -> _Component:
    if not events:
        return _Component(score=0.0, alignment="none")

    score = 0.0
    bullish: list[str] = []
    bearish: list[str] = []
    for event in events[:5]:
        title = str(_get_event_value(event, "title") or "").lower()
        detail = str(_get_event_value(event, "detail") or "").lower()
        text = f"{title} {detail}"
        if any(token in text for token in ("beat", "upgrade", "raises", "positive")):
            score += 0.25
            bullish.append("Recent events include a positive catalyst")
        if any(token in text for token in ("miss", "downgrade", "sec investigation", "negative")):
            score -= 0.25
            bearish.append("Recent events include a negative catalyst")

    if score > 0:
        alignment: CatalystAlignment = "positive"
    elif score < 0:
        alignment = "negative"
    else:
        alignment = "neutral"
    return _Component(
        score=max(-1.0, min(1.0, score)),
        alignment=alignment,
        bullish=tuple(_dedupe(bullish)),
        bearish=tuple(_dedupe(bearish)),
    )


def _extract_levels(payload: dict[str, Any], *, bias: str) -> TradingBiasLevels:
    chart = _get(payload, "chart_intelligence", "chartIntelligence") or {}
    selected = _get(chart, "selected_levels", "selectedLevels") or {}
    supports = _get(chart, "support_zones", "supportZones") or []
    resistances = _get(chart, "resistance_zones", "resistanceZones") or []
    support = _selected_zone_price(selected, "nearest_support", "nearestSupport", kind="support")
    if support is None:
        support = _zone_price(supports, "support")
    resistance = _selected_zone_price(
        selected,
        "nearest_resistance",
        "nearestResistance",
        kind="resistance",
    )
    if resistance is None:
        resistance = _zone_price(resistances, "resistance")
    actionable_support = _selected_zone_price(
        selected,
        "actionable_support",
        "actionableSupport",
        kind="support",
    )
    actionable_resistance = _selected_zone_price(
        selected,
        "actionable_resistance",
        "actionableResistance",
        kind="resistance",
    )
    breakout = resistance
    stop_invalid = actionable_support if bias != "Bearish" else actionable_resistance
    return TradingBiasLevels(
        support=support,
        resistance=resistance,
        breakout_level=breakout,
        stop_invalid_level=stop_invalid,
    )


def _invalidation_text(*, bias: str, levels: TradingBiasLevels) -> str | None:
    if bias == "Bullish" and levels.stop_invalid_level is not None:
        return f"Bias weakens on a close below ${levels.stop_invalid_level:.2f}."
    if bias == "Bearish" and levels.stop_invalid_level is not None:
        return f"Bias weakens on a close above ${levels.stop_invalid_level:.2f}."
    if bias == "Neutral":
        if levels.resistance is not None and levels.support is not None:
            return (
                f"Neutral range resolves above ${levels.resistance:.2f} "
                f"or below ${levels.support:.2f}."
            )
        if levels.resistance is not None:
            return f"Neutral range resolves above ${levels.resistance:.2f}."
        if levels.support is not None:
            return f"Neutral range resolves below ${levels.support:.2f}."
    return None


def _confidence_from_score(
    *,
    score: float,
    bias: str,
    pattern: _Component,
    market: _Component,
    rs: _Component,
    data_gaps: list[str],
) -> str:
    if bias == "Neutral" or data_gaps:
        return "Low" if abs(score) < 0.35 or data_gaps else "Medium"
    if market.alignment == "against" and bias == "Bullish":
        return "Medium"
    agreement = sum(
        1
        for component in (pattern, market, rs)
        if component.alignment == "aligned"
    )
    if abs(score) >= 0.55 and agreement >= 2:
        return "High"
    if abs(score) >= 0.30:
        return "Medium"
    return "Low"


def _action_for_bias(
    *,
    bias: str,
    confidence: str,
    market: _Component,
    levels_component: _Component,
) -> TradingBiasAction:
    if market.alignment == "against":
        return "Risk-off"
    if bias == "Bearish":
        return "Avoid"
    if bias == "Bullish":
        if levels_component.score >= 0.4:
            return "Confirm breakout"
        if confidence in {"High", "Medium"}:
            return "Pullback setup"
    return "Watch"


def _bias_from_score(score: float) -> str:
    if score >= 0.22:
        return "Bullish"
    if score <= -0.22:
        return "Bearish"
    return "Neutral"


def _component_from_score(
    score: float,
    *,
    bullish: list[str],
    bearish: list[str],
) -> _Component:
    score = max(-1.0, min(1.0, score))
    if score >= 0.25:
        alignment: AlignmentState = "aligned"
    elif score <= -0.25:
        alignment = "against"
    else:
        alignment = "mixed"
    return _Component(
        score=score,
        alignment=alignment,
        bullish=tuple(_dedupe(bullish)),
        bearish=tuple(_dedupe(bearish)),
    )


def _zone_price(zones: list[Any], kind: str) -> float | None:
    if not zones:
        return None
    zone = zones[0]
    if not isinstance(zone, dict):
        return None
    if kind == "support":
        value = _float(_get(zone, "display_level", "displayLevel"))
        if value is None:
            value = _float(_get(zone, "price_high", "priceHigh"))
    else:
        value = _float(_get(zone, "breakout_level", "breakoutLevel"))
        if value is None:
            value = _float(_get(zone, "display_level", "displayLevel"))
        if value is None:
            value = _float(_get(zone, "price_low", "priceLow"))
    if value is None:
        value = _float(_get(zone, "price", "level"))
    return round(value, 2) if value is not None else None


def _selected_zone_price(
    selected: dict[str, Any],
    snake_key: str,
    camel_key: str,
    *,
    kind: str,
) -> float | None:
    zone = _get(selected, snake_key, camel_key)
    if not isinstance(zone, dict):
        return None
    return _zone_price([zone], kind)


def _top_factors(items: tuple[str, ...]) -> list[str]:
    return _dedupe([item for item in items if item])[:3]


def _dedupe(items: list[str] | tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _get(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _get_event_value(event: Any, key: str) -> Any:
    if isinstance(event, dict):
        return event.get(key)
    return getattr(event, key, None)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
