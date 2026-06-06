from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from app.models.intraday_trading_bias_models import (
    IntradayTradingBiasAlignment,
    IntradayTradingBiasLevels,
    IntradayTradingBiasResponse,
)

EASTERN = ZoneInfo("America/New_York")
MARKET_OPEN = time(9, 30)
OPENING_RANGE_END = time(10, 0)
MARKET_CLOSE = time(16, 0)
STALE_WARNING_SECONDS = 15 * 60
STALE_INACTIVE_SECONDS = 60 * 60


@dataclass(frozen=True)
class IntradayBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    session: str


@dataclass(frozen=True)
class IntradayTradingBiasInputs:
    symbol: str
    bars: list[IntradayBar] = field(default_factory=list)
    market_bars: list[IntradayBar] = field(default_factory=list)
    support: float | None = None
    resistance: float | None = None
    catalyst: str = "none"
    data_gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    now: datetime | None = None


@dataclass(frozen=True)
class _TrendRead:
    score: float
    alignment: str
    reason: str | None = None


def evaluate_intraday_trading_bias(
    inputs: IntradayTradingBiasInputs,
) -> IntradayTradingBiasResponse:
    data_gaps = _dedupe(inputs.data_gaps)
    warnings = _dedupe(
        ["Delayed/polled yfinance bars; not real-time."]
        + list(inputs.warnings)
    )

    bars = sorted(inputs.bars, key=lambda bar: bar.timestamp)
    if not bars:
        return _neutral_response(
            data_gaps=_with_gap(data_gaps, "Intraday 5m bars unavailable"),
            warnings=warnings,
        )

    regular = [bar for bar in bars if bar.session == "regular"]
    if not regular:
        return _neutral_response(
            data_gaps=_with_gap(data_gaps, "Regular-session intraday bars unavailable"),
            warnings=warnings,
            last_updated=_latest_timestamp(bars),
            staleness_seconds=_staleness_seconds(bars, inputs.now),
        )

    last_bar = regular[-1]
    last_updated = _latest_timestamp(bars)
    staleness = _staleness_seconds(bars, inputs.now)
    if staleness is not None and staleness > STALE_WARNING_SECONDS:
        warnings = _with_gap(
            warnings,
            f"Latest yfinance bar is stale by {round(staleness / 60)} minutes.",
        )

    if _is_inactive_intraday_read(staleness=staleness, now=inputs.now):
        reason = "Intraday read is inactive because market is closed or bars are stale."
        return _neutral_response(
            data_gaps=_with_gap(data_gaps, "Intraday read is stale or outside market hours"),
            warnings=_with_gap(warnings, reason),
            last_updated=last_updated,
            staleness_seconds=staleness,
        )

    premarket = [bar for bar in bars if bar.session == "premarket"]
    opening_range = [
        bar
        for bar in regular
        if _local_time(bar.timestamp) < OPENING_RANGE_END
    ]
    opening_range_complete = _local_time(last_bar.timestamp) >= OPENING_RANGE_END
    if not opening_range_complete:
        warnings = _with_gap(warnings, "Opening range is not complete yet.")

    premarket_high = _max_high(premarket)
    premarket_low = _min_low(premarket)
    open_range_high = _max_high(opening_range) if opening_range_complete else None
    open_range_low = _min_low(opening_range) if opening_range_complete else None
    vwap = _vwap(regular)
    high_of_day = _max_high(regular)
    low_of_day = _min_low(regular)
    current = last_bar.close

    trend = _trend_read(regular)
    market = _market_read(inputs.market_bars)
    volume_state, volume_score, volume_reason = _volume_read(regular)
    vwap_state, vwap_score, vwap_reason = _vwap_read(regular, vwap)
    setup_type, setup_score, setup_reason = _setup_read(
        regular=regular,
        premarket_high=premarket_high,
        premarket_low=premarket_low,
        open_range_high=open_range_high,
        open_range_low=open_range_low,
        vwap=vwap,
    )
    catalyst_score = _catalyst_score(inputs.catalyst)

    if not inputs.market_bars:
        data_gaps = _with_gap(data_gaps, "SPY/QQQ intraday market bars unavailable")
    if premarket_high is None or premarket_low is None:
        data_gaps = _with_gap(data_gaps, "Premarket bars unavailable")
    if vwap is None:
        data_gaps = _with_gap(data_gaps, "VWAP unavailable")

    score = (
        setup_score * 0.30
        + vwap_score * 0.20
        + volume_score * 0.20
        + market.score * 0.15
        + trend.score * 0.10
        + catalyst_score * 0.05
    )
    bias = _bias_from_score(score)
    confidence = _confidence_from_score(score, data_gaps=data_gaps, warnings=warnings)
    if market.alignment == "against" and bias == "Bullish" and confidence == "High":
        confidence = "Medium"

    invalidation = _invalidation_for_bias(
        bias=bias,
        vwap=vwap,
        open_range_low=open_range_low,
        open_range_high=open_range_high,
        support=inputs.support,
        resistance=inputs.resistance,
        low_of_day=low_of_day,
        high_of_day=high_of_day,
    )
    action = _action_for_bias(
        bias=bias,
        confidence=confidence,
        setup_type=setup_type,
        market_alignment=market.alignment,
    )
    levels = IntradayTradingBiasLevels(
        premarket_high=_round(premarket_high),
        premarket_low=_round(premarket_low),
        open_range_high=_round(open_range_high),
        open_range_low=_round(open_range_low),
        vwap=_round(vwap),
        support=_round(inputs.support),
        resistance=_round(inputs.resistance),
        invalidation=_round(invalidation),
    )
    reasons = _top_reasons(
        [
            setup_reason,
            vwap_reason,
            trend.reason,
            volume_reason,
            market.reason,
            _level_reason(current, inputs.support, inputs.resistance),
            _catalyst_reason(inputs.catalyst),
        ]
    )

    return IntradayTradingBiasResponse(
        bias=bias,
        confidence=confidence,
        setup_type=setup_type,
        action=action,
        levels=levels,
        alignment=IntradayTradingBiasAlignment(
            market=market.alignment,  # type: ignore[arg-type]
            intraday_trend=trend.alignment,  # type: ignore[arg-type]
            vwap=vwap_state,  # type: ignore[arg-type]
            volume=volume_state,  # type: ignore[arg-type]
            catalyst=inputs.catalyst,  # type: ignore[arg-type]
        ),
        reasons=reasons,
        warnings=warnings,
        data_gaps=data_gaps,
        last_updated=last_updated,
        staleness_seconds=staleness,
    )


def _neutral_response(
    *,
    data_gaps: list[str],
    warnings: list[str],
    last_updated: datetime | None = None,
    staleness_seconds: int | None = None,
) -> IntradayTradingBiasResponse:
    return IntradayTradingBiasResponse(
        bias="Neutral",
        confidence="Low",
        setup_type="None",
        action="Watch",
        levels=IntradayTradingBiasLevels(),
        alignment=IntradayTradingBiasAlignment(
            market="mixed",
            intraday_trend="mixed",
            vwap="below",
            volume="neutral",
            catalyst="none",
        ),
        reasons=[],
        warnings=warnings,
        data_gaps=data_gaps,
        last_updated=last_updated,
        staleness_seconds=staleness_seconds,
    )


def classify_session(timestamp: datetime) -> str:
    local = timestamp.astimezone(EASTERN)
    local_time = local.time()
    if local_time < MARKET_OPEN:
        return "premarket"
    if local_time < MARKET_CLOSE:
        return "regular"
    return "afterhours"


def _setup_read(
    *,
    regular: list[IntradayBar],
    premarket_high: float | None,
    premarket_low: float | None,
    open_range_high: float | None,
    open_range_low: float | None,
    vwap: float | None,
) -> tuple[str, float, str | None]:
    if not regular:
        return "None", 0.0, None

    current = regular[-1].close
    previous = regular[-2].close if len(regular) >= 2 else regular[-1].open
    trend = _trend_read(regular)

    if open_range_high is not None and open_range_low is not None:
        if current > open_range_high and trend.score > 0.1:
            return (
                "OpeningRangeBreakout",
                0.75,
                "Price is holding above the completed opening range high.",
            )
        if current < open_range_low and trend.score < -0.1:
            return (
                "FailedBreakout",
                -0.75,
                "Price is breaking below the completed opening range low.",
            )

    if premarket_high is not None and current > premarket_high:
        return "GapAndGo", 0.7, "Price is holding above premarket high."

    if premarket_low is not None and current < premarket_low:
        return "GapFade", -0.65, "Price has faded below premarket support."

    if vwap is not None and len(regular) >= 2:
        if previous <= vwap < current:
            return "VWAPReclaim", 0.6, "Price reclaimed VWAP on the latest move."
        if previous >= vwap > current:
            return "FailedBreakout", -0.55, "Price rejected VWAP on the latest move."

    if abs(trend.score) >= 0.55:
        direction = "higher" if trend.score > 0 else "lower"
        score = 0.45 if trend.score > 0 else -0.45
        return "TrendDay", score, f"5-minute structure is trending {direction}."

    return "RangeDay", 0.0, "Price is trading like a range day so far."


def _vwap_read(
    regular: list[IntradayBar],
    vwap: float | None,
) -> tuple[str, float, str | None]:
    if not regular or vwap is None:
        return "below", 0.0, None
    current = regular[-1].close
    previous = regular[-2].close if len(regular) >= 2 else regular[-1].open
    if previous <= vwap < current:
        return "reclaiming", 0.65, "Price is reclaiming VWAP."
    if previous >= vwap > current:
        return "rejecting", -0.65, "Price is rejecting VWAP."
    if current > vwap:
        return "above", 0.45, "Price is above VWAP."
    return "below", -0.45, "Price is below VWAP."


def _trend_read(bars: list[IntradayBar]) -> _TrendRead:
    if len(bars) < 4:
        return _TrendRead(0.0, "mixed")
    recent = bars[-6:] if len(bars) >= 6 else bars
    first = recent[0].close
    last = recent[-1].close
    if first <= 0:
        return _TrendRead(0.0, "mixed")
    change = (last - first) / first
    higher_highs = sum(
        1 for prev, cur in zip(recent, recent[1:]) if cur.high > prev.high
    )
    lower_lows = sum(1 for prev, cur in zip(recent, recent[1:]) if cur.low < prev.low)

    score = 0.0
    if change >= 0.006:
        score += 0.45
    elif change <= -0.006:
        score -= 0.45
    if higher_highs >= max(2, len(recent) // 2):
        score += 0.25
    if lower_lows >= max(2, len(recent) // 2):
        score -= 0.25

    if score >= 0.35:
        return _TrendRead(score, "aligned", "Recent 5-minute bars show buyer control.")
    if score <= -0.35:
        return _TrendRead(score, "against", "Recent 5-minute bars show seller control.")
    return _TrendRead(score, "mixed", "Recent 5-minute trend is mixed.")


def _market_read(market_bars: list[IntradayBar]) -> _TrendRead:
    regular = [bar for bar in market_bars if bar.session == "regular"]
    if not regular:
        return _TrendRead(0.0, "mixed")
    trend = _trend_read(regular)
    if trend.alignment == "aligned":
        return _TrendRead(0.65, "aligned", "Broad market intraday trend supports risk.")
    if trend.alignment == "against":
        return _TrendRead(-0.65, "against", "Broad market intraday trend is a headwind.")
    return _TrendRead(0.0, "mixed", "Broad market intraday trend is mixed.")


def _volume_read(regular: list[IntradayBar]) -> tuple[str, float, str | None]:
    if len(regular) < 6:
        return "neutral", 0.0, "Not enough 5-minute bars for volume confirmation."
    latest_window = regular[-3:]
    prior_window = regular[:-3]
    prior_avg = sum(bar.volume for bar in prior_window) / max(len(prior_window), 1)
    latest_avg = sum(bar.volume for bar in latest_window) / len(latest_window)
    if prior_avg <= 0:
        return "neutral", 0.0, None
    ratio = latest_avg / prior_avg
    if ratio >= 1.25:
        return "confirmed", 0.55, "Recent volume is expanding versus earlier bars."
    if ratio <= 0.75:
        return "weak", -0.35, "Recent volume is fading versus earlier bars."
    return "neutral", 0.0, "Recent volume is near same-day average."


def _vwap(bars: list[IntradayBar]) -> float | None:
    volume_total = sum(max(bar.volume, 0) for bar in bars)
    if volume_total <= 0:
        return None
    dollar_volume = sum(
        ((bar.high + bar.low + bar.close) / 3) * max(bar.volume, 0)
        for bar in bars
    )
    return dollar_volume / volume_total


def _bias_from_score(score: float) -> str:
    if score >= 0.25:
        return "Bullish"
    if score <= -0.25:
        return "Bearish"
    return "Neutral"


def _confidence_from_score(
    score: float,
    *,
    data_gaps: list[str],
    warnings: list[str],
) -> str:
    absolute = abs(score)
    if data_gaps or any("stale" in warning.lower() for warning in warnings):
        if absolute >= 0.55:
            return "Medium"
        return "Low"
    if absolute >= 0.55:
        return "High"
    if absolute >= 0.30:
        return "Medium"
    return "Low"


def _action_for_bias(
    *,
    bias: str,
    confidence: str,
    setup_type: str,
    market_alignment: str,
) -> str:
    if market_alignment == "against" and bias == "Bullish":
        return "RiskOff"
    if bias == "Bearish":
        return "Avoid"
    if bias == "Bullish" and confidence in {"High", "Medium"}:
        if setup_type in {"OpeningRangeBreakout", "GapAndGo", "VWAPReclaim"}:
            return "ConfirmBreakout"
        return "WaitForPullback"
    return "Watch"


def _invalidation_for_bias(
    *,
    bias: str,
    vwap: float | None,
    open_range_low: float | None,
    open_range_high: float | None,
    support: float | None,
    resistance: float | None,
    low_of_day: float | None,
    high_of_day: float | None,
) -> float | None:
    if bias == "Bullish":
        return _first_not_none(open_range_low, vwap, low_of_day)
    if bias == "Bearish":
        return _first_not_none(open_range_high, vwap, high_of_day)
    return vwap


def _level_reason(
    current: float,
    support: float | None,
    resistance: float | None,
) -> str | None:
    if support is not None and current > support:
        return "Price remains above higher-timeframe support."
    if resistance is not None and current < resistance:
        return "Price remains below higher-timeframe resistance."
    return None


def _catalyst_score(catalyst: str) -> float:
    if catalyst == "positive":
        return 0.35
    if catalyst == "negative":
        return -0.35
    return 0.0


def _catalyst_reason(catalyst: str) -> str | None:
    if catalyst == "positive":
        return "Recent catalyst tone is positive."
    if catalyst == "negative":
        return "Recent catalyst tone is negative."
    return None


def _top_reasons(reasons: list[str | None], *, limit: int = 5) -> list[str]:
    result: list[str] = []
    for reason in reasons:
        if reason and reason not in result:
            result.append(reason)
        if len(result) >= limit:
            break
    return result


def _latest_timestamp(bars: list[IntradayBar]) -> datetime | None:
    if not bars:
        return None
    latest = max(bar.timestamp for bar in bars)
    if latest.tzinfo is None:
        return latest.replace(tzinfo=timezone.utc)
    return latest.astimezone(timezone.utc)


def _staleness_seconds(
    bars: list[IntradayBar],
    now: datetime | None,
) -> int | None:
    latest = _latest_timestamp(bars)
    if latest is None:
        return None
    now_utc = now or datetime.now(timezone.utc)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    return max(0, int((now_utc.astimezone(timezone.utc) - latest).total_seconds()))


def _local_time(timestamp: datetime) -> time:
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(EASTERN).time()


def _is_inactive_intraday_read(
    *,
    staleness: int | None,
    now: datetime | None,
) -> bool:
    if staleness is not None and staleness > STALE_INACTIVE_SECONDS:
        return True
    now_value = now or datetime.now(timezone.utc)
    if now_value.tzinfo is None:
        now_value = now_value.replace(tzinfo=timezone.utc)
    local_now = now_value.astimezone(EASTERN)
    if local_now.weekday() >= 5:
        return True
    return not (MARKET_OPEN <= local_now.time() < MARKET_CLOSE)


def _max_high(bars: list[IntradayBar]) -> float | None:
    return max((bar.high for bar in bars), default=None)


def _min_low(bars: list[IntradayBar]) -> float | None:
    return min((bar.low for bar in bars), default=None)


def _first_not_none(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _with_gap(values: list[str], value: str) -> list[str]:
    if value not in values:
        return values + [value]
    return values
