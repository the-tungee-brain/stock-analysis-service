"""Market structure analysis for chart intelligence overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from features.indicators import compute_indicators

SwingKind = Literal["high", "low"]
ZoneTimeframe = Literal["shortTerm", "intermediate", "longTerm"]
ZoneRole = Literal["actionable", "nearbyContext", "majorHistorical"]
ZoneSource = Literal[
    "swing",
    "movingAverage",
    "priorHighLow",
    "roundNumber",
]
StructureLabel = Literal[
    "higher_high",
    "higher_low",
    "lower_high",
    "lower_low",
    "swing_high",
    "swing_low",
]


@dataclass(frozen=True)
class SwingPoint:
    bar_index: int
    date: str
    price: float
    kind: SwingKind


@dataclass(frozen=True)
class TrendStructure:
    bias: str
    summary: str
    higher_highs: int
    higher_lows: int
    lower_highs: int
    lower_lows: int
    trend_break: bool
    exhaustion: bool
    acceleration: bool
    swing_points: tuple[SwingPoint, ...]
    trendline: dict[str, Any] | None


@dataclass(frozen=True)
class PriceZone:
    price_low: float
    price_high: float
    label: str
    zone_type: Literal["support", "resistance"]
    touches: int
    strength: float
    midpoint: float | None = None
    timeframe: ZoneTimeframe = "intermediate"
    sources: tuple[ZoneSource, ...] = ("swing",)
    recency_bars: int | None = None
    distance_pct_from_current: float | None = None
    atr_distance: float | None = None
    level_role: ZoneRole = "nearbyContext"
    actionable_for: dict[str, bool] | None = None


@dataclass(frozen=True)
class _LevelCandidate:
    price: float
    zone_type: Literal["support", "resistance"]
    source: ZoneSource
    bar_index: int | None
    touches: int = 1
    strength: float = 0.45


@dataclass(frozen=True)
class VolumeContext:
    label: str
    summary: str
    vol_ratio_20d: float | None
    vol_zscore_20d: float | None
    spike: bool
    accumulation: bool
    distribution: bool
    breakout_confirmed: bool
    weak_move: bool


@dataclass(frozen=True)
class MovingAverageContext:
    sma_20: float | None
    sma_50: float | None
    sma_200: float | None
    above_sma_20: bool | None
    above_sma_50: bool | None
    above_sma_200: bool | None
    dist_sma_20_pct: float | None
    dist_sma_50_pct: float | None
    dist_sma_200_pct: float | None
    golden_cross: bool
    death_cross: bool
    summary: str
    sma_series: dict[str, list[dict[str, Any]]]


def _date_str(index: pd.DatetimeIndex, bar_index: int) -> str:
    return pd.Timestamp(index[bar_index]).strftime("%Y-%m-%d")


def find_swing_points(
    ohlcv: pd.DataFrame,
    *,
    lookback: int = 5,
    max_points: int = 12,
) -> list[SwingPoint]:
    if len(ohlcv) < lookback * 2 + 1:
        return []

    high = ohlcv["high"].to_numpy(dtype=float)
    low = ohlcv["low"].to_numpy(dtype=float)
    swings: list[SwingPoint] = []

    for i in range(lookback, len(ohlcv) - lookback):
        window_high = high[i - lookback : i + lookback + 1]
        window_low = low[i - lookback : i + lookback + 1]
        if high[i] >= window_high.max():
            swings.append(
                SwingPoint(
                    bar_index=i,
                    date=_date_str(ohlcv.index, i),
                    price=float(high[i]),
                    kind="high",
                )
            )
        if low[i] <= window_low.min():
            swings.append(
                SwingPoint(
                    bar_index=i,
                    date=_date_str(ohlcv.index, i),
                    price=float(low[i]),
                    kind="low",
                )
            )

    swings.sort(key=lambda item: item.bar_index)
    if len(swings) <= max_points:
        return swings
    return swings[-max_points:]


def analyze_trend_structure(
    ohlcv: pd.DataFrame,
    *,
    lookback: int = 5,
) -> TrendStructure:
    swings = find_swing_points(ohlcv, lookback=lookback)
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]

    hh = hl = lh = ll = 0
    for prev, curr in zip(highs, highs[1:]):
        if curr.price > prev.price:
            hh += 1
        elif curr.price < prev.price:
            lh += 1
    for prev, curr in zip(lows, lows[1:]):
        if curr.price > prev.price:
            hl += 1
        elif curr.price < prev.price:
            ll += 1

    close = float(ohlcv["close"].iloc[-1])
    bias = "mixed"
    if hh >= 1 and hl >= 1 and lh == 0:
        bias = "uptrend"
    elif lh >= 1 and ll >= 1 and hh == 0:
        bias = "downtrend"
    elif hh + hl > lh + ll:
        bias = "uptrend"
    elif lh + ll > hh + hl:
        bias = "downtrend"

    trendline = _build_trendline(ohlcv, bias=bias, lows=lows, highs=highs)
    trend_break = _detect_trend_break(bias, trendline, close)
    exhaustion = _detect_exhaustion(ohlcv, bias, highs, lows)
    acceleration = _detect_acceleration(ohlcv, bias)

    weeks = max(1, len(ohlcv) // 5)
    if bias == "uptrend":
        summary = (
            f"Higher highs and higher lows for roughly {weeks} weeks. "
            f"Uptrend structure intact."
        )
        if trendline:
            summary += f" Trendline support near ${trendline['end_price']:.2f}."
    elif bias == "downtrend":
        summary = (
            f"Lower highs and lower lows dominate the recent structure. "
            f"Downtrend bias remains in control."
        )
        if trendline:
            summary += f" Trendline resistance near ${trendline['end_price']:.2f}."
    else:
        summary = "Mixed swing structure — no clean trend sequence yet."

    if trend_break:
        summary += " Recent price action suggests a potential trend break."
    if exhaustion:
        summary += " Momentum appears to be fading at the latest swings."
    if acceleration:
        summary += " Trend acceleration visible in recent slope."

    return TrendStructure(
        bias=bias,
        summary=summary,
        higher_highs=hh,
        higher_lows=hl,
        lower_highs=lh,
        lower_lows=ll,
        trend_break=trend_break,
        exhaustion=exhaustion,
        acceleration=acceleration,
        swing_points=tuple(swings),
        trendline=trendline,
    )


def _build_trendline(
    ohlcv: pd.DataFrame,
    *,
    bias: str,
    lows: list[SwingPoint],
    highs: list[SwingPoint],
) -> dict[str, Any] | None:
    points = lows[-2:] if bias in {"uptrend", "mixed"} and len(lows) >= 2 else None
    label = "Trendline support"
    if bias == "downtrend" and len(highs) >= 2:
        points = highs[-2:]
        label = "Trendline resistance"
    if not points or len(points) < 2:
        return None

    start, end = points[0], points[1]
    span = max(end.bar_index - start.bar_index, 1)
    slope = (end.price - start.price) / span
    end_index = len(ohlcv) - 1
    end_price = end.price + slope * (end_index - end.bar_index)

    return {
        "label": label,
        "start_bar_index": start.bar_index,
        "end_bar_index": end_index,
        "start_date": start.date,
        "end_date": _date_str(ohlcv.index, end_index),
        "start_price": round(start.price, 2),
        "end_price": round(float(end_price), 2),
        "style": "trendline",
    }


def _detect_trend_break(bias: str, trendline: dict[str, Any] | None, close: float) -> bool:
    if trendline is None:
        return False
    level = float(trendline["end_price"])
    if bias == "uptrend":
        return close < level * 0.985
    if bias == "downtrend":
        return close > level * 1.015
    return False


def _detect_exhaustion(
    ohlcv: pd.DataFrame,
    bias: str,
    highs: list[SwingPoint],
    lows: list[SwingPoint],
) -> bool:
    if len(ohlcv) < 25:
        return False
    ret_10 = float(ohlcv["close"].pct_change(10).iloc[-1])
    if bias == "uptrend" and highs:
        return ret_10 < 0.01 and highs[-1].bar_index >= len(ohlcv) - 8
    if bias == "downtrend" and lows:
        return ret_10 > -0.01 and lows[-1].bar_index >= len(ohlcv) - 8
    return False


def _detect_acceleration(ohlcv: pd.DataFrame, bias: str) -> bool:
    if len(ohlcv) < 40:
        return False
    ret_20 = float(ohlcv["close"].pct_change(20).iloc[-1])
    ret_5 = float(ohlcv["close"].pct_change(5).iloc[-1])
    if bias == "uptrend":
        return ret_5 > 0 and abs(ret_5) > abs(ret_20) * 0.6
    if bias == "downtrend":
        return ret_5 < 0 and abs(ret_5) > abs(ret_20) * 0.6
    return False


def find_support_resistance_zones(
    ohlcv: pd.DataFrame,
    *,
    lookback: int = 5,
    tolerance_pct: float = 0.012,
    max_zones: int = 4,
    recent_swing_bars: int = 252,
) -> tuple[list[PriceZone], list[PriceZone]]:
    """Cluster chart levels; return supports below price and resistances above price."""
    swings = find_swing_points(ohlcv, lookback=lookback, max_points=20)
    if recent_swing_bars > 0 and len(ohlcv) > recent_swing_bars:
        min_index = len(ohlcv) - recent_swing_bars
        swings = [s for s in swings if s.bar_index >= min_index]

    close = float(ohlcv["close"].iloc[-1])
    indicators = compute_indicators(ohlcv)
    atr = _latest_atr(ohlcv, indicators=indicators)
    candidates = _level_candidates(
        ohlcv=ohlcv,
        swings=swings,
        indicators=indicators,
        close=close,
    )
    supports = _cluster_level_candidates(
        [candidate for candidate in candidates if candidate.zone_type == "support"],
        zone_type="support",
        tolerance_pct=tolerance_pct,
        max_zones=max_zones,
        close=close,
        atr=atr,
        total_bars=len(ohlcv),
    )
    resistances = _cluster_level_candidates(
        [candidate for candidate in candidates if candidate.zone_type == "resistance"],
        zone_type="resistance",
        tolerance_pct=tolerance_pct,
        max_zones=max_zones,
        close=close,
        atr=atr,
        total_bars=len(ohlcv),
    )

    supports = [z for z in supports if z.price_high < close * 1.005]
    resistances = [z for z in resistances if z.price_low > close * 0.995]

    supports.sort(key=lambda z: close - z.price_high)
    resistances.sort(key=lambda z: z.price_low - close)
    return supports[:max_zones], resistances[:max_zones]


def _level_candidates(
    *,
    ohlcv: pd.DataFrame,
    swings: list[SwingPoint],
    indicators: pd.DataFrame,
    close: float,
) -> list[_LevelCandidate]:
    candidates = [
        _LevelCandidate(
            price=swing.price,
            zone_type="support" if swing.kind == "low" else "resistance",
            source="swing",
            bar_index=swing.bar_index,
            strength=0.5,
        )
        for swing in swings
    ]
    last_index = len(ohlcv) - 1

    def add_prior_high_low(window: int, source_strength: float) -> None:
        if len(ohlcv) <= window:
            return
        window_frame = ohlcv.iloc[-window - 1 : -1]
        if window_frame.empty:
            return
        low_idx = int(ohlcv.index.get_loc(window_frame["low"].idxmin()))
        high_idx = int(ohlcv.index.get_loc(window_frame["high"].idxmax()))
        candidates.append(
            _LevelCandidate(
                price=float(window_frame["low"].min()),
                zone_type="support",
                source="priorHighLow",
                bar_index=low_idx,
                strength=source_strength,
            )
        )
        candidates.append(
            _LevelCandidate(
                price=float(window_frame["high"].max()),
                zone_type="resistance",
                source="priorHighLow",
                bar_index=high_idx,
                strength=source_strength,
            )
        )

    if len(ohlcv) >= 2:
        prior = ohlcv.iloc[-2]
        candidates.extend(
            [
                _LevelCandidate(
                    price=float(prior["low"]),
                    zone_type="support",
                    source="priorHighLow",
                    bar_index=last_index - 1,
                    strength=0.72,
                ),
                _LevelCandidate(
                    price=float(prior["high"]),
                    zone_type="resistance",
                    source="priorHighLow",
                    bar_index=last_index - 1,
                    strength=0.72,
                ),
            ]
        )
    add_prior_high_low(5, 0.64)
    add_prior_high_low(20, 0.58)

    for col in ("sma_20", "sma_50", "sma_200"):
        if col not in indicators.columns:
            continue
        value = _last_float(indicators[col])
        if value is None or value <= 0:
            continue
        candidates.append(
            _LevelCandidate(
                price=value,
                zone_type="support" if value <= close else "resistance",
                source="movingAverage",
                bar_index=last_index,
                strength=0.56 if col != "sma_200" else 0.5,
            )
        )

    round_step = _round_number_step(close)
    if round_step > 0:
        lower = np.floor(close / round_step) * round_step
        upper = np.ceil(close / round_step) * round_step
        for price, zone_type in ((lower, "support"), (upper, "resistance")):
            if price > 0 and abs(price - close) / max(close, 1e-9) <= 0.08:
                candidates.append(
                    _LevelCandidate(
                        price=float(price),
                        zone_type=zone_type,  # type: ignore[arg-type]
                        source="roundNumber",
                        bar_index=last_index,
                        strength=0.35,
                    )
                )

    return [
        candidate
        for candidate in candidates
        if candidate.price > 0 and np.isfinite(candidate.price)
    ]


def _cluster_level_candidates(
    candidates: list[_LevelCandidate],
    *,
    zone_type: Literal["support", "resistance"],
    tolerance_pct: float,
    max_zones: int,
    close: float,
    atr: float | None,
    total_bars: int,
) -> list[PriceZone]:
    if not candidates:
        return []

    clusters: list[list[_LevelCandidate]] = []
    for candidate in sorted(candidates, key=lambda item: item.price):
        placed = False
        for cluster in clusters:
            anchor = cluster[0].price
            tolerance = _cluster_tolerance(anchor, tolerance_pct=tolerance_pct, atr=atr)
            if abs(candidate.price - anchor) <= tolerance:
                cluster.append(candidate)
                placed = True
                break
        if not placed:
            clusters.append([candidate])

    zones: list[PriceZone] = []
    for cluster in clusters:
        prices = [s.price for s in cluster]
        low = min(prices)
        high = max(prices)
        pad = max((high - low) * 0.15, low * 0.002)
        if atr is not None and atr > 0:
            pad = max(pad, atr * 0.15)
        midpoint = (low + high) / 2
        distance_pct = abs(midpoint - close) / max(close, 1e-9)
        atr_distance = abs(midpoint - close) / atr if atr and atr > 0 else None
        recency_bars = min(
            (
                total_bars - 1 - candidate.bar_index
                for candidate in cluster
                if candidate.bar_index is not None
            ),
            default=None,
        )
        sources = tuple(sorted({candidate.source for candidate in cluster}))
        touches = sum(candidate.touches for candidate in cluster)
        price_low = round(low - pad, 2)
        price_high = round(high + pad, 2)
        role = _level_role(
            zone_type=zone_type,
            price_low=price_low,
            price_high=price_high,
            close=close,
            distance_pct=distance_pct,
            atr_distance=atr_distance,
            recency_bars=recency_bars,
            sources=sources,
            touches=touches,
        )
        timeframe = _level_timeframe(recency_bars=recency_bars, sources=sources)
        strength = _zone_strength(
            cluster=cluster,
            touches=touches,
            role=role,
            recency_bars=recency_bars,
            distance_pct=distance_pct,
        )
        label_prefix = "Support" if zone_type == "support" else "Resistance"
        zones.append(
            PriceZone(
                price_low=price_low,
                price_high=price_high,
                label=f"{label_prefix}: ${low:.2f}",
                zone_type=zone_type,
                touches=touches,
                strength=strength,
                midpoint=round(midpoint, 2),
                timeframe=timeframe,
                sources=sources,
                recency_bars=recency_bars,
                distance_pct_from_current=round(distance_pct * 100.0, 2),
                atr_distance=round(atr_distance, 2) if atr_distance is not None else None,
                level_role=role,
                actionable_for=_actionable_for(
                    zone_type,
                    role,
                    price_low=price_low,
                    price_high=price_high,
                    midpoint=midpoint,
                    close=close,
                ),
            )
        )

    zones.sort(
        key=lambda z: (
            z.level_role != "actionable",
            z.level_role != "nearbyContext",
            abs((z.midpoint or 0.0) - close),
            -z.strength,
        )
    )
    return zones[:max_zones]


def _cluster_tolerance(anchor: float, *, tolerance_pct: float, atr: float | None) -> float:
    pct_band = abs(anchor) * tolerance_pct
    atr_band = (atr or 0.0) * 0.35
    return max(pct_band, atr_band)


def _level_role(
    *,
    zone_type: Literal["support", "resistance"],
    price_low: float,
    price_high: float,
    close: float,
    distance_pct: float,
    atr_distance: float | None,
    recency_bars: int | None,
    sources: tuple[ZoneSource, ...],
    touches: int,
) -> ZoneRole:
    recent = recency_bars is None or recency_bars <= 65
    atr_ok = atr_distance is None or atr_distance <= 3.0
    side_ok = price_high < close if zone_type == "support" else price_low > close
    if (
        distance_pct <= 0.08
        and atr_ok
        and recent
        and side_ok
        and _has_actionable_structure(
            sources=sources,
            touches=touches,
            recency_bars=recency_bars,
        )
    ):
        return "actionable"
    if distance_pct <= 0.14 or (atr_distance is not None and atr_distance <= 5.0):
        return "nearbyContext"
    if "movingAverage" in sources and distance_pct <= 0.18:
        return "nearbyContext"
    return "majorHistorical"


def _has_actionable_structure(
    *,
    sources: tuple[ZoneSource, ...],
    touches: int,
    recency_bars: int | None,
) -> bool:
    recent = recency_bars is not None and recency_bars <= 65
    structural_sources = set(sources) - {"movingAverage", "roundNumber"}
    if touches >= 2 and recent and structural_sources:
        return True
    if recent and any(source in sources for source in ("swing", "priorHighLow")):
        return True
    if "movingAverage" in sources and len(set(sources) - {"roundNumber"}) >= 2:
        return True
    return False


def _level_timeframe(
    *,
    recency_bars: int | None,
    sources: tuple[ZoneSource, ...],
) -> ZoneTimeframe:
    if recency_bars is not None and recency_bars <= 20:
        return "shortTerm"
    if "movingAverage" in sources and recency_bars is not None and recency_bars <= 65:
        return "shortTerm"
    if recency_bars is None or recency_bars <= 120:
        return "intermediate"
    return "longTerm"


def _zone_strength(
    *,
    cluster: list[_LevelCandidate],
    touches: int,
    role: ZoneRole,
    recency_bars: int | None,
    distance_pct: float,
) -> float:
    source_score = max(candidate.strength for candidate in cluster)
    touch_bonus = min(0.25, touches * 0.04)
    recency_bonus = 0.12 if recency_bars is not None and recency_bars <= 20 else 0.0
    role_penalty = 0.0 if role == "actionable" else 0.08 if role == "nearbyContext" else 0.18
    distance_penalty = min(0.18, distance_pct)
    return round(max(0.05, min(1.0, source_score + touch_bonus + recency_bonus - role_penalty - distance_penalty)), 3)


def _actionable_for(
    zone_type: Literal["support", "resistance"],
    role: ZoneRole,
    *,
    price_low: float,
    price_high: float,
    midpoint: float,
    close: float,
) -> dict[str, bool]:
    actionable = role == "actionable"
    support_tradeable = actionable and zone_type == "support" and price_high < close and midpoint < close
    resistance_tradeable = actionable and zone_type == "resistance" and price_low > close and midpoint > close
    return {
        "chartContext": True,
        "tradeStop": support_tradeable,
        "tradeTarget": resistance_tradeable,
        "breakoutTrigger": resistance_tradeable,
    }


def _latest_atr(ohlcv: pd.DataFrame, *, indicators: pd.DataFrame) -> float | None:
    if "atr_14" in indicators.columns:
        value = _last_float(indicators["atr_14"])
        if value is not None and value > 0:
            return value
    if len(ohlcv) < 15:
        return None
    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    value = tr.rolling(14).mean().iloc[-1]
    return float(value) if not pd.isna(value) and value > 0 else None


def _round_number_step(close: float) -> float:
    if close >= 500:
        return 25.0
    if close >= 100:
        return 10.0
    if close >= 25:
        return 5.0
    return 1.0


def analyze_volume(ohlcv: pd.DataFrame, structure: TrendStructure) -> VolumeContext:
    volume = ohlcv["volume"].astype(float)
    vol_ma20 = volume.rolling(20).mean()
    vol_std20 = volume.rolling(20).std()
    latest_vol = float(volume.iloc[-1])
    ratio = float(latest_vol / vol_ma20.iloc[-1]) if vol_ma20.iloc[-1] else None
    zscore = (
        float((latest_vol - vol_ma20.iloc[-1]) / vol_std20.iloc[-1])
        if vol_std20.iloc[-1]
        else None
    )

    ret_5 = float(ohlcv["close"].pct_change(5).iloc[-1])
    spike = ratio is not None and ratio >= 1.8
    weak_move = ratio is not None and ratio <= 0.75 and abs(ret_5) >= 0.02
    up_day = float(ohlcv["close"].iloc[-1]) >= float(ohlcv["open"].iloc[-1])
    accumulation = (
        structure.bias == "uptrend"
        and ratio is not None
        and ratio >= 1.1
        and up_day
        and ret_5 > 0
    )
    distribution = (
        structure.bias in {"downtrend", "mixed"}
        and ratio is not None
        and ratio >= 1.1
        and not up_day
        and ret_5 < 0
    )
    breakout_confirmed = spike and structure.bias == "uptrend" and ret_5 > 0.01

    if breakout_confirmed and ratio is not None:
        summary = (
            f"Breakout occurred on {ratio:.1f}x average volume. "
            "Institutional participation likely."
        )
        label = "Breakout volume"
    elif accumulation:
        summary = "Volume supports recent gains — accumulation characteristics present."
        label = "Accumulation"
    elif distribution:
        summary = "Heavy volume on down days suggests distribution pressure."
        label = "Distribution"
    elif spike:
        summary = f"Volume spike at {ratio:.1f}x the 20-day average — watch follow-through."
        label = "Volume spike"
    elif weak_move:
        summary = "Price move lacks volume confirmation — conviction is limited."
        label = "Weak volume"
    else:
        summary = "Volume profile is unremarkable relative to the last 20 sessions."
        label = "Normal volume"

    return VolumeContext(
        label=label,
        summary=summary,
        vol_ratio_20d=ratio,
        vol_zscore_20d=zscore,
        spike=spike,
        accumulation=accumulation,
        distribution=distribution,
        breakout_confirmed=breakout_confirmed,
        weak_move=weak_move,
    )


def analyze_moving_averages(ohlcv: pd.DataFrame) -> MovingAverageContext:
    indicators = compute_indicators(ohlcv)
    close = float(ohlcv["close"].iloc[-1])
    sma_20 = _last_float(indicators.get("sma_20"))
    sma_50 = _last_float(indicators.get("sma_50"))
    sma_200 = _last_float(indicators.get("sma_200"))

    dist_20 = _dist_pct(close, sma_20)
    dist_50 = _dist_pct(close, sma_50)
    dist_200 = _dist_pct(close, sma_200)

    golden = False
    death = False
    if sma_50 is not None and sma_200 is not None and len(indicators) >= 2:
        prev_50 = _last_float(indicators["sma_50"].iloc[:-1])
        prev_200 = _last_float(indicators["sma_200"].iloc[:-1])
        if prev_50 is not None and prev_200 is not None:
            golden = prev_50 <= prev_200 and sma_50 > sma_200
            death = prev_50 >= prev_200 and sma_50 < sma_200

    parts: list[str] = []
    if dist_200 is not None:
        direction = "above" if dist_200 >= 0 else "below"
        parts.append(
            f"Price {abs(dist_200):.1f}% {direction} SMA200. "
            f"Long-term trend remains {'bullish' if dist_200 >= 0 else 'bearish'}."
        )
    if dist_50 is not None:
        parts.append(
            f"{'Above' if dist_50 >= 0 else 'Below'} SMA50 by {abs(dist_50):.1f}%."
        )
    if golden:
        parts.append("Fresh golden cross (SMA50 crossed above SMA200).")
    elif death:
        parts.append("Death cross in play (SMA50 below SMA200).")

    summary = " ".join(parts) if parts else "Moving-average context unavailable."

    window = min(len(ohlcv), 120)
    tail = ohlcv.iloc[-window:]
    tail_ind = indicators.iloc[-window:]
    sma_series: dict[str, list[dict[str, Any]]] = {}
    for key, col in [("sma20", "sma_20"), ("sma50", "sma_50"), ("sma200", "sma_200")]:
        if col not in tail_ind.columns:
            continue
        points = []
        for idx, value in tail_ind[col].items():
            if pd.isna(value):
                continue
            points.append(
                {
                    "date": pd.Timestamp(idx).strftime("%Y-%m-%d"),
                    "price": round(float(value), 2),
                }
            )
        sma_series[key] = points

    return MovingAverageContext(
        sma_20=sma_20,
        sma_50=sma_50,
        sma_200=sma_200,
        above_sma_20=(close > sma_20) if sma_20 else None,
        above_sma_50=(close > sma_50) if sma_50 else None,
        above_sma_200=(close > sma_200) if sma_200 else None,
        dist_sma_20_pct=dist_20,
        dist_sma_50_pct=dist_50,
        dist_sma_200_pct=dist_200,
        golden_cross=golden,
        death_cross=death,
        summary=summary,
        sma_series=sma_series,
    )


def _last_float(series: pd.Series | None) -> float | None:
    if series is None or series.empty:
        return None
    value = series.iloc[-1]
    if pd.isna(value):
        return None
    return float(value)


def _dist_pct(close: float, sma: float | None) -> float | None:
    if sma is None or sma == 0:
        return None
    return round((close / sma - 1.0) * 100.0, 1)


BreakoutKind = Literal[
    "failed_breakout",
    "failed_breakdown",
    "confirmed_breakout",
    "confirmed_breakdown",
]

FIB_CHANNEL_RATIOS = (0.0, 0.382, 0.5, 0.618, 1.0)


@dataclass(frozen=True)
class BreakoutEvent:
    kind: BreakoutKind
    bar_index: int
    date: str
    price: float
    zone_label: str
    label: str
    volume_ratio: float | None


def detect_breakout_events(
    ohlcv: pd.DataFrame,
    supports: list[PriceZone],
    resistances: list[PriceZone],
    *,
    lookback_bars: int = 60,
    follow_through_bars: int = 5,
    recent_only_bars: int = 25,
) -> list[BreakoutEvent]:
    """Detect confirmed and failed breaks of nearby support/resistance zones."""
    if len(ohlcv) < 25:
        return []

    high = ohlcv["high"].to_numpy(dtype=float)
    low = ohlcv["low"].to_numpy(dtype=float)
    close = ohlcv["close"].to_numpy(dtype=float)
    volume = ohlcv["volume"].astype(float)
    vol_ma20 = volume.rolling(20).mean()
    start = max(0, len(ohlcv) - lookback_bars)
    min_bar = max(0, len(ohlcv) - recent_only_bars)
    events: list[BreakoutEvent] = []

    def volume_ratio_at(bar_index: int) -> float | None:
        baseline = vol_ma20.iloc[bar_index]
        if baseline is None or pd.isna(baseline) or baseline <= 0:
            return None
        return float(volume.iloc[bar_index] / baseline)

    def scan_resistance(zone: PriceZone) -> BreakoutEvent | None:
        level = zone.price_high
        for pierce_idx in range(len(ohlcv) - 2, start, -1):
            if high[pierce_idx] <= level * 1.001:
                continue
            end_idx = min(pierce_idx + follow_through_bars, len(ohlcv) - 1)
            for check_idx in range(pierce_idx + 1, end_idx + 1):
                if close[check_idx] < level:
                    if check_idx < min_bar:
                        return None
                    return BreakoutEvent(
                        kind="failed_breakout",
                        bar_index=check_idx,
                        date=_date_str(ohlcv.index, check_idx),
                        price=round(float(close[check_idx]), 2),
                        zone_label=zone.label,
                        label="Failed breakout",
                        volume_ratio=volume_ratio_at(check_idx),
                    )
            if close[end_idx] > level * 1.002:
                if end_idx < min_bar:
                    return None
                return BreakoutEvent(
                    kind="confirmed_breakout",
                    bar_index=end_idx,
                    date=_date_str(ohlcv.index, end_idx),
                    price=round(float(close[end_idx]), 2),
                    zone_label=zone.label,
                    label="Breakout",
                    volume_ratio=volume_ratio_at(end_idx),
                )
        return None

    def scan_support(zone: PriceZone) -> BreakoutEvent | None:
        level = zone.price_low
        for pierce_idx in range(len(ohlcv) - 2, start, -1):
            if low[pierce_idx] >= level * 0.999:
                continue
            end_idx = min(pierce_idx + follow_through_bars, len(ohlcv) - 1)
            for check_idx in range(pierce_idx + 1, end_idx + 1):
                if close[check_idx] > level:
                    if check_idx < min_bar:
                        return None
                    return BreakoutEvent(
                        kind="failed_breakdown",
                        bar_index=check_idx,
                        date=_date_str(ohlcv.index, check_idx),
                        price=round(float(close[check_idx]), 2),
                        zone_label=zone.label,
                        label="Failed breakdown",
                        volume_ratio=volume_ratio_at(check_idx),
                    )
            if close[end_idx] < level * 0.998:
                if end_idx < min_bar:
                    return None
                return BreakoutEvent(
                    kind="confirmed_breakdown",
                    bar_index=end_idx,
                    date=_date_str(ohlcv.index, end_idx),
                    price=round(float(close[end_idx]), 2),
                    zone_label=zone.label,
                    label="Breakdown",
                    volume_ratio=volume_ratio_at(end_idx),
                )
        return None

    for zone in resistances[:2]:
        event = scan_resistance(zone)
        if event:
            events.append(event)
    for zone in supports[:2]:
        event = scan_support(zone)
        if event:
            events.append(event)

    return events


def build_fib_channel(
    ohlcv: pd.DataFrame,
    structure: TrendStructure,
) -> dict[str, Any] | None:
    """Build parallel fib-style channel rails from recent swing structure."""
    swings = list(structure.swing_points)
    lows = [s for s in swings if s.kind == "low"]
    highs = [s for s in swings if s.kind == "high"]
    if len(lows) < 2 and len(highs) < 2:
        return None

    end_idx = len(ohlcv) - 1
    use_downtrend = structure.bias == "downtrend" and len(highs) >= 2

    if use_downtrend:
        base_a, base_b = highs[-2], highs[-1]
        anchor_candidates = [
            low for low in lows if base_a.bar_index <= low.bar_index <= end_idx
        ]
        if not anchor_candidates:
            return None
        anchor = min(anchor_candidates, key=lambda point: point.price)
    else:
        if len(lows) < 2:
            return None
        base_a, base_b = lows[-2], lows[-1]
        anchor_candidates = [
            high for high in highs if base_a.bar_index <= high.bar_index <= end_idx
        ]
        if not anchor_candidates:
            return None
        anchor = max(anchor_candidates, key=lambda point: point.price)

    if base_b.bar_index <= base_a.bar_index:
        return None

    slope = (base_b.price - base_a.price) / (base_b.bar_index - base_a.bar_index)

    def rail_price(bar_index: int) -> float:
        return base_a.price + slope * (bar_index - base_a.bar_index)

    if use_downtrend:
        width = rail_price(anchor.bar_index) - anchor.price
        if width <= 0:
            return None
    else:
        width = anchor.price - rail_price(anchor.bar_index)
        if width <= 0:
            return None

    start_idx = base_a.bar_index
    sample_indexes = sorted(
        {
            start_idx,
            end_idx,
            *range(start_idx, end_idx + 1, max(1, (end_idx - start_idx) // 40)),
        }
    )

    lines: list[dict[str, Any]] = []
    for ratio in FIB_CHANNEL_RATIOS:
        points: list[dict[str, Any]] = []
        for bar_index in sample_indexes:
            base = rail_price(bar_index)
            price = base - width * ratio if use_downtrend else base + width * ratio
            points.append(
                {
                    "date": _date_str(ohlcv.index, bar_index),
                    "price": round(price, 2),
                }
            )
        pct_label = "100%" if ratio >= 1 else f"{ratio * 100:.1f}".rstrip("0").rstrip(".") + "%"
        lines.append(
            {
                "label": f"Fib {pct_label}",
                "style": "fib_channel",
                "ratio": ratio,
                "points": points,
            }
        )

    summary = (
        "Parallel fib channel anchored to swing highs — price testing upper rail."
        if use_downtrend
        else "Parallel fib channel anchored to higher lows — watch mid-channel support."
    )
    return {
        "bias": structure.bias,
        "summary": summary,
        "lines": lines,
    }


def breakout_event_dict(event: BreakoutEvent) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "bar_index": event.bar_index,
        "date": event.date,
        "price": event.price,
        "zone_label": event.zone_label,
        "label": event.label,
        "volume_ratio": event.volume_ratio,
    }
