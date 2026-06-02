"""Market structure analysis for chart intelligence overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from features.indicators import compute_indicators

SwingKind = Literal["high", "low"]
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
) -> tuple[list[PriceZone], list[PriceZone]]:
    swings = find_swing_points(ohlcv, lookback=lookback, max_points=20)
    close = float(ohlcv["close"].iloc[-1])
    supports = _cluster_zones(
        [s for s in swings if s.kind == "low"],
        zone_type="support",
        tolerance_pct=tolerance_pct,
        max_zones=max_zones,
    )
    resistances = _cluster_zones(
        [s for s in swings if s.kind == "high"],
        zone_type="resistance",
        tolerance_pct=tolerance_pct,
        max_zones=max_zones,
    )

    supports.sort(key=lambda z: abs(close - z.price_low))
    resistances.sort(key=lambda z: abs(close - z.price_high))
    return supports[:max_zones], resistances[:max_zones]


def _cluster_zones(
    swings: list[SwingPoint],
    *,
    zone_type: Literal["support", "resistance"],
    tolerance_pct: float,
    max_zones: int,
) -> list[PriceZone]:
    if not swings:
        return []

    clusters: list[list[SwingPoint]] = []
    for swing in sorted(swings, key=lambda s: s.price):
        placed = False
        for cluster in clusters:
            anchor = cluster[0].price
            if abs(swing.price - anchor) / max(anchor, 1e-9) <= tolerance_pct:
                cluster.append(swing)
                placed = True
                break
        if not placed:
            clusters.append([swing])

    zones: list[PriceZone] = []
    for cluster in clusters:
        prices = [s.price for s in cluster]
        low = min(prices)
        high = max(prices)
        pad = max((high - low) * 0.15, low * 0.002)
        label_prefix = "Support" if zone_type == "support" else "Resistance"
        zones.append(
            PriceZone(
                price_low=round(low - pad, 2),
                price_high=round(high + pad, 2),
                label=f"{label_prefix}: ${low:.2f}",
                zone_type=zone_type,
                touches=len(cluster),
                strength=min(1.0, 0.35 + 0.15 * len(cluster)),
            )
        )

    zones.sort(key=lambda z: z.strength, reverse=True)
    return zones[:max_zones]


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
