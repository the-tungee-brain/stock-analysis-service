"""Candlestick pattern recognition engine (explanation layer, not standalone alpha)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

PatternDirection = Literal["bullish", "bearish", "neutral"]

PATTERN_CATALOG: tuple[str, ...] = (
    "hammer",
    "doji",
    "bullish_engulfing",
    "bearish_engulfing",
    "morning_star",
    "evening_star",
    "shooting_star",
    "three_white_soldiers",
    "three_black_crows",
)


@dataclass(frozen=True)
class CandlestickPatternHit:
    pattern_id: str
    label: str
    direction: PatternDirection
    strength: float  # 0..1 quality of the formation
    as_of_date: str
    bar_index: int


def _body(open_: pd.Series, close: pd.Series) -> pd.Series:
    return (close - open_).abs()


def _upper_shadow(open_: pd.Series, high: pd.Series, close: pd.Series) -> pd.Series:
    return high - np.maximum(open_, close)


def _lower_shadow(open_: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return np.minimum(open_, close) - low


def _candle_range(high: pd.Series, low: pd.Series) -> pd.Series:
    return (high - low).replace(0, np.nan)


def _is_bullish(open_: pd.Series, close: pd.Series) -> pd.Series:
    return close > open_


def _is_bearish(open_: pd.Series, close: pd.Series) -> pd.Series:
    return close < open_


def _detect_hammer(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = _body(o, c)
    rng = _candle_range(h, l)
    lower = _lower_shadow(o, l, c)
    upper = _upper_shadow(o, h, c)
    body_ratio = body / rng
    hit = (
        (body_ratio <= 0.35)
        & (lower >= body * 2.0)
        & (upper <= body * 0.6)
        & (rng > 0)
    )
    strength = np.clip((lower / rng) - (body / rng), 0.0, 1.0)
    return hit.fillna(False), strength.fillna(0.0)


def _detect_shooting_star(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = _body(o, c)
    rng = _candle_range(h, l)
    lower = _lower_shadow(o, l, c)
    upper = _upper_shadow(o, h, c)
    body_ratio = body / rng
    hit = (
        (body_ratio <= 0.35)
        & (upper >= body * 2.0)
        & (lower <= body * 0.6)
        & (rng > 0)
    )
    strength = np.clip((upper / rng) - (body / rng), 0.0, 1.0)
    return hit.fillna(False), strength.fillna(0.0)


def _detect_doji(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = _body(o, c)
    rng = _candle_range(h, l)
    hit = (body / rng <= 0.1) & (rng > 0)
    strength = np.clip(1.0 - (body / rng), 0.0, 1.0)
    return hit.fillna(False), strength.fillna(0.0)


def _detect_engulfing(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    o, c = df["open"], df["close"]
    prev_o = o.shift(1)
    prev_c = c.shift(1)
    prev_bear = prev_c < prev_o
    prev_bull = prev_c > prev_o
    curr_bull = c > o
    curr_bear = c < o

    bullish = (
        prev_bear
        & curr_bull
        & (c >= prev_o)
        & (o <= prev_c)
        & (_body(o, c) > _body(prev_o, prev_c))
    )
    bearish = (
        prev_bull
        & curr_bear
        & (c <= prev_o)
        & (o >= prev_c)
        & (_body(o, c) > _body(prev_o, prev_c))
    )
    strength = np.clip(
        (_body(o, c) / _body(prev_o, prev_c).replace(0, np.nan)) / 3.0,
        0.0,
        1.0,
    )
    return (
        bullish.fillna(False),
        bearish.fillna(False),
        strength.fillna(0.0),
    )


def _detect_morning_star(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    o0, c0 = o.shift(2), c.shift(2)
    o1, c1 = o.shift(1), c.shift(1)
    body0 = _body(o0, c0)
    body1 = _body(o1, c1)
    body2 = _body(o, c)
    hit = (
        (c0 < o0)
        & (body0 > body1 * 2)
        & (body1 / _candle_range(h.shift(1), l.shift(1)) <= 0.4)
        & (c > o)
        & (c > (o0 + c0) / 2)
        & (body2 > body1)
    )
    strength = np.clip(body2 / body0.replace(0, np.nan), 0.0, 1.0)
    return hit.fillna(False), strength.fillna(0.0)


def _detect_evening_star(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    o0, c0 = o.shift(2), c.shift(2)
    o1, c1 = o.shift(1), c.shift(1)
    body0 = _body(o0, c0)
    body1 = _body(o1, c1)
    body2 = _body(o, c)
    hit = (
        (c0 > o0)
        & (body0 > body1 * 2)
        & (body1 / _candle_range(h.shift(1), l.shift(1)) <= 0.4)
        & (c < o)
        & (c < (o0 + c0) / 2)
        & (body2 > body1)
    )
    strength = np.clip(body2 / body0.replace(0, np.nan), 0.0, 1.0)
    return hit.fillna(False), strength.fillna(0.0)


def _detect_three_white_soldiers(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    o, c = df["open"], df["close"]
    bull0 = _is_bullish(o.shift(2), c.shift(2))
    bull1 = _is_bullish(o.shift(1), c.shift(1))
    bull2 = _is_bullish(o, c)
    rising = (c > c.shift(1)) & (c.shift(1) > c.shift(2))
    opens_inside = (o.shift(1) >= o.shift(2)) & (o.shift(1) <= c.shift(2))
    opens_inside2 = (o >= o.shift(1)) & (o <= c.shift(1))
    hit = bull0 & bull1 & bull2 & rising & opens_inside & opens_inside2
    strength = np.clip(
        (c - c.shift(2)) / c.shift(2).replace(0, np.nan) * 10,
        0.0,
        1.0,
    )
    return hit.fillna(False), strength.fillna(0.0)


def _detect_three_black_crows(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    o, c = df["open"], df["close"]
    bear0 = _is_bearish(o.shift(2), c.shift(2))
    bear1 = _is_bearish(o.shift(1), c.shift(1))
    bear2 = _is_bearish(o, c)
    falling = (c < c.shift(1)) & (c.shift(1) < c.shift(2))
    opens_inside = (o.shift(1) <= o.shift(2)) & (o.shift(1) >= c.shift(2))
    opens_inside2 = (o <= o.shift(1)) & (o >= c.shift(1))
    hit = bear0 & bear1 & bear2 & falling & opens_inside & opens_inside2
    strength = np.clip(
        (c.shift(2) - c) / c.shift(2).replace(0, np.nan) * 10,
        0.0,
        1.0,
    )
    return hit.fillna(False), strength.fillna(0.0)


PATTERN_LABELS: dict[str, str] = {
    "hammer": "Hammer",
    "doji": "Doji",
    "bullish_engulfing": "Bullish engulfing",
    "bearish_engulfing": "Bearish engulfing",
    "morning_star": "Morning star",
    "evening_star": "Evening star",
    "shooting_star": "Shooting star",
    "three_white_soldiers": "Three white soldiers",
    "three_black_crows": "Three black crows",
}


def scan_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a boolean/strength matrix indexed like ``df`` for each pattern."""
    if len(df) < 3:
        return pd.DataFrame(index=df.index)

    hits: dict[str, pd.Series] = {}
    strengths: dict[str, pd.Series] = {}

    hammer_hit, hammer_str = _detect_hammer(df)
    hits["hammer"] = hammer_hit
    strengths["hammer"] = hammer_str

    doji_hit, doji_str = _detect_doji(df)
    hits["doji"] = doji_hit
    strengths["doji"] = doji_str

    bull_eng, bear_eng, eng_str = _detect_engulfing(df)
    hits["bullish_engulfing"] = bull_eng
    strengths["bullish_engulfing"] = eng_str
    hits["bearish_engulfing"] = bear_eng
    strengths["bearish_engulfing"] = eng_str

    mstar_hit, mstar_str = _detect_morning_star(df)
    hits["morning_star"] = mstar_hit
    strengths["morning_star"] = mstar_str

    estar_hit, estar_str = _detect_evening_star(df)
    hits["evening_star"] = estar_hit
    strengths["evening_star"] = estar_str

    shoot_hit, shoot_str = _detect_shooting_star(df)
    hits["shooting_star"] = shoot_hit
    strengths["shooting_star"] = shoot_str

    tws_hit, tws_str = _detect_three_white_soldiers(df)
    hits["three_white_soldiers"] = tws_hit
    strengths["three_white_soldiers"] = tws_str

    tbc_hit, tbc_str = _detect_three_black_crows(df)
    hits["three_black_crows"] = tbc_hit
    strengths["three_black_crows"] = tbc_str

    out = pd.DataFrame(index=df.index)
    for pattern_id in PATTERN_CATALOG:
        out[f"hit_{pattern_id}"] = hits[pattern_id].astype(bool)
        out[f"strength_{pattern_id}"] = strengths[pattern_id].astype(float)
    return out


def _direction_for_pattern(pattern_id: str) -> PatternDirection:
    if pattern_id in {
        "hammer",
        "bullish_engulfing",
        "morning_star",
        "three_white_soldiers",
    }:
        return "bullish"
    if pattern_id in {
        "bearish_engulfing",
        "evening_star",
        "shooting_star",
        "three_black_crows",
    }:
        return "bearish"
    return "neutral"


def active_patterns_on_date(
    scan: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    lookback_days: int = 3,
) -> list[CandlestickPatternHit]:
    """Patterns detected on ``as_of`` or within the recent lookback window."""
    if scan.empty or as_of not in scan.index:
        return []

    loc = scan.index.get_loc(as_of)
    if isinstance(loc, slice):
        loc = loc.stop - 1
    start = max(0, int(loc) - lookback_days + 1)
    window = scan.iloc[start : int(loc) + 1]
    results: list[CandlestickPatternHit] = []

    for pattern_id in PATTERN_CATALOG:
        hit_col = f"hit_{pattern_id}"
        str_col = f"strength_{pattern_id}"
        if hit_col not in window.columns:
            continue
        hit_rows = window.index[window[hit_col].astype(bool)]
        if len(hit_rows) == 0:
            continue
        hit_date = hit_rows[-1]
        strength = float(window.loc[hit_date, str_col])
        results.append(
            CandlestickPatternHit(
                pattern_id=pattern_id,
                label=PATTERN_LABELS[pattern_id],
                direction=_direction_for_pattern(pattern_id),
                strength=float(np.clip(strength, 0.05, 1.0)),
                as_of_date=pd.Timestamp(hit_date).strftime("%Y-%m-%d"),
                bar_index=int(scan.index.get_loc(hit_date)),
            )
        )

    results.sort(key=lambda item: item.strength, reverse=True)
    return results
