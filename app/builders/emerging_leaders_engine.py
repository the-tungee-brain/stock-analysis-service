from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from app.models.emerging_leaders_models import SetupStageId

SetupStage = SetupStageId

STAGE_LABELS: dict[SetupStage, str] = {
    "BASE_BUILDING": "Stage 1: Base Building",
    "TIGHTENING": "Stage 2: Tightening",
    "BREAKOUT_WATCH": "Stage 3: Breakout Watch",
    "BREAKOUT_TRIGGERED": "Stage 4: Breakout Triggered",
    "EXTENDED": "Stage 5: Extended",
}

# Prefer Stage 2 tightening; Stage 3 watch is selective; extended deprioritized.
STAGE_RANK_PRIORITY: dict[SetupStage, int] = {
    "TIGHTENING": 120,
    "BASE_BUILDING": 85,
    "BREAKOUT_WATCH": 70,
    "BREAKOUT_TRIGGERED": 25,
    "EXTENDED": 5,
}

WEIGHT_VOL_CONTRACTION = 0.14
WEIGHT_RANGE_TIGHT = 0.14
WEIGHT_RESISTANCE = 0.12
WEIGHT_VOLUME_DRYUP = 0.10
WEIGHT_RS_IMPROVEMENT = 0.06
WEIGHT_ACCUMULATION = 0.08
WEIGHT_BASE_QUALITY = 0.12
WEIGHT_BREAKOUT_PROXIMITY = 0.10
WEIGHT_DORMANCY = 0.08
WEIGHT_TIGHTENING_TREND = 0.10
WEIGHT_SETUP_PURITY = 0.06

MIN_BARS = 60
MIN_SETUP_PURITY = 38
MIN_PURITY_STAGE_2 = 52
MIN_PURITY_STAGE_3 = 58


@dataclass(frozen=True)
class SetupComponentScores:
    volatility_contraction: float
    range_tightening: float
    resistance_tests_score: float
    resistance_tests: int
    volume_dryup: float
    rs_improvement: float
    accumulation: float
    base_quality: float
    breakout_proximity: float
    ret_5d: float
    ret_10d: float
    ret_20d: float
    distance_from_breakout_pct: float
    resistance_level: float
    tightening_trend: float
    vol_contraction_deep: float
    dormancy_days: int
    base_age: int
    failed_resistance_tests: int
    rsi_14: float
    setup_purity_score: float
    momentum_leader_like: bool


@dataclass(frozen=True)
class EmergingLeaderEvaluation:
    symbol: str
    setup_quality_score: int
    setup_stage: SetupStage
    why_it_ranks: str
    positive_factors: list[str]
    missing_factors: list[str]
    next_confirmation: list[str]
    sort_priority: int
    components: SetupComponentScores


def evaluate_emerging_leader(symbol: str, ohlcv: pd.DataFrame) -> EmergingLeaderEvaluation | None:
    frame = _normalize_ohlcv(ohlcv)
    if frame is None or len(frame) < MIN_BARS:
        return None

    components = _compute_components(frame)
    if not _passes_structural_filter(components):
        return None

    setup_score = _setup_quality_score(components)
    stage = _resolve_stage(frame, components)
    positive, missing = _factor_lists(components, stage)
    next_conf = _next_confirmation(stage, components)
    why = _why_it_ranks(stage, setup_score, components, positive)

    return EmergingLeaderEvaluation(
        symbol=symbol.upper(),
        setup_quality_score=setup_score,
        setup_stage=stage,
        why_it_ranks=why,
        positive_factors=positive,
        missing_factors=missing,
        next_confirmation=next_conf,
        sort_priority=STAGE_RANK_PRIORITY[stage],
        components=components,
    )


def ranking_sort_key(item: EmergingLeaderEvaluation) -> tuple[int, int, int, str]:
    return (
        item.sort_priority,
        int(item.components.setup_purity_score),
        item.setup_quality_score,
        item.symbol,
    )


def passes_emerging_leader_list(item: EmergingLeaderEvaluation) -> bool:
    """Exclude momentum-heavy / extended names from surfaced results."""
    if item.components.momentum_leader_like:
        return False
    if item.components.setup_purity_score < MIN_SETUP_PURITY:
        return False
    if item.setup_stage == "EXTENDED":
        return False
    if item.setup_stage == "BREAKOUT_TRIGGERED":
        return item.components.setup_purity_score >= 45
    return True


def _normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    df = raw.copy()
    rename = {}
    for need in ("open", "high", "low", "close", "volume"):
        for existing in df.columns:
            if str(existing).lower() == need:
                rename[existing] = need
    df = df.rename(columns=rename)
    required = {"high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.tail(252)


def _compute_components(frame: pd.DataFrame) -> SetupComponentScores:
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    volume = frame["volume"].astype(float)

    atr14 = _atr(high, low, close, 14)
    atr_recent = float(atr14.tail(5).mean())
    atr_mid = float(atr14.tail(20).head(10).mean()) or atr_recent
    atr_prior = float(atr14.tail(45).head(25).mean()) or atr_mid
    vol_ratio_shallow = atr_recent / atr_mid if atr_mid > 0 else 1.0
    vol_ratio_deep = atr_recent / atr_prior if atr_prior > 0 else 1.0
    volatility_contraction = _score_lower_is_better(vol_ratio_shallow, good=0.68, ok=0.82, bad=0.98)
    vol_contraction_deep = _score_lower_is_better(vol_ratio_deep, good=0.62, ok=0.78, bad=0.95)

    daily_range = (high - low) / close.replace(0, np.nan)
    range_5 = float(daily_range.tail(5).mean())
    range_15 = float(daily_range.tail(15).mean()) or range_5
    range_30 = float(daily_range.tail(30).mean()) or range_15
    range_ratio = range_5 / range_30 if range_30 > 0 else 1.0
    range_tightening = _score_lower_is_better(range_ratio, good=0.58, ok=0.72, bad=0.92)
    tightening_trend = _tightening_trend_score(daily_range)

    resistance_level = float(high.tail(60).max())
    tests = _count_resistance_tests(high, resistance_level, lookback=60)
    failed_tests = _count_failed_resistance_tests(
        high, close, resistance_level, lookback=60
    )
    resistance_tests_score = float(min(100, 30 + tests * 12 + failed_tests * 8))

    vol_10 = float(volume.tail(10).mean())
    vol_30 = float(volume.tail(30).mean()) or vol_10
    vol_dry_ratio = vol_10 / vol_30 if vol_30 > 0 else 1.0
    volume_dryup = _score_lower_is_better(vol_dry_ratio, good=0.68, ok=0.85, bad=1.05)

    rs_improvement = _rs_improvement_score(close)
    accumulation = _accumulation_score(close, volume)

    dormancy_days = _dormancy_days(close, daily_range)
    base_age = _base_age_days(close, resistance_level)
    base_depth = (
        (resistance_level - float(close.iloc[-1])) / resistance_level
        if resistance_level > 0
        else 0
    )
    base_len_score = _base_length_score_from_age(base_age, dormancy_days)
    base_quality = float(
        min(
            100,
            base_len_score * 0.5
            + _score_mid_band(base_depth, low=0.05, high=0.14) * 0.35
            + min(100, dormancy_days * 2.5) * 0.15,
        )
    )

    distance_pct = max(
        0.0,
        (resistance_level - float(close.iloc[-1])) / float(close.iloc[-1]),
    )
    breakout_proximity = _proximity_score(distance_pct)

    ret_5d = float(close.pct_change(5).iloc[-1]) if len(close) > 5 else 0.0
    ret_10d = float(close.pct_change(10).iloc[-1]) if len(close) > 10 else 0.0
    ret_20d = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0
    rsi_14 = _rsi(close, 14)

    setup_purity_score = _setup_purity_score(
        volatility_contraction=volatility_contraction,
        vol_contraction_deep=vol_contraction_deep,
        range_tightening=range_tightening,
        tightening_trend=tightening_trend,
        dormancy_days=dormancy_days,
        base_age=base_age,
        distance_pct=distance_pct * 100.0,
        ret_5d=ret_5d,
        ret_10d=ret_10d,
        ret_20d=ret_20d,
        rsi_14=rsi_14,
        failed_tests=failed_tests,
        resistance_level=resistance_level,
        close=float(close.iloc[-1]),
    )
    momentum_leader_like = _momentum_leader_like(
        ret_5d=ret_5d,
        ret_10d=ret_10d,
        ret_20d=ret_20d,
        rs_improvement=rs_improvement,
        rsi_14=rsi_14,
        distance_pct=distance_pct * 100.0,
        setup_purity_score=setup_purity_score,
    )

    return SetupComponentScores(
        volatility_contraction=volatility_contraction,
        range_tightening=range_tightening,
        resistance_tests_score=resistance_tests_score,
        resistance_tests=tests,
        volume_dryup=volume_dryup,
        rs_improvement=rs_improvement,
        accumulation=accumulation,
        base_quality=base_quality,
        breakout_proximity=breakout_proximity,
        ret_5d=ret_5d,
        ret_10d=ret_10d,
        ret_20d=ret_20d,
        distance_from_breakout_pct=distance_pct * 100.0,
        resistance_level=resistance_level,
        tightening_trend=tightening_trend,
        vol_contraction_deep=vol_contraction_deep,
        dormancy_days=dormancy_days,
        base_age=base_age,
        failed_resistance_tests=failed_tests,
        rsi_14=rsi_14,
        setup_purity_score=setup_purity_score,
        momentum_leader_like=momentum_leader_like,
    )


def _passes_structural_filter(c: SetupComponentScores) -> bool:
    if c.momentum_leader_like:
        return False
    if c.setup_purity_score < MIN_SETUP_PURITY:
        return False
    if c.ret_20d > 0.14:
        return False
    if c.ret_10d > 0.09 and c.rsi_14 > 68:
        return False
    if c.rsi_14 > 74 and c.distance_from_breakout_pct < 2.0:
        return False
    if c.ret_20d > 0.08 and c.rs_improvement >= 78:
        return False
    if c.distance_from_breakout_pct < 0.4 and c.ret_5d > 0.03:
        return False
    return True


def _setup_quality_score(c: SetupComponentScores) -> int:
    dormancy_score = float(min(100, 35 + c.dormancy_days * 2.2))
    raw = (
        WEIGHT_VOL_CONTRACTION * c.volatility_contraction
        + WEIGHT_RANGE_TIGHT * c.range_tightening
        + WEIGHT_RESISTANCE * c.resistance_tests_score
        + WEIGHT_VOLUME_DRYUP * c.volume_dryup
        + WEIGHT_RS_IMPROVEMENT * c.rs_improvement
        + WEIGHT_ACCUMULATION * c.accumulation
        + WEIGHT_BASE_QUALITY * c.base_quality
        + WEIGHT_BREAKOUT_PROXIMITY * c.breakout_proximity
        + WEIGHT_DORMANCY * dormancy_score
        + WEIGHT_TIGHTENING_TREND * c.tightening_trend
        + WEIGHT_SETUP_PURITY * c.setup_purity_score
    )
    penalty = 0.0
    if c.ret_20d > 0.08:
        penalty += min(18.0, (c.ret_20d - 0.08) * 100)
    if c.ret_10d > 0.06:
        penalty += min(10.0, (c.ret_10d - 0.06) * 80)
    if c.rsi_14 > 70:
        penalty += min(12.0, (c.rsi_14 - 70) * 1.2)
    blended = raw * 0.88 + c.setup_purity_score * 0.12
    return int(round(max(0.0, min(100.0, blended - penalty))))


def _resolve_stage(frame: pd.DataFrame, c: SetupComponentScores) -> SetupStage:
    close = float(frame["close"].iloc[-1])
    resistance = c.resistance_level
    vol_recent = float(frame["volume"].tail(5).mean())
    vol_base = float(frame["volume"].tail(30).mean()) or vol_recent
    rel_vol = vol_recent / vol_base if vol_base > 0 else 1.0
    dist = c.distance_from_breakout_pct

    if c.ret_20d > 0.13 or (c.ret_10d > 0.08 and dist < 1.5):
        return "EXTENDED"
    if c.rsi_14 > 72 and dist < 2.0 and c.ret_10d > 0.04:
        return "EXTENDED"
    if close >= resistance * 0.999 and rel_vol >= 1.4:
        return "BREAKOUT_TRIGGERED"

    tightening_ok = (
        c.vol_contraction_deep >= 68
        and c.tightening_trend >= 65
        and c.range_tightening >= 62
        and c.dormancy_days >= 10
        and c.setup_purity_score >= MIN_PURITY_STAGE_2
        and abs(c.ret_10d) <= 0.05
        and 2.5 <= dist <= 9.0
    )
    if tightening_ok:
        return "TIGHTENING"

    watch_ok = (
        c.volatility_contraction >= 70
        and c.tightening_trend >= 60
        and 1.2 <= dist <= 3.8
        and c.failed_resistance_tests >= 2
        and c.setup_purity_score >= MIN_PURITY_STAGE_3
        and c.ret_20d <= 0.07
        and rel_vol < 1.25
    )
    if watch_ok:
        return "BREAKOUT_WATCH"

    if c.volatility_contraction >= 55 or c.dormancy_days >= 8:
        return "BASE_BUILDING"
    return "BASE_BUILDING"


def _factor_lists(
    c: SetupComponentScores,
    stage: SetupStage,
) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    missing: list[str] = []

    if c.vol_contraction_deep >= 65:
        positive.append("Meaningful volatility contraction")
    elif c.volatility_contraction >= 55:
        positive.append("Volatility contraction")
    else:
        missing.append("Volatility contraction")

    if c.tightening_trend >= 65:
        positive.append("Multi-day range compression")
    elif c.range_tightening >= 55:
        positive.append("Tight range")
    else:
        missing.append("Tight range")

    if c.failed_resistance_tests >= 2:
        positive.append(
            f"Resistance tested {c.failed_resistance_tests} times without breakout"
        )
    elif c.resistance_tests >= 2:
        positive.append(f"Resistance tested {c.resistance_tests} times")
    else:
        missing.append("Repeated resistance tests")

    if c.dormancy_days >= 12:
        positive.append(f"Dormant consolidation ({c.dormancy_days} days)")
    elif c.dormancy_days >= 6:
        positive.append("Consolidation dormancy building")
    else:
        missing.append("Consolidation dormancy")

    if c.base_age >= 15:
        positive.append(f"Base structure age ({c.base_age} days)")
    elif c.base_age >= 8:
        positive.append("Developing base structure")

    if c.volume_dryup >= 55:
        positive.append("Volume dry-up during consolidation")
    else:
        missing.append("Volume dry-up during consolidation")

    if c.setup_purity_score >= 60:
        positive.append("High pre-breakout setup purity")
    elif c.setup_purity_score >= 45:
        positive.append("Moderate setup purity")
    else:
        missing.append("Pre-breakout purity")

    if stage in {"BREAKOUT_TRIGGERED", "EXTENDED"}:
        missing.append("Still pre-breakout (already advanced)")
    elif 1.5 <= c.distance_from_breakout_pct <= 5.0:
        positive.append("Near breakout level")
    else:
        missing.append("Breakout above resistance")

    if stage not in {"BREAKOUT_TRIGGERED", "EXTENDED"}:
        missing.append("Relative volume > 1.5x")

    return positive[:6], missing[:5]


def _next_confirmation(stage: SetupStage, c: SetupComponentScores) -> list[str]:
    if stage == "EXTENDED":
        return ["Pullback to support", "Reset base before re-entry"]
    if stage == "BREAKOUT_TRIGGERED":
        return ["Hold breakout on volume", "Avoid chasing extension"]
    if stage == "BREAKOUT_WATCH":
        return [
            "Break above resistance on volume",
            "New 20-day high with expansion",
            "Sustain above base",
        ]
    if stage == "TIGHTENING":
        return [
            "Further range contraction",
            "Quiet test of resistance",
            "Volume expansion on first close above base",
        ]
    return [
        "Extend consolidation dormancy",
        "Tighten volatility further",
        "Build higher lows under resistance",
    ]


def _why_it_ranks(
    stage: SetupStage,
    score: int,
    c: SetupComponentScores,
    positive: list[str],
) -> str:
    lead = positive[0] if positive else "Consolidation structure forming"
    return (
        f"{STAGE_LABELS[stage]} with setup quality {score}/100 "
        f"(purity {int(c.setup_purity_score)}/100) — led by {lead.lower()}."
    )


def _setup_purity_score(
    *,
    volatility_contraction: float,
    vol_contraction_deep: float,
    range_tightening: float,
    tightening_trend: float,
    dormancy_days: int,
    base_age: int,
    distance_pct: float,
    ret_5d: float,
    ret_10d: float,
    ret_20d: float,
    rsi_14: float,
    failed_tests: int,
    resistance_level: float,
    close: float,
) -> float:
    quiet = 90.0 - min(50.0, abs(ret_5d) * 400 + abs(ret_10d) * 200 + abs(ret_20d) * 80)
    compression = (
        volatility_contraction * 0.25
        + vol_contraction_deep * 0.2
        + range_tightening * 0.2
        + tightening_trend * 0.2
    )
    structure = min(100.0, 30 + dormancy_days * 2.0 + base_age * 1.5 + failed_tests * 6)
    dist_score = 85.0 if 2.0 <= distance_pct <= 8.0 else 45.0 if distance_pct <= 12 else 25.0
    if distance_pct < 1.0:
        dist_score = 20.0
    if close >= resistance_level * 0.998:
        dist_score = 15.0

    penalty = 0.0
    if ret_20d > 0.07:
        penalty += (ret_20d - 0.07) * 120
    if ret_10d > 0.05:
        penalty += (ret_10d - 0.05) * 80
    if rsi_14 > 68:
        penalty += (rsi_14 - 68) * 1.5
    if ret_20d > 0.05 and rsi_14 > 62:
        penalty += 8.0

    raw = quiet * 0.3 + compression * 0.35 + structure * 0.2 + dist_score * 0.15
    return float(max(0.0, min(100.0, raw - penalty)))


def _momentum_leader_like(
    *,
    ret_5d: float,
    ret_10d: float,
    ret_20d: float,
    rs_improvement: float,
    rsi_14: float,
    distance_pct: float,
    setup_purity_score: float,
) -> bool:
    if ret_20d > 0.11:
        return True
    if ret_10d > 0.07 and rsi_14 > 65:
        return True
    if ret_20d > 0.07 and rs_improvement >= 75:
        return True
    if rsi_14 > 73 and distance_pct < 3.0:
        return True
    if ret_5d > 0.04 and distance_pct < 1.5:
        return True
    if setup_purity_score < 32:
        return True
    return False


def _tightening_trend_score(daily_range: pd.Series) -> float:
    r5 = daily_range.rolling(5, min_periods=3).mean()
    r20 = daily_range.rolling(20, min_periods=10).mean()
    ratio = (r5 / r20.replace(0, np.nan)).tail(18).dropna()
    if len(ratio) < 8:
        return 40.0
    last = float(ratio.iloc[-1])
    x = np.arange(len(ratio), dtype=float)
    slope = float(np.polyfit(x, ratio.values, 1)[0])
    level_score = _score_lower_is_better(last, good=0.55, ok=0.72, bad=0.9)
    trend_bonus = 15.0 if slope < -0.008 else 5.0 if slope < 0 else -10.0
    return float(max(0.0, min(100.0, level_score + trend_bonus)))


def _dormancy_days(close: pd.Series, daily_range: pd.Series) -> int:
    median_range = float(daily_range.tail(40).median()) or 0.01
    threshold = median_range * 0.82
    ret = close.pct_change().abs()
    quiet = (daily_range < threshold) & (ret < 0.012)
    streak = 0
    best = 0
    for val in quiet.tail(45):
        if bool(val):
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def _base_age_days(close: pd.Series, resistance: float) -> int:
    if resistance <= 0:
        return 0
    under = close < resistance * 0.985
    if not under.any():
        return 0
    streak = 0
    for val in under.tail(80):
        if bool(val):
            streak += 1
        else:
            streak = 0
    return streak


def _base_length_score_from_age(base_age: int, dormancy_days: int) -> float:
    age = max(base_age, dormancy_days)
    if age >= 28:
        return 92.0
    if age >= 18:
        return 78.0
    if age >= 10:
        return 58.0
    return 38.0


def _count_failed_resistance_tests(
    high: pd.Series,
    close: pd.Series,
    resistance: float,
    lookback: int,
) -> int:
    if resistance <= 0:
        return 0
    h = high.tail(lookback)
    c = close.tail(lookback)
    threshold = resistance * 0.985
    fail = (h >= threshold) & (c < resistance * 0.992)
    touches = fail.astype(int)
    return int((touches.diff().fillna(0) == 1).sum())


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    val = 100 - (100 / (1 + rs))
    if val.empty or np.isnan(val.iloc[-1]):
        return 50.0
    return float(val.iloc[-1])


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _count_resistance_tests(high: pd.Series, resistance: float, lookback: int) -> int:
    if resistance <= 0:
        return 0
    window = high.tail(lookback)
    threshold = resistance * 0.985
    touches = (window >= threshold).astype(int)
    groups = (touches.diff().fillna(0) == 1).sum()
    return int(groups) if groups > 0 else int(touches.sum() > 0)


def _rs_improvement_score(close: pd.Series) -> float:
    ret_21 = float(close.pct_change(21).iloc[-1]) if len(close) > 21 else 0.0
    ret_63 = float(close.pct_change(63).iloc[-1]) if len(close) > 63 else ret_21
    slope = ret_21 - ret_63 * 0.35
    if slope > 0.05:
        return 62.0
    if slope > 0.02:
        return 52.0
    if slope > -0.01:
        return 48.0
    if slope > -0.04:
        return 42.0
    return 35.0


def _accumulation_score(close: pd.Series, volume: pd.Series) -> float:
    ret = close.pct_change()
    recent = pd.DataFrame({"ret": ret, "vol": volume}).tail(20)
    up_vol = recent.loc[recent["ret"] > 0, "vol"].sum()
    down_vol = recent.loc[recent["ret"] <= 0, "vol"].sum() or 1.0
    ratio = up_vol / down_vol
    mean_ret = float(recent["ret"].mean())
    if ratio >= 1.25 and abs(mean_ret) < 0.003:
        return 72.0
    if ratio >= 1.1 and abs(mean_ret) < 0.006:
        return 58.0
    if ratio >= 0.95:
        return 48.0
    return 32.0


def _proximity_score(distance_pct: float) -> float:
    pct = distance_pct * 100.0
    if 2.0 <= pct <= 7.0:
        return 88.0
    if 7.0 < pct <= 11.0:
        return 62.0
    if 1.0 <= pct < 2.0:
        return 55.0
    if pct < 1.0:
        return 28.0
    return 35.0


def _score_lower_is_better(value: float, *, good: float, ok: float, bad: float) -> float:
    if value <= good:
        return 90.0
    if value <= ok:
        return 68.0
    if value <= bad:
        return 45.0
    return 25.0


def _score_mid_band(value: float, *, low: float, high: float) -> float:
    if low <= value <= high:
        return 85.0
    if value < low:
        return 55.0
    return 40.0
