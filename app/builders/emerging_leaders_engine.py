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

STAGE_RANK_PRIORITY: dict[SetupStage, int] = {
    "TIGHTENING": 100,
    "BREAKOUT_WATCH": 110,
    "BASE_BUILDING": 60,
    "BREAKOUT_TRIGGERED": 35,
    "EXTENDED": 15,
}

WEIGHT_VOL_CONTRACTION = 0.15
WEIGHT_RANGE_TIGHT = 0.15
WEIGHT_RESISTANCE = 0.15
WEIGHT_VOLUME_DRYUP = 0.12
WEIGHT_RS_IMPROVEMENT = 0.18
WEIGHT_ACCUMULATION = 0.10
WEIGHT_BASE_QUALITY = 0.10
WEIGHT_BREAKOUT_PROXIMITY = 0.15

MIN_BARS = 60


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
    ret_20d: float
    distance_from_breakout_pct: float
    resistance_level: float


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


def ranking_sort_key(item: EmergingLeaderEvaluation) -> tuple[int, int, str]:
    return (item.sort_priority, item.setup_quality_score, item.symbol)


def _normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None
    df = raw.copy()
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    for need in ("open", "high", "low", "close", "volume"):
        for existing in df.columns:
            if existing.lower() == need:
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

    ret = close.pct_change()
    atr14 = _atr(high, low, close, 14)
    atr_recent = float(atr14.tail(10).mean())
    atr_prior = float(atr14.tail(30).head(20).mean()) or atr_recent
    vol_ratio = atr_recent / atr_prior if atr_prior > 0 else 1.0
    volatility_contraction = _score_lower_is_better(vol_ratio, good=0.75, ok=0.95, bad=1.15)

    range_10 = float(((high - low) / close).tail(10).mean())
    range_30 = float(((high - low) / close).tail(30).mean()) or range_10
    range_ratio = range_10 / range_30 if range_30 > 0 else 1.0
    range_tightening = _score_lower_is_better(range_ratio, good=0.65, ok=0.85, bad=1.05)

    resistance_level = float(high.tail(60).max())
    tests = _count_resistance_tests(high, resistance_level, lookback=60)
    resistance_tests_score = float(min(100, 35 + tests * 16))

    vol_10 = float(volume.tail(10).mean())
    vol_30 = float(volume.tail(30).mean()) or vol_10
    vol_dry_ratio = vol_10 / vol_30 if vol_30 > 0 else 1.0
    volume_dryup = _score_lower_is_better(vol_dry_ratio, good=0.72, ok=0.9, bad=1.1)

    rs_improvement = _rs_improvement_score(close)

    accumulation = _accumulation_score(close, volume)

    base_depth = (resistance_level - float(close.iloc[-1])) / resistance_level if resistance_level > 0 else 0
    base_len = _base_length_score(close, resistance_level)
    base_quality = float(min(100, base_len * 0.55 + _score_mid_band(base_depth, low=0.04, high=0.18) * 0.45))

    distance_pct = max(0.0, (resistance_level - float(close.iloc[-1])) / float(close.iloc[-1]))
    breakout_proximity = _proximity_score(distance_pct)

    ret_20d = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0

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
        ret_20d=ret_20d,
        distance_from_breakout_pct=distance_pct * 100.0,
        resistance_level=resistance_level,
    )


def _setup_quality_score(c: SetupComponentScores) -> int:
    raw = (
        WEIGHT_VOL_CONTRACTION * c.volatility_contraction
        + WEIGHT_RANGE_TIGHT * c.range_tightening
        + WEIGHT_RESISTANCE * c.resistance_tests_score
        + WEIGHT_VOLUME_DRYUP * c.volume_dryup
        + WEIGHT_RS_IMPROVEMENT * c.rs_improvement
        + WEIGHT_ACCUMULATION * c.accumulation
        + WEIGHT_BASE_QUALITY * c.base_quality
        + WEIGHT_BREAKOUT_PROXIMITY * c.breakout_proximity
    )
    penalty = 0.0
    if c.ret_20d > 0.12:
        penalty += min(20.0, (c.ret_20d - 0.12) * 120)
    if c.ret_20d > 0.18:
        penalty += 10.0
    return int(round(max(0.0, min(100.0, raw - penalty))))


def _resolve_stage(frame: pd.DataFrame, c: SetupComponentScores) -> SetupStage:
    close = float(frame["close"].iloc[-1])
    resistance = c.resistance_level
    vol_recent = float(frame["volume"].tail(5).mean())
    vol_base = float(frame["volume"].tail(30).mean()) or vol_recent
    rel_vol = vol_recent / vol_base if vol_base > 0 else 1.0

    if c.ret_20d > 0.14 and close >= resistance * 0.97:
        return "EXTENDED"
    if close >= resistance * 0.998 and rel_vol >= 1.25:
        return "BREAKOUT_TRIGGERED"
    if c.distance_from_breakout_pct <= 3.5 and c.volatility_contraction >= 55:
        return "BREAKOUT_WATCH"
    if c.volatility_contraction >= 58 and c.range_tightening >= 55:
        return "TIGHTENING"
    return "BASE_BUILDING"


def _factor_lists(
    c: SetupComponentScores,
    stage: SetupStage,
) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    missing: list[str] = []

    if c.volatility_contraction >= 55:
        positive.append("Volatility contraction")
    else:
        missing.append("Volatility contraction")

    if c.range_tightening >= 55:
        positive.append("Tight range")
    else:
        missing.append("Tight range")

    if c.resistance_tests >= 3:
        positive.append(f"Resistance tested {c.resistance_tests} times")
    elif c.resistance_tests >= 2:
        positive.append("Resistance tested multiple times")
    else:
        missing.append("Repeated resistance tests")

    if c.volume_dryup >= 55:
        positive.append("Volume dry-up during consolidation")
    else:
        missing.append("Volume dry-up during consolidation")

    if c.rs_improvement >= 55:
        positive.append("Relative strength improving")
    else:
        missing.append("Relative strength trend improvement")

    if c.accumulation >= 55:
        positive.append("Accumulation-style volume pattern")
    else:
        missing.append("Institutional accumulation signals")

    if c.base_quality >= 55:
        positive.append("Base / consolidation structure")
    else:
        missing.append("Base quality")

    if stage in {"BREAKOUT_TRIGGERED", "EXTENDED"}:
        missing.append("Still pre-breakout (already advanced)")
    else:
        if c.distance_from_breakout_pct <= 4:
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
            "Break above recent high",
            "New 20-day high",
            "Volume expansion",
        ]
    if stage == "TIGHTENING":
        return [
            "Tighter daily range",
            "Test resistance again",
            "Volume expansion on breakout",
        ]
    return [
        "Form higher lows in base",
        "Tighten volatility",
        "Build RS vs market",
    ]


def _why_it_ranks(
    stage: SetupStage,
    score: int,
    c: SetupComponentScores,
    positive: list[str],
) -> str:
    lead = positive[0] if positive else "Consolidation structure forming"
    return (
        f"{STAGE_LABELS[stage]} with setup quality {score}/100 — "
        f"led by {lead.lower()}."
    )


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
    if slope > 0.06:
        return 88.0
    if slope > 0.03:
        return 72.0
    if slope > 0.0:
        return 58.0
    if slope > -0.04:
        return 42.0
    return 25.0


def _accumulation_score(close: pd.Series, volume: pd.Series) -> float:
    ret = close.pct_change()
    recent = pd.DataFrame({"ret": ret, "vol": volume}).tail(20)
    up_vol = recent.loc[recent["ret"] > 0, "vol"].sum()
    down_vol = recent.loc[recent["ret"] <= 0, "vol"].sum() or 1.0
    ratio = up_vol / down_vol
    if ratio >= 1.35 and float(recent["ret"].mean()) >= 0:
        return 82.0
    if ratio >= 1.15:
        return 65.0
    if ratio >= 0.95:
        return 50.0
    return 32.0


def _base_length_score(close: pd.Series, resistance: float) -> float:
    below = close < resistance * 0.99
    if not below.any():
        return 40.0
    streak = 0
    best = 0
    for val in below.tail(60):
        if val:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    if best >= 25:
        return 90.0
    if best >= 15:
        return 72.0
    if best >= 8:
        return 55.0
    return 38.0


def _proximity_score(distance_pct: float) -> float:
    pct = distance_pct * 100.0
    if 0.5 <= pct <= 3.5:
        return 92.0
    if 3.5 < pct <= 6.0:
        return 70.0
    if 6.0 < pct <= 10.0:
        return 48.0
    if pct < 0.5:
        return 35.0
    return 30.0


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
