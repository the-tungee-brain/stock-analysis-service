"""Detect market regime from SPY trend and volatility."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta

from ranking_pipeline.regime.constants import (
    REGIME_HIGH_VOL_CHOP,
    REGIME_RISK_OFF,
    REGIME_RISK_ON_CHOP,
    REGIME_RISK_ON_TREND,
)
from ranking_pipeline.regime.multipliers import regime_score_multiplier


@dataclass(frozen=True)
class RegimeSnapshot:
    date: str
    regime_id: str
    regime_multiplier: float
    spy_trend_score: float
    vol_percentile: float
    risk_tone: str


def compute_spy_regime_series(spy_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily regime_id from SPY OHLCV.

    Uses only data through each bar's close (no lookahead).
    """
    df = spy_ohlcv.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")

    sma20 = ta.sma(close, length=20)
    sma200 = ta.sma(close, length=200)
    atr = ta.atr(high, low, close, length=14)

    trend_spread = sma20 / sma200 - 1.0
    sma20_slope = sma20.pct_change(5)

    vol_pct = atr.rolling(252, min_periods=60).apply(
        lambda s: float(pd.Series(s).rank(pct=True).iloc[-1]) if len(s) else float("nan"),
        raw=False,
    )

    out = pd.DataFrame(index=df.index)
    out["spy_trend_spread"] = trend_spread
    out["spy_sma20_slope_5d"] = sma20_slope
    out["spy_vol_percentile"] = vol_pct

    regime_ids: list[str] = []
    risk_tones: list[str] = []
    multipliers: list[float] = []

    for idx in out.index:
        spread = out.at[idx, "spy_trend_spread"]
        slope = out.at[idx, "spy_sma20_slope_5d"]
        vol_p = out.at[idx, "spy_vol_percentile"]

        if pd.isna(spread) or pd.isna(slope) or pd.isna(vol_p):
            regime_ids.append(REGIME_RISK_ON_CHOP)
            risk_tones.append("unknown")
            multipliers.append(regime_score_multiplier(REGIME_RISK_ON_CHOP))
            continue

        bullish_trend = spread > 0 and slope > 0
        bearish_trend = spread < 0 and slope < 0
        high_vol = vol_p >= 0.70
        low_vol = vol_p <= 0.35

        if bearish_trend or (spread < 0 and high_vol):
            regime_id = REGIME_RISK_OFF
            tone = "risk_off"
        elif high_vol and not bullish_trend:
            regime_id = REGIME_HIGH_VOL_CHOP
            tone = "chop"
        elif bullish_trend and low_vol:
            regime_id = REGIME_RISK_ON_TREND
            tone = "risk_on"
        else:
            regime_id = REGIME_RISK_ON_CHOP
            tone = "neutral"

        regime_ids.append(regime_id)
        risk_tones.append(tone)
        multipliers.append(regime_score_multiplier(regime_id))

    out["regime_id"] = regime_ids
    out["risk_tone"] = risk_tones
    out["regime_multiplier"] = multipliers
    return out.dropna(subset=["regime_id"])


def regime_for_date(regime_df: pd.DataFrame, as_of: pd.Timestamp) -> RegimeSnapshot | None:
    """Return regime snapshot for the latest row on or before ``as_of``."""
    if regime_df.empty:
        return None
    subset = regime_df[regime_df.index <= as_of]
    if subset.empty:
        return None
    row = subset.iloc[-1]
    date_str = pd.Timestamp(row.name).strftime("%Y-%m-%d")
    return RegimeSnapshot(
        date=date_str,
        regime_id=str(row["regime_id"]),
        regime_multiplier=float(row["regime_multiplier"]),
        spy_trend_score=float(row.get("spy_trend_spread", 0.0) or 0.0),
        vol_percentile=float(row.get("spy_vol_percentile", 0.5) or 0.5),
        risk_tone=str(row.get("risk_tone", "neutral")),
    )
