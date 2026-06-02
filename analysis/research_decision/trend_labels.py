"""Multi-timeframe trend classification."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

TrendLabel = Literal["bullish", "bearish", "neutral"]

TREND_LABELS: tuple[TrendLabel, ...] = ("bullish", "bearish", "neutral")


def _label_from_momentum(
    *,
    above_long: bool | None,
    above_short: bool | None,
    momentum: float | None,
    short_momentum: float | None = None,
) -> TrendLabel:
    bullish_signals = 0
    bearish_signals = 0

    if above_long is True:
        bullish_signals += 1
    elif above_long is False:
        bearish_signals += 1

    if above_short is True:
        bullish_signals += 1
    elif above_short is False:
        bearish_signals += 1

    if momentum is not None:
        if momentum > 0.01:
            bullish_signals += 1
        elif momentum < -0.01:
            bearish_signals += 1

    if short_momentum is not None:
        if short_momentum > 0.005:
            bullish_signals += 1
        elif short_momentum < -0.005:
            bearish_signals += 1

    if bullish_signals >= 2 and bearish_signals == 0:
        return "bullish"
    if bearish_signals >= 2 and bullish_signals == 0:
        return "bearish"
    if bullish_signals > bearish_signals:
        return "bullish"
    if bearish_signals > bullish_signals:
        return "bearish"
    return "neutral"


def classify_daily_trend(indicators: dict[str, float]) -> TrendLabel:
    close_vs_sma20 = indicators.get("close_vs_sma20")
    close_vs_sma200 = indicators.get("close_vs_sma200")
    ret_21d = indicators.get("ret_21d")
    return _label_from_momentum(
        above_long=(close_vs_sma200 > 0) if close_vs_sma200 is not None else None,
        above_short=(close_vs_sma20 > 0) if close_vs_sma20 is not None else None,
        momentum=ret_21d,
    )


def classify_weekly_trend(ohlcv: pd.DataFrame) -> TrendLabel:
    weekly = ohlcv.resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna(subset=["close"])
    if len(weekly) < 12:
        return "neutral"

    close = weekly["close"].astype("float64")
    sma_10 = close.rolling(10, min_periods=10).mean()
    sma_30 = close.rolling(30, min_periods=20).mean()
    ret_4w = close.pct_change(4)

    latest_close = float(close.iloc[-1])
    latest_sma10 = float(sma_10.iloc[-1]) if pd.notna(sma_10.iloc[-1]) else None
    latest_sma30 = float(sma_30.iloc[-1]) if pd.notna(sma_30.iloc[-1]) else None
    latest_ret = float(ret_4w.iloc[-1]) if pd.notna(ret_4w.iloc[-1]) else None

    return _label_from_momentum(
        above_long=(latest_close > latest_sma30) if latest_sma30 is not None else None,
        above_short=(latest_close > latest_sma10) if latest_sma10 is not None else None,
        momentum=latest_ret,
    )


def classify_forecast_trend(
    *,
    prediction: int | None,
    ranking_score: float | None,
    min_up_prob: float | None = None,
) -> TrendLabel:
    if ranking_score is not None and min_up_prob is not None:
        if ranking_score >= min_up_prob:
            return "bullish"
        if ranking_score <= (1.0 - min_up_prob):
            return "bearish"
    if ranking_score is not None:
        if ranking_score >= 0.55:
            return "bullish"
        if ranking_score <= 0.45:
            return "bearish"
    if prediction is not None:
        if prediction > 0:
            return "bullish"
        if prediction < 0:
            return "bearish"
    return "neutral"


def synthesize_multi_timeframe_conclusion(
    weekly: TrendLabel,
    daily: TrendLabel,
    forecast: TrendLabel,
) -> str:
    if weekly == daily == forecast:
        tone = weekly.capitalize()
        return f"Aligned {tone.lower()} view across weekly, daily, and 5-day horizon."

    if weekly == "bullish" and daily == "bullish" and forecast == "bearish":
        return "Short-term pullback inside a longer-term uptrend."
    if weekly == "bearish" and daily == "bearish" and forecast == "bullish":
        return "Near-term bounce inside a longer-term downtrend."
    if weekly == "bullish" and forecast == "bearish":
        return "5-day model flags weakness against a constructive weekly backdrop."
    if weekly == "bearish" and forecast == "bullish":
        return "5-day model improves while the weekly trend remains under pressure."
    if daily != forecast and weekly == "neutral":
        return f"Daily {daily} signal diverges from the 5-day {forecast} forecast."
    return (
        f"Weekly {weekly}, daily {daily}, and 5-day {forecast} — mixed timeframe alignment."
    )


def trend_label_display(label: TrendLabel) -> str:
    return label.capitalize()


def build_multi_timeframe_payload(
    *,
    weekly: TrendLabel,
    daily: TrendLabel,
    forecast: TrendLabel,
) -> dict[str, Any]:
    return {
        "weekly_trend": weekly,
        "weekly_trend_label": trend_label_display(weekly),
        "daily_trend": daily,
        "daily_trend_label": trend_label_display(daily),
        "forecast_trend": forecast,
        "forecast_trend_label": trend_label_display(forecast),
        "conclusion": synthesize_multi_timeframe_conclusion(weekly, daily, forecast),
    }
