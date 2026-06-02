"""Rule-based model contribution breakdown."""

from __future__ import annotations

from typing import Any

from models.prediction_service import KEY_INDICATORS


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_contributors(
    indicators: dict[str, float],
    *,
    chart_context: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    positive: list[str] = []
    negative: list[str] = []

    rs21 = indicators.get("rs_vs_spy_21d")
    rs63 = indicators.get("rs_vs_spy_63d")
    rs126 = indicators.get("rs_vs_spy_126d")
    vs20 = indicators.get("close_vs_sma20")
    vs200 = indicators.get("close_vs_sma200")
    ret21 = indicators.get("ret_21d")
    ret63 = indicators.get("ret_63d")

    if rs21 is not None and rs21 > 0.02:
        positive.append(f"Strong RS 21d ({_fmt_pct(rs21)})")
    elif rs21 is not None and rs21 < -0.02:
        negative.append(f"Weak RS 21d ({_fmt_pct(rs21)})")

    if rs63 is not None and rs63 > 0.03:
        positive.append(f"Strong RS 63d ({_fmt_pct(rs63)})")
    elif rs63 is not None and rs63 < -0.03:
        negative.append(f"Weak RS 63d ({_fmt_pct(rs63)})")

    if rs126 is not None and rs126 > 0.05:
        positive.append(f"Strong RS 126d ({_fmt_pct(rs126)})")
    elif rs126 is not None and rs126 < -0.03:
        negative.append(f"Weak RS 126d ({_fmt_pct(rs126)})")

    if vs200 is not None and vs200 > 0:
        positive.append("Above SMA200")
    elif vs200 is not None and vs200 < -0.02:
        negative.append("Below SMA200")

    if vs20 is not None and vs20 > 0:
        positive.append("Above SMA20")
    elif vs20 is not None and vs20 < 0:
        negative.append("Price lost SMA20")

    if ret21 is not None and ret21 > 0.03:
        positive.append("Positive trend structure (21d)")
    elif ret21 is not None and ret21 < -0.03:
        negative.append("Negative trend structure (21d)")

    if ret21 is not None and ret63 is not None and ret63 > 0.05 and ret21 < ret63 * 0.4:
        negative.append("Momentum slowing")

    if chart_context:
        structure_bias = chart_context.get("structure_bias")
        if structure_bias == "bullish":
            positive.append("Positive trend structure")
        elif structure_bias == "bearish":
            negative.append("Bearish market structure")

        if chart_context.get("near_resistance"):
            negative.append("Near resistance")
        if chart_context.get("near_support"):
            positive.append("Near support")
        if chart_context.get("volume_confirmed") is True:
            positive.append("Volume confirmation present")
        elif chart_context.get("volume_confirmed") is False:
            negative.append("Volume confirmation absent")

    if not positive and not negative:
        for key in KEY_INDICATORS:
            value = indicators.get(key)
            if value is None:
                continue
            if value > 0:
                positive.append(f"{key.replace('_', ' ')} supportive")
            elif value < 0:
                negative.append(f"{key.replace('_', ' ')} headwind")

    return {
        "positive": positive[:5],
        "negative": negative[:5],
    }


def contributor_deltas(
    today: dict[str, float],
    prior: dict[str, float],
    *,
    chart_context: dict[str, Any] | None = None,
    prior_chart_context: dict[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Explain score change from feature deltas."""
    drivers_positive: list[str] = []
    drivers_negative: list[str] = []

    rs21_delta = _delta(today, prior, "rs_vs_spy_21d")
    if rs21_delta is not None:
        if rs21_delta < -0.01:
            drivers_negative.append("Relative strength weakened")
        elif rs21_delta > 0.01:
            drivers_positive.append("Relative strength improved")

    vs20_today = today.get("close_vs_sma20")
    vs20_prior = prior.get("close_vs_sma20")
    if vs20_today is not None and vs20_prior is not None:
        if vs20_prior >= 0 and vs20_today < 0:
            drivers_negative.append("Price lost SMA20")
        elif vs20_prior < 0 and vs20_today >= 0:
            drivers_positive.append("Price reclaimed SMA20")

    ret21_delta = _delta(today, prior, "ret_21d")
    if ret21_delta is not None and ret21_delta < -0.02:
        drivers_negative.append("Momentum slowing")
    elif ret21_delta is not None and ret21_delta > 0.02:
        drivers_positive.append("Momentum improving")

    if chart_context and prior_chart_context:
        if not prior_chart_context.get("near_resistance") and chart_context.get("near_resistance"):
            drivers_negative.append("Resistance rejection")
        if prior_chart_context.get("volume_confirmed") and not chart_context.get("volume_confirmed"):
            drivers_negative.append("Volume confirmation absent")
        if not prior_chart_context.get("volume_confirmed") and chart_context.get("volume_confirmed"):
            drivers_positive.append("Volume confirmation returned")

    today_contrib = build_contributors(today, chart_context=chart_context)
    for item in today_contrib["negative"]:
        if item not in drivers_negative and len(drivers_negative) < 4:
            drivers_negative.append(item.split(" (")[0])

    return {
        "positive": drivers_positive[:4],
        "negative": drivers_negative[:4],
    }


def _delta(today: dict[str, float], prior: dict[str, float], key: str) -> float | None:
    if key not in today or key not in prior:
        return None
    return float(today[key]) - float(prior[key])
