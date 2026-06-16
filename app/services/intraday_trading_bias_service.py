from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.builders.intraday_trading_bias_engine import (
    EASTERN,
    IntradayBar,
    IntradayTradingBiasInputs,
    OPENING_RANGE_END,
    classify_session,
    evaluate_intraday_trading_bias,
)
from app.models.intraday_trading_bias_models import IntradayTradingBiasResponse
from app.services.pattern_intelligence_service import (
    build_pattern_intelligence_payload,
    pattern_intelligence_to_api_dict,
)
from models.prediction_service import LoadedModel

logger = logging.getLogger(__name__)


def build_intraday_trading_bias(
    symbol: str,
    *,
    yfinance_adapter: YFinanceAdapter,
    loaded_model: LoadedModel | None = None,
    pattern_analysis_service=None,
    research_events_service=None,
) -> IntradayTradingBiasResponse:
    """Build a delayed intraday bias from polled 5-minute yfinance bars."""
    symbol_upper = symbol.strip().upper()
    data_gaps: list[str] = []
    warnings: list[str] = []

    bars = _load_intraday_bars(
        yfinance_adapter,
        symbol_upper,
        data_gaps=data_gaps,
        warnings=warnings,
    )
    market_bars = _load_market_bars(
        yfinance_adapter,
        data_gaps=data_gaps,
        warnings=warnings,
    )
    support, resistance = _load_daily_levels(
        symbol_upper,
        loaded_model=loaded_model,
        pattern_analysis_service=pattern_analysis_service,
        data_gaps=data_gaps,
    )
    catalyst = _load_catalyst_alignment(
        symbol_upper,
        research_events_service=research_events_service,
    )

    response = evaluate_intraday_trading_bias(
        IntradayTradingBiasInputs(
            symbol=symbol_upper,
            bars=bars,
            market_bars=market_bars,
            support=support,
            resistance=resistance,
            catalyst=catalyst,
            data_gaps=data_gaps,
            warnings=warnings,
        )
    )
    _log_intraday_setup_trace(symbol_upper, bars, response)
    return response


def _log_intraday_setup_trace(
    symbol_upper: str,
    bars: list[IntradayBar],
    response: IntradayTradingBiasResponse,
) -> None:
    regular_bars = [bar for bar in bars if bar.session == "regular"]
    trading_date = _latest_trading_date(regular_bars or bars)
    opening_range_bars = [
        bar
        for bar in regular_bars
        if bar.timestamp.astimezone(EASTERN).time() < OPENING_RANGE_END
    ]
    opening_range_high = _max_high(opening_range_bars)
    opening_range_low = _min_low(opening_range_bars)
    vwap = _vwap(regular_bars)
    rejection_reason = _intraday_rejection_reason(response)

    logger.info(
        (
            "Intraday day-trade setup trace: symbol=%s trading_date=%s "
            "intraday_bars=%s regular_bars=%s opening_range_high=%s "
            "opening_range_low=%s vwap=%s setup_status=%s rejection_reason=%s"
        ),
        symbol_upper,
        trading_date.isoformat() if trading_date else None,
        len(bars),
        len(regular_bars),
        _round_for_log(opening_range_high),
        _round_for_log(opening_range_low),
        _round_for_log(vwap),
        f"{response.bias}/{response.setup_type}/{response.action}",
        rejection_reason,
    )


def _latest_trading_date(bars: list[IntradayBar]) -> date | None:
    if not bars:
        return None
    return max(bar.timestamp.astimezone(EASTERN).date() for bar in bars)


def _intraday_rejection_reason(response: IntradayTradingBiasResponse) -> str | None:
    if response.levels.open_range_high is None or response.levels.open_range_low is None:
        if response.data_gaps:
            return "; ".join(response.data_gaps)
        if response.warnings:
            return "; ".join(response.warnings)
        return "opening_range_missing"
    if response.action in {"Avoid", "RiskOff", "Watch"}:
        reasons = response.reasons or response.warnings or response.data_gaps
        return "; ".join(reasons) if reasons else response.action
    return None


def _max_high(bars: list[IntradayBar]) -> float | None:
    if not bars:
        return None
    return max(bar.high for bar in bars)


def _min_low(bars: list[IntradayBar]) -> float | None:
    if not bars:
        return None
    return min(bar.low for bar in bars)


def _vwap(bars: list[IntradayBar]) -> float | None:
    volume_total = sum(max(bar.volume, 0) for bar in bars)
    if volume_total <= 0:
        return None
    dollar_volume = sum(
        ((bar.high + bar.low + bar.close) / 3) * max(bar.volume, 0)
        for bar in bars
    )
    return dollar_volume / volume_total


def _round_for_log(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _load_intraday_bars(
    yfinance_adapter: YFinanceAdapter,
    symbol_upper: str,
    *,
    data_gaps: list[str],
    warnings: list[str],
) -> list[IntradayBar]:
    try:
        hist = yfinance_adapter.get_history(
            symbol_upper,
            period="5d",
            interval="5m",
            prepost=True,
        )
    except Exception:
        logger.warning("Intraday yfinance bars unavailable for %s", symbol_upper)
        data_gaps.append("Intraday 5m bars unavailable")
        return []

    bars = _normalize_intraday_history(hist)
    if not bars:
        data_gaps.append("Intraday 5m bars unavailable")
        return []
    if not any(bar.session == "premarket" for bar in bars):
        warnings.append("Premarket bars were not provided by yfinance.")
    return _latest_session_bars(bars)


def _load_market_bars(
    yfinance_adapter: YFinanceAdapter,
    *,
    data_gaps: list[str],
    warnings: list[str],
) -> list[IntradayBar]:
    for market_symbol in ("SPY", "QQQ"):
        bars = _load_intraday_bars(
            yfinance_adapter,
            market_symbol,
            data_gaps=[],
            warnings=[],
        )
        if bars:
            return bars
    data_gaps.append("SPY/QQQ intraday market bars unavailable")
    warnings.append("Broad-market intraday context is unavailable.")
    return []


def _normalize_intraday_history(hist: pd.DataFrame | None) -> list[IntradayBar]:
    if hist is None or hist.empty or not isinstance(hist.index, pd.DatetimeIndex):
        return []

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(hist.columns):
        return []

    bars: list[IntradayBar] = []
    for timestamp, row in hist.iterrows():
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        py_ts = ts.to_pydatetime()
        open_price = _float_or_none(row.get("Open"))
        high = _float_or_none(row.get("High"))
        low = _float_or_none(row.get("Low"))
        close = _float_or_none(row.get("Close"))
        if open_price is None or high is None or low is None or close is None:
            continue
        bars.append(
            IntradayBar(
                timestamp=py_ts,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=max(0, int(_float_or_none(row.get("Volume")) or 0)),
                session=classify_session(py_ts),
            )
        )
    return sorted(bars, key=lambda bar: bar.timestamp)


def _latest_session_bars(bars: list[IntradayBar]) -> list[IntradayBar]:
    if not bars:
        return []
    latest_regular_dates = [
        bar.timestamp.date() for bar in bars if bar.session == "regular"
    ]
    if latest_regular_dates:
        latest_date = max(latest_regular_dates)
    else:
        latest_date = max(bar.timestamp.date() for bar in bars)
    return [bar for bar in bars if bar.timestamp.date() == latest_date]


def _load_daily_levels(
    symbol_upper: str,
    *,
    loaded_model: LoadedModel | None,
    pattern_analysis_service,
    data_gaps: list[str],
) -> tuple[float | None, float | None]:
    try:
        payload = _load_pattern_payload(
            symbol_upper,
            loaded_model=loaded_model,
            pattern_analysis_service=pattern_analysis_service,
        )
    except Exception:
        logger.warning(
            "Intraday bias daily levels unavailable for %s",
            symbol_upper,
            exc_info=True,
        )
        data_gaps.append("Daily support/resistance unavailable")
        return None, None

    support, resistance = _extract_levels(payload)
    if support is None and resistance is None:
        data_gaps.append("Daily support/resistance unavailable")
    return support, resistance


def _load_pattern_payload(
    symbol_upper: str,
    *,
    loaded_model: LoadedModel | None,
    pattern_analysis_service,
) -> dict[str, Any] | None:
    if loaded_model is not None and pattern_analysis_service is not None:
        snapshot = pattern_analysis_service.get_or_build(symbol_upper, loaded_model)
        return dict(snapshot.pattern_intelligence)

    pattern = build_pattern_intelligence_payload(symbol_upper, loaded_model)
    if pattern is None:
        return None
    return pattern_intelligence_to_api_dict(pattern)


def _extract_levels(payload: dict[str, Any] | None) -> tuple[float | None, float | None]:
    if not payload:
        return None, None
    chart = _get(payload, "chart_intelligence", "chartIntelligence") or {}
    supports = _get(chart, "support_zones", "supportZones") or []
    resistances = _get(chart, "resistance_zones", "resistanceZones") or []

    support = None
    if supports:
        first = supports[0]
        support = _float_or_none(_get(first, "price_high", "priceHigh"))
    resistance = None
    if resistances:
        first = resistances[0]
        resistance = _float_or_none(_get(first, "price_low", "priceLow"))
    return support, resistance


def _load_catalyst_alignment(
    symbol_upper: str,
    *,
    research_events_service,
) -> str:
    if research_events_service is None:
        return "none"
    try:
        events = list(research_events_service.get_events(symbol=symbol_upper) or [])[:3]
    except Exception:
        logger.warning(
            "Intraday bias catalyst context unavailable for %s",
            symbol_upper,
            exc_info=True,
        )
        return "none"
    text = " ".join(
        f"{getattr(event, 'title', '')} {getattr(event, 'detail', '')}".lower()
        for event in events
    )
    if not text.strip():
        return "none"
    positive_terms = ("beat", "raise", "raised", "upgrade", "approval", "growth")
    negative_terms = ("miss", "cut", "downgrade", "probe", "lawsuit", "warning")
    positive = sum(1 for term in positive_terms if term in text)
    negative = sum(1 for term in negative_terms if term in text)
    if positive > negative:
        return "positive"
    if negative > positive:
        return "negative"
    return "neutral"


def _get(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
