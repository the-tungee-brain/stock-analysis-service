from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.builders.intraday_trading_bias_engine import (
    IntradayBar,
    IntradayTradingBiasInputs,
    evaluate_intraday_trading_bias,
)

ET = ZoneInfo("America/New_York")


def _bar(
    hour: int,
    minute: int,
    *,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: int = 1_000,
    session: str = "regular",
) -> IntradayBar:
    return IntradayBar(
        timestamp=datetime(2026, 6, 5, hour, minute, tzinfo=ET),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        session=session,
    )


def _bullish_bars() -> list[IntradayBar]:
    return [
        _bar(8, 30, open_price=97, high=98, low=96, close=97.5, session="premarket"),
        _bar(9, 0, open_price=98, high=100, low=97, close=99.5, session="premarket"),
        _bar(9, 30, open_price=100, high=101, low=99, close=100.5, volume=800),
        _bar(9, 35, open_price=100.5, high=101.5, low=100, close=101, volume=900),
        _bar(9, 40, open_price=101, high=102, low=100.8, close=101.7, volume=950),
        _bar(9, 45, open_price=101.7, high=102.4, low=101.4, close=102, volume=1_000),
        _bar(9, 50, open_price=102, high=102.8, low=101.8, close=102.5, volume=1_050),
        _bar(9, 55, open_price=102.5, high=103, low=102.2, close=102.8, volume=1_100),
        _bar(10, 0, open_price=102.8, high=104, low=102.6, close=103.8, volume=1_600),
        _bar(10, 5, open_price=103.8, high=105, low=103.5, close=104.7, volume=1_900),
        _bar(10, 10, open_price=104.7, high=106, low=104.5, close=105.8, volume=2_200),
    ]


def test_missing_intraday_bars_returns_neutral_low_with_data_gap():
    result = evaluate_intraday_trading_bias(
        IntradayTradingBiasInputs(symbol="AAPL", bars=[])
    )

    assert result.bias == "Neutral"
    assert result.confidence == "Low"
    assert result.setup_type == "None"
    assert result.action == "Watch"
    assert "Intraday 5m bars unavailable" in result.data_gaps
    assert result.is_realtime is False


def test_gap_and_hold_above_premarket_high_above_vwap_strong_volume_is_bullish():
    result = evaluate_intraday_trading_bias(
        IntradayTradingBiasInputs(
            symbol="AAPL",
            bars=_bullish_bars(),
            market_bars=_bullish_bars(),
            support=98,
            resistance=108,
            now=datetime(2026, 6, 5, 10, 15, tzinfo=ET),
        )
    )

    assert result.bias == "Bullish"
    assert result.confidence in {"High", "Medium"}
    assert result.setup_type in {"GapAndGo", "OpeningRangeBreakout"}
    assert result.alignment.vwap in {"above", "reclaiming"}
    assert result.alignment.volume == "confirmed"
    assert result.is_realtime is False


def test_vwap_reject_and_weak_market_returns_bearish_or_neutral():
    bars = [
        _bar(9, 30, open_price=100, high=101, low=99.5, close=100.8, volume=1_800),
        _bar(9, 35, open_price=100.8, high=101, low=100, close=100.5, volume=1_600),
        _bar(9, 40, open_price=100.5, high=100.7, low=99.8, close=100, volume=1_500),
        _bar(9, 45, open_price=100, high=100.2, low=99.2, close=99.5, volume=1_200),
        _bar(9, 50, open_price=99.5, high=99.8, low=98.8, close=99, volume=1_100),
        _bar(9, 55, open_price=99, high=99.2, low=98.5, close=98.7, volume=1_000),
        _bar(10, 0, open_price=98.7, high=99, low=98, close=98.2, volume=700),
        _bar(10, 5, open_price=98.2, high=98.5, low=97.5, close=97.7, volume=650),
        _bar(10, 10, open_price=97.7, high=98, low=97, close=97.2, volume=600),
    ]

    result = evaluate_intraday_trading_bias(
        IntradayTradingBiasInputs(
            symbol="AAPL",
            bars=bars,
            market_bars=bars,
            now=datetime(2026, 6, 5, 10, 15, tzinfo=ET),
        )
    )

    assert result.bias in {"Bearish", "Neutral"}
    assert result.alignment.market == "against"
    assert result.alignment.intraday_trend == "against"
    assert result.alignment.volume == "weak"


def test_before_opening_range_completes_no_orb_setup_and_warns():
    bars = [
        _bar(9, 30, open_price=100, high=101, low=99, close=100.5),
        _bar(9, 35, open_price=100.5, high=101.5, low=100, close=101),
        _bar(9, 40, open_price=101, high=102, low=100.8, close=101.5),
        _bar(9, 45, open_price=101.5, high=102.4, low=101.2, close=102),
    ]

    result = evaluate_intraday_trading_bias(
        IntradayTradingBiasInputs(
            symbol="AAPL",
            bars=bars,
            market_bars=bars,
            now=datetime(2026, 6, 5, 9, 50, tzinfo=ET),
        )
    )

    assert result.setup_type != "OpeningRangeBreakout"
    assert result.levels.open_range_high is None
    assert "Opening range is not complete yet." in result.warnings


def test_missing_market_bars_does_not_fail():
    result = evaluate_intraday_trading_bias(
        IntradayTradingBiasInputs(
            symbol="AAPL",
            bars=_bullish_bars(),
            market_bars=[],
            now=datetime(2026, 6, 5, 10, 15, tzinfo=ET),
        )
    )

    assert result.bias in {"Bullish", "Neutral", "Bearish"}
    assert "SPY/QQQ intraday market bars unavailable" in result.data_gaps
    assert result.alignment.market == "mixed"


def test_stale_intraday_bars_return_inactive_neutral_read():
    result = evaluate_intraday_trading_bias(
        IntradayTradingBiasInputs(
            symbol="AAPL",
            bars=_bullish_bars(),
            market_bars=_bullish_bars(),
            now=datetime(2026, 6, 5, 16, 42, tzinfo=ET),
        )
    )

    assert result.bias == "Neutral"
    assert result.confidence == "Low"
    assert result.setup_type == "None"
    assert result.action == "Watch"
    assert any("stale" in item.lower() for item in result.data_gaps)
    assert any("inactive" in item.lower() for item in result.warnings)


def test_after_market_close_returns_previous_session_neutral_read():
    result = evaluate_intraday_trading_bias(
        IntradayTradingBiasInputs(
            symbol="AAPL",
            bars=_bullish_bars(),
            market_bars=_bullish_bars(),
            now=datetime(2026, 6, 5, 16, 5, tzinfo=ET),
        )
    )

    assert result.bias == "Neutral"
    assert result.confidence == "Low"
    assert result.setup_type == "None"
    assert "Intraday read is stale or outside market hours" in result.data_gaps


def test_response_shape_uses_stable_aliases():
    result = evaluate_intraday_trading_bias(
        IntradayTradingBiasInputs(
            symbol="AAPL",
            bars=_bullish_bars(),
            market_bars=_bullish_bars(),
            now=datetime(2026, 6, 5, 10, 15, tzinfo=ET),
        )
    )

    payload = result.model_dump(mode="json", by_alias=True)
    assert set(payload) == {
        "bias",
        "confidence",
        "horizon",
        "setupType",
        "action",
        "levels",
        "alignment",
        "reasons",
        "warnings",
        "dataGaps",
        "lastUpdated",
        "stalenessSeconds",
        "provider",
        "isRealtime",
    }
    assert set(payload["levels"]) == {
        "premarketHigh",
        "premarketLow",
        "openRangeHigh",
        "openRangeLow",
        "vwap",
        "support",
        "resistance",
        "invalidation",
    }
    assert set(payload["alignment"]) == {
        "market",
        "intradayTrend",
        "vwap",
        "volume",
        "catalyst",
    }
    assert payload["provider"] == "yfinance"
    assert payload["isRealtime"] is False
