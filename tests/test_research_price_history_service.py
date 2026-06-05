from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from app.builders.performance_builder import PerformanceBuilder
from app.services.market_service import MarketService
from app.services.research_price_history_service import ResearchPriceHistoryService
from data.store import save_raw


def _ohlcv(rows: int = 260) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=rows)
    close = pd.Series(range(100, 100 + rows), index=dates, dtype=float)
    return pd.DataFrame(
        {
            "open": close - 1,
            "high": close + 1,
            "low": close - 2,
            "close": close,
            "volume": 1_000_000,
        },
        index=dates,
    )


def test_local_ohlcv_path_avoids_yahoo_history(monkeypatch, tmp_path):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path)
    save_raw(_ohlcv(), "AAPL")
    yahoo = MagicMock()
    service = ResearchPriceHistoryService(yahoo_fallback=yahoo)

    closes = service.get_daily_closes_1y("AAPL")

    assert not closes.empty
    assert float(closes.iloc[-1]) == 359.0
    yahoo.get_daily_closes_1y.assert_not_called()


def test_missing_local_ohlcv_falls_back_to_yahoo_once(monkeypatch, tmp_path):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path)
    yahoo = MagicMock()
    yahoo.get_daily_closes_1y.return_value = pd.Series(
        [10.0, 11.0],
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )
    service = ResearchPriceHistoryService(yahoo_fallback=yahoo)

    closes = service.get_daily_closes_1y("MSFT")

    assert closes.tolist() == [10.0, 11.0]
    yahoo.get_daily_closes_1y.assert_called_once_with(symbol="MSFT")


def test_insufficient_local_ohlcv_falls_back_cleanly(monkeypatch, tmp_path):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path)
    save_raw(_ohlcv(rows=1), "TSLA")
    yahoo = MagicMock()
    yahoo.get_daily_closes_1y.return_value = pd.Series(
        [20.0, 22.0],
        index=pd.to_datetime(["2026-01-02", "2026-01-03"]),
    )
    service = ResearchPriceHistoryService(yahoo_fallback=yahoo)

    closes = service.get_daily_closes_1y("TSLA")

    assert closes.tolist() == [20.0, 22.0]
    yahoo.get_daily_closes_1y.assert_called_once_with(symbol="TSLA")


def test_performance_response_shape_unchanged_with_local_ohlcv(monkeypatch, tmp_path):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path)
    save_raw(_ohlcv(), "NVDA")
    yahoo = MagicMock()
    price_history = ResearchPriceHistoryService(yahoo_fallback=yahoo)
    builder = PerformanceBuilder(price_history_service=price_history)

    snapshot = builder.build("NVDA")

    assert set(snapshot.model_dump(by_alias=True)) == {
        "oneMonth",
        "threeMonth",
        "oneYear",
        "trendLabel",
        "volatilityNote",
    }
    yahoo.get_daily_closes_1y.assert_not_called()


def test_research_performance_still_works_without_local_data(monkeypatch, tmp_path):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path)
    yahoo = MagicMock()
    yahoo.get_daily_closes_1y.return_value = pd.Series(
        [100.0, 120.0],
        index=pd.to_datetime(["2025-01-02", "2026-01-02"]),
    )
    price_history = ResearchPriceHistoryService(yahoo_fallback=yahoo)
    market_service = MarketService(
        schwab_market_builder=MagicMock(),
        performance_builder=PerformanceBuilder(price_history_service=price_history),
    )

    snapshot = market_service.get_performance("SPY")

    assert snapshot.oneYear == "+20.0%"
    yahoo.get_daily_closes_1y.assert_called_once_with(symbol="SPY")
