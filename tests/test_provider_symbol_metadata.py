from __future__ import annotations

from unittest.mock import MagicMock, patch

from ranking_pipeline.providers.symbol_metadata import (
    ProviderFirstSymbolMetadataResolver,
    SymbolMetadata,
)


class _Store:
    def __init__(self, rows: dict[str, SymbolMetadata]) -> None:
        self.rows = rows
        self.calls: list[list[str]] = []

    def get_many(self, symbols: list[str]) -> dict[str, SymbolMetadata]:
        self.calls.append(symbols)
        return {symbol: self.rows[symbol] for symbol in symbols if symbol in self.rows}


class _Writer:
    def __init__(self) -> None:
        self.upserts: list[tuple[str, str, dict]] = []

    def upsert_success(self, provider: str, symbol: str, info: dict, *, fetched_at=None) -> None:
        self.upserts.append((provider, symbol, dict(info)))


def test_provider_table_hit_does_not_call_yfinance():
    resolver = ProviderFirstSymbolMetadataResolver(
        _Store(
            {
                "AAPL": SymbolMetadata(
                    symbol="AAPL",
                    market_cap=3_000_000_000_000,
                    source="oracle:yahoo",
                )
            }
        ),
        allow_yfinance_fallback=True,
    )

    with patch("ranking_pipeline.providers.symbol_metadata.yf.Ticker") as ticker_cls:
        result = resolver.resolve_many(["AAPL"], required_fields=("market_cap",))

    assert result["AAPL"].market_cap == 3_000_000_000_000
    assert result["AAPL"].source == "oracle:yahoo"
    ticker_cls.assert_not_called()


def test_missing_required_field_triggers_controlled_fallback():
    writer = _Writer()
    resolver = ProviderFirstSymbolMetadataResolver(
        _Store({"AAPL": SymbolMetadata(symbol="AAPL", market_cap=None)}),
        profile_writer=writer,
        allow_yfinance_fallback=True,
    )
    ticker = MagicMock()
    ticker.info = {
        "marketCap": 3_000_000_000_000,
        "longName": "Apple Inc.",
        "exchange": "NMS",
        "sector": "Technology",
    }

    with patch("ranking_pipeline.providers.symbol_metadata.yf.Ticker", return_value=ticker):
        result = resolver.resolve_many(["AAPL"], required_fields=("market_cap",))

    assert result["AAPL"].market_cap == 3_000_000_000_000
    assert result["AAPL"].source == "yfinance:fallback"
    assert writer.upserts == [("yahoo", "AAPL", ticker.info)]


def test_fallback_can_be_disabled():
    resolver = ProviderFirstSymbolMetadataResolver(
        _Store({}),
        allow_yfinance_fallback=False,
    )

    with patch("ranking_pipeline.providers.symbol_metadata.yf.Ticker") as ticker_cls:
        result = resolver.resolve_many(["AAPL"], required_fields=("market_cap",))

    assert result["AAPL"].market_cap is None
    ticker_cls.assert_not_called()


def test_fallback_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("RANKING_YFINANCE_METADATA_FALLBACK", "false")
    resolver = ProviderFirstSymbolMetadataResolver(_Store({}))

    with patch("ranking_pipeline.providers.symbol_metadata.yf.Ticker") as ticker_cls:
        result = resolver.resolve_many(["AAPL"], required_fields=("market_cap",))

    assert result["AAPL"].market_cap is None
    ticker_cls.assert_not_called()
