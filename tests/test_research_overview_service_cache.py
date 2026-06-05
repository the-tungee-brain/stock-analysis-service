from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.adapters.cache.research_overview_symbol_cache import (
    ResearchOverviewSymbolCache,
)
from app.models.company_research_models import PerformanceSnapshot, ResearchSnapshot
from app.models.intelligence_models import SymbolIntelligence
from app.models.yfinance_analysis_models import StreetAnalysisSnapshot
from app.services import research_overview_service as service_module
from app.services.research_overview_service import ResearchOverviewService


class _DictSymbolCache:
    def __init__(self, initial: dict | None = None) -> None:
        self.store: dict[str, dict] = dict(initial or {})

    def get(self, symbol: str) -> dict | None:
        return self.store.get(symbol.strip().upper())

    def put(self, symbol: str, payload: dict) -> None:
        self.store[symbol.strip().upper()] = payload


class _FailingRedis:
    def get(self, key: str):
        raise RuntimeError("redis unavailable")

    def setex(self, key: str, ttl: int, value: str) -> None:
        raise RuntimeError("redis unavailable")


def _snapshot(price: float = 200.0) -> ResearchSnapshot:
    return ResearchSnapshot(
        symbol="AAPL",
        name="Apple Inc.",
        sector="Technology",
        country="US",
        price=price,
        changePct=0.5,
        marketCap="3.0T",
        weburl="https://apple.com",
        logo="https://example.com/logo.png",
    )


def _performance() -> PerformanceSnapshot:
    return PerformanceSnapshot(
        oneMonth="+2%",
        threeMonth="+5%",
        oneYear="+10%",
        trendLabel="Up",
        volatilityNote="Moderate",
    )


def _context():
    return SimpleNamespace(
        symbol="AAPL",
        snapshot=_snapshot(),
        performance=_performance(),
        asset_type="STOCK",
        etf_holdings=None,
    )


def _service(symbol_cache=None) -> ResearchOverviewService:
    company_research_service = MagicMock()
    company_research_service.build_context.return_value = _context()

    company_profile_service = MagicMock()
    company_profile_service.get_snapshot.return_value = _snapshot()

    market_service = MagicMock()
    market_service.get_performance.return_value = _performance()

    ticker_service = MagicMock()
    ticker_service.get_by_symbol.return_value = SimpleNamespace(asset_type="STOCK")

    yfinance_analysis_builder = MagicMock()
    yfinance_analysis_builder.build.return_value = StreetAnalysisSnapshot(
        consensus_label="Buy"
    )

    return ResearchOverviewService(
        company_research_service=company_research_service,
        company_profile_service=company_profile_service,
        market_service=market_service,
        ticker_service=ticker_service,
        portfolio_service=MagicMock(),
        schwab_auth_service=MagicMock(),
        portfolio_analysis_service=MagicMock(),
        yfinance_analysis_builder=yfinance_analysis_builder,
        yfinance_funds_builder=MagicMock(),
        etf_research_service=MagicMock(),
        symbol_cache=symbol_cache,
    )


@pytest.fixture(autouse=True)
def fake_symbol_intelligence(monkeypatch):
    calls: list[str] = []

    def _fake_fetch_symbol_intelligence(**kwargs):
        calls.append(kwargs["user_id"])
        return SymbolIntelligence(symbol=kwargs["symbol_upper"])

    monkeypatch.setattr(
        service_module,
        "fetch_symbol_intelligence",
        _fake_fetch_symbol_intelligence,
    )
    return calls


def test_symbol_cache_hit_avoids_rebuilding_symbol_only_sections(
    fake_symbol_intelligence,
):
    cached = {
        "AAPL": {
            "symbol": "AAPL",
            "asset_type": "STOCK",
            "snapshot": _snapshot(price=201.0).model_dump(
                mode="json",
            ),
            "performance": _performance().model_dump(mode="json"),
            "street_analysis": StreetAnalysisSnapshot(
                consensus_label="Cached Buy"
            ).model_dump(mode="json"),
        }
    }
    cache = _DictSymbolCache(cached)
    service = _service(symbol_cache=cache)
    service.company_research_service.build_context.side_effect = AssertionError(
        "context should not rebuild on cache hit"
    )
    service.company_profile_service.get_snapshot.side_effect = AssertionError(
        "snapshot should not rebuild on cache hit"
    )
    service.market_service.get_performance.side_effect = AssertionError(
        "performance should not rebuild on cache hit"
    )
    service.ticker_service.get_by_symbol.side_effect = AssertionError(
        "asset type should not rebuild on cache hit"
    )
    service.yfinance_analysis_builder.build.side_effect = AssertionError(
        "street analysis should not rebuild on cache hit"
    )

    bundle = service.build_bundle(user_id="user-1", symbol="AAPL")

    assert bundle.snapshot.price == 201.0
    assert bundle.street_analysis is not None
    assert bundle.street_analysis.consensus_label == "Cached Buy"
    assert bundle.intelligence.symbol == "AAPL"
    assert fake_symbol_intelligence == ["user-1"]


def test_shared_symbol_cache_stores_only_non_private_sections():
    cache = _DictSymbolCache()
    service = _service(symbol_cache=cache)

    bundle = service.build_bundle(user_id="user-1", symbol="AAPL")

    assert bundle.symbol == "AAPL"
    stored = cache.store["AAPL"]
    assert set(stored) == {
        "symbol",
        "asset_type",
        "snapshot",
        "performance",
        "street_analysis",
    }
    encoded = json.dumps(stored, sort_keys=True)
    assert "user-1" not in encoded
    assert "portfolio" not in encoded.lower()
    assert "account" not in encoded.lower()
    assert "position" not in encoded.lower()
    assert "summary" not in encoded.lower()
    assert "prompt" not in encoded.lower()
    assert "intelligence" not in encoded.lower()


def test_redis_failure_falls_back_to_live_overview_build():
    cache = ResearchOverviewSymbolCache(
        redis_client=_FailingRedis(),
        ttl_seconds=60,
    )
    service = _service(symbol_cache=cache)

    bundle = service.build_bundle(user_id="user-1", symbol="AAPL")

    assert bundle.symbol == "AAPL"
    assert bundle.snapshot.name == "Apple Inc."
    service.company_research_service.build_context.assert_called_once_with(
        symbol="AAPL"
    )


def test_section_latency_logs_do_not_include_private_payloads(caplog):
    service = _service(symbol_cache=_DictSymbolCache())

    with caplog.at_level("INFO", logger="app.services.research_overview_service"):
        service.build_bundle(user_id="user-1", symbol="AAPL")

    assert "section=" in caplog.text
    assert "user-1" not in caplog.text
    assert "account" not in caplog.text.lower()
    assert "position" not in caplog.text.lower()
    assert "prompt" not in caplog.text.lower()
