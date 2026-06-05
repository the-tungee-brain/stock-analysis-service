from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.auth.dependencies import get_current_user
from app.dependencies.service_dependencies import get_research_symbol_data_service
from app.main import app
from app.models.company_research_models import ResearchSnapshot
from app.services.research_symbol_data_service import ResearchSymbolDataService


def _service(
    *,
    asset_type_service: MagicMock | None = None,
    yfinance_adapter: MagicMock | None = None,
    company_profile_service: MagicMock | None = None,
) -> ResearchSymbolDataService:
    return ResearchSymbolDataService(
        asset_type_service=asset_type_service or MagicMock(),
        yfinance_adapter=yfinance_adapter or MagicMock(),
        company_profile_service=company_profile_service or MagicMock(),
    )


def _snapshot(symbol: str = "AAPL") -> ResearchSnapshot:
    return ResearchSnapshot(
        symbol=symbol,
        name="Apple Inc.",
        sector="Technology",
        country="United States",
        price=200.0,
        changePct=2.56,
        marketCap="3.0T",
        range52w="$170.00 - $220.00",
        weburl="https://www.apple.com",
        logo="https://static2.finnhub.io/file/publicdatany/finnhubimage/stock_logo/AAPL.png",
        dividendYieldPct=0.44,
        peRatio=28.5,
        volume=52_000_000,
        avgVolume=48_000_000,
        expenseRatioPct=None,
    )


def test_research_symbol_data_service_normalizes_symbols():
    service = _service()

    assert service.normalize_symbol(" aapl ") == "AAPL"


def test_get_asset_type_delegates_to_asset_type_service():
    asset_type_service = MagicMock()
    asset_type_service.resolve.return_value = "STOCK"
    service = _service(asset_type_service=asset_type_service)

    assert service.get_asset_type(" aapl ") == "STOCK"
    asset_type_service.resolve.assert_called_once_with("AAPL")


def test_get_profile_info_uses_yfinance_ticker_info():
    yfinance_adapter = MagicMock()
    yfinance_adapter.get_ticker_info.return_value = {"longName": "Apple Inc."}
    service = _service(yfinance_adapter=yfinance_adapter)

    assert service.get_profile_info(" aapl ") == {"longName": "Apple Inc."}
    yfinance_adapter.get_ticker_info.assert_called_once_with("AAPL")


def test_get_snapshot_returns_company_profile_service_snapshot_object():
    expected = _snapshot()
    company_profile_service = MagicMock()
    company_profile_service.get_snapshot.return_value = expected
    service = _service(company_profile_service=company_profile_service)

    assert service.get_snapshot(" aapl ") is expected
    company_profile_service.get_snapshot.assert_called_once_with(symbol="AAPL")


def test_research_snapshot_route_json_response_is_unchanged():
    expected = _snapshot()
    research_symbol_data_service = MagicMock()
    research_symbol_data_service.get_snapshot.return_value = expected

    class _FakeUser:
        identity_sub = "user-1"

    async def _user() -> _FakeUser:
        return _FakeUser()

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[get_research_symbol_data_service] = (
        lambda: research_symbol_data_service
    )

    client = TestClient(app)
    try:
        response = client.get("/api/v1/research/snapshot?symbol=aapl")
        assert response.status_code == 200
        assert response.json() == expected.model_dump(mode="json")
        research_symbol_data_service.get_snapshot.assert_called_once_with(
            symbol="aapl"
        )
    finally:
        app.dependency_overrides.clear()


def test_research_symbol_data_service_does_not_cache_provider_or_user_data():
    service = _service()

    assert not hasattr(service, "_cache")
    assert not hasattr(service, "cache")
    assert not hasattr(service, "user_id")
    assert not hasattr(service, "portfolio")
