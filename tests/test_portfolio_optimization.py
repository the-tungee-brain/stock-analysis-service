from datetime import date
from datetime import datetime
from datetime import timezone

import pytest

from app.api.get_portfolio_optimization_route import get_portfolio_optimization
from app.models.portfolio_memory_models import PortfolioSnapshotRecord, SnapshotPosition
from app.models.provider_symbol_profile_models import ProviderSymbolProfileMetadata
from app.models.schwab_models import Position
from app.services.portfolio_optimization_metadata_service import (
    PortfolioOptimizationMetadata,
    PortfolioOptimizationMetadataService,
)
from app.services.portfolio_optimization_service import PortfolioOptimizationService
from tests.test_position_prompt_metrics import (
    _make_account,
    _make_instrument,
    _make_position,
)


def _account_with_cash(*, liquidation: float, cash: float):
    account = _make_account(liquidation_value=liquidation)
    current = account.securitiesAccount.currentBalances.model_copy(
        update={"cashBalance": cash}
    )
    securities = account.securitiesAccount.model_copy(
        update={"currentBalances": current}
    )
    return account.model_copy(update={"securitiesAccount": securities})


class FakeProfileMetadataAdapter:
    def __init__(self, rows: list[ProviderSymbolProfileMetadata]):
        self.rows = rows
        self.requested_symbols: list[str] | None = None

    def list_metadata(self, symbols: list[str]):
        self.requested_symbols = symbols
        return self.rows


def _profile_row(
    *,
    symbol: str,
    provider: str = "yahoo",
    status: str = "available",
    sector: str | None = None,
    industry: str | None = None,
    asset_type: str | None = None,
    quote_type: str | None = None,
) -> ProviderSymbolProfileMetadata:
    return ProviderSymbolProfileMetadata(
        provider=provider,
        symbol=symbol,
        status=status,
        fetched_at=datetime(2026, 6, 6, tzinfo=timezone.utc),
        sector=sector,
        industry=industry,
        asset_type=asset_type,
        quote_type=quote_type,
    )


def test_portfolio_optimization_metadata_resolves_cached_profile_rows():
    adapter = FakeProfileMetadataAdapter(
        [
            _profile_row(
                symbol="NVDA",
                sector="Technology",
                industry="Semiconductors",
                asset_type="Equity",
                quote_type="EQUITY",
            ),
            _profile_row(
                symbol="SCHD",
                asset_type="ETF",
                quote_type="ETF",
            ),
        ]
    )

    metadata = PortfolioOptimizationMetadataService(adapter).resolve(
        ["NVDA", "SCHD", "SPYM"]
    )

    assert adapter.requested_symbols == ["NVDA", "SCHD", "SPYM"]
    assert metadata.sector_by_symbol["NVDA"] == "Technology"
    assert metadata.industry_by_symbol["NVDA"] == "Semiconductors"
    assert metadata.asset_type_by_symbol["SCHD"] == "ETF"
    assert metadata.missing_profile_symbols == ["SPYM"]
    assert metadata.missing_sector_symbols == ["SCHD", "SPYM"]
    assert metadata.missing_asset_type_symbols == ["SPYM"]


def test_portfolio_optimization_scores_concentrated_portfolio():
    account = _account_with_cash(liquidation=100_000, cash=5_000)
    positions = [
        _make_position(symbol="NVDA", market_value=80_000),
        _make_position(symbol="AAPL", market_value=10_000),
        _make_position(symbol="MSFT", market_value=5_000),
    ]

    response = PortfolioOptimizationService().build_portfolio_optimization(
        positions=positions,
        account=account,
    )

    assert response.diversification_score < 55
    assert response.stock_weights[0].symbol == "NVDA"
    assert response.stock_weights[0].portfolio_weight_pct == 80.0
    assert response.stock_weights[0].invested_weight_pct == pytest.approx(84.21, rel=1e-3)
    assert response.stock_weights[0].weight_pct == 80.0
    assert response.stock_weights[0].level == "critical"
    assert response.ranked_suggestions[0].category == "stockConcentration"
    assert "NVDA" in response.ranked_suggestions[0].title


def test_portfolio_optimization_scores_diversified_portfolio():
    account = _account_with_cash(liquidation=100_000, cash=50_000)
    positions = [
        _make_position(symbol=f"S{i}", market_value=5_000)
        for i in range(1, 11)
    ]

    response = PortfolioOptimizationService().build_portfolio_optimization(
        positions=positions,
        account=account,
    )

    assert response.diversification_score >= 55
    assert response.breakdown.stock_concentration.status in {"good", "strong"}
    assert response.stock_weights[0].portfolio_weight_pct == 5.0


def test_portfolio_optimization_local_sector_fallback_is_metadata_limited():
    account = _account_with_cash(liquidation=100_000, cash=10_000)
    spy = _make_position(symbol="SPY", market_value=30_000)
    spy = Position(
        **{
            **spy.model_dump(),
            "instrument": _make_instrument(symbol="SPY", asset_type="ETF"),
        }
    )
    positions = [
        _make_position(symbol="AAPL", market_value=50_000),
        spy,
    ]

    response = PortfolioOptimizationService().build_portfolio_optimization(
        positions=positions,
        account=account,
    )

    by_sector = {item.sector: item.weight_pct for item in response.sector_weights}
    assert by_sector["Unknown"] == 50.0
    assert by_sector["ETF"] == 30.0
    assert "Sector metadata unavailable; showing limited fallback." in response.data_gaps


def test_portfolio_optimization_sector_weights_use_market_values():
    account = _account_with_cash(liquidation=100_000, cash=20_000)
    spy = _make_position(symbol="SPY", market_value=20_000)
    spy = Position(
        **{
            **spy.model_dump(),
            "instrument": _make_instrument(symbol="SPY", asset_type="ETF"),
        }
    )
    positions = [
        _make_position(symbol="AAPL", market_value=60_000),
        spy,
    ]

    response = PortfolioOptimizationService().build_portfolio_optimization(
        positions=positions,
        account=account,
    )

    by_sector = {item.sector: item.weight_pct for item in response.sector_weights}
    assert by_sector["Unknown"] == 60.0
    assert by_sector["ETF"] == 20.0


def test_portfolio_optimization_csp_reserve_does_not_inflate_weights():
    from tests.test_option_utils import _make_option_position

    account = _account_with_cash(liquidation=100_000, cash=80_000)
    put = _make_option_position(
        symbol="NVDA_061726P170",
        strike_price=170,
        short_qty=2,
    )
    put = put.model_copy(
        update={
            "instrument": put.instrument.model_copy(
                update={"underlyingSymbol": "NVDA"}
            )
        }
    )
    positions = [
        _make_position(symbol="NVDA", market_value=10_000),
        put,
    ]

    response = PortfolioOptimizationService().build_portfolio_optimization(
        positions=positions,
        account=account,
    )

    nvda = next(item for item in response.stock_weights if item.symbol == "NVDA")
    assert nvda.market_value == 10_250.0
    assert nvda.portfolio_weight_pct == 10.25
    assert nvda.portfolio_weight_pct < 45.0
    assert response.sector_weights[0].weight_pct == 10.25


def test_portfolio_optimization_uses_cached_profile_sectors():
    account = _account_with_cash(liquidation=100_000, cash=12_000)
    positions = [
        _make_position(symbol="NVDA", market_value=40_000),
        _make_position(symbol="AMZN", market_value=20_000),
        _make_position(symbol="GOOG", market_value=10_000),
        _make_position(symbol="TSM", market_value=8_000),
        _make_position(symbol="SMCI", market_value=7_000),
        _make_position(symbol="NOK", market_value=3_000),
    ]
    metadata = PortfolioOptimizationMetadata(
        sector_by_symbol={
            "NVDA": "Technology",
            "AMZN": "Consumer Cyclical",
            "GOOG": "Communication Services",
            "TSM": "Technology",
            "SMCI": "Technology",
            "NOK": "Technology",
        },
        asset_type_by_symbol={
            "NVDA": "Equity",
            "AMZN": "Equity",
            "GOOG": "Equity",
            "TSM": "Equity",
            "SMCI": "Equity",
            "NOK": "Equity",
        },
    )

    response = PortfolioOptimizationService().build_portfolio_optimization(
        positions=positions,
        account=account,
        metadata=metadata,
    )

    by_sector = {item.sector: item.weight_pct for item in response.sector_weights}
    assert by_sector["Technology"] == 58.0
    assert by_sector["Consumer Cyclical"] == 20.0
    assert by_sector["Communication Services"] == 10.0
    assert "Unknown" not in by_sector
    assert response.data_gaps == []


def test_portfolio_optimization_resolves_etf_from_cached_asset_type():
    account = _account_with_cash(liquidation=100_000, cash=60_000)
    positions = [
        _make_position(symbol="SCHD", market_value=20_000),
        _make_position(symbol="SPYM", market_value=10_000),
    ]
    metadata = PortfolioOptimizationMetadata(
        sector_by_symbol={},
        asset_type_by_symbol={"SCHD": "ETF", "SPYM": "Equity"},
        quote_type_by_symbol={"SCHD": "ETF", "SPYM": "EQUITY"},
        missing_sector_symbols=["SCHD", "SPYM"],
    )

    response = PortfolioOptimizationService().build_portfolio_optimization(
        positions=positions,
        account=account,
        metadata=metadata,
    )

    by_sector = {item.sector: item.weight_pct for item in response.sector_weights}
    assert by_sector["ETF"] == 20.0
    assert by_sector["Unknown"] == 10.0
    assert "Missing sector metadata for SCHD, SPYM." in response.data_gaps


def test_portfolio_optimization_missing_sector_uses_unknown_with_data_gap():
    account = _account_with_cash(liquidation=100_000, cash=80_000)
    positions = [_make_position(symbol="SPYM", market_value=10_000)]
    metadata = PortfolioOptimizationMetadata(
        asset_type_by_symbol={"SPYM": "Equity"},
        quote_type_by_symbol={"SPYM": "EQUITY"},
        missing_sector_symbols=["SPYM"],
    )

    response = PortfolioOptimizationService().build_portfolio_optimization(
        positions=positions,
        account=account,
        metadata=metadata,
    )

    assert response.sector_weights[0].sector == "Unknown"
    assert response.sector_weights[0].weight_pct == 10.0
    assert "Missing sector metadata for SPYM." in response.data_gaps


def test_portfolio_optimization_excludes_alert_categories():
    account = _account_with_cash(liquidation=100_000, cash=10_000)
    positions = [
        _make_position(symbol="NVDA", market_value=70_000),
        _make_position(symbol="AAPL", market_value=10_000),
    ]

    response = PortfolioOptimizationService().build_portfolio_optimization(
        positions=positions,
        account=account,
    )

    text = " ".join(
        [
            *(item.category for item in response.ranked_suggestions),
            *(item.title for item in response.ranked_suggestions),
            *(item.why for item in response.ranked_suggestions),
        ]
    ).lower()
    assert "tax" not in text
    assert "wash" not in text
    assert "earnings" not in text
    assert "exit" not in text


def test_portfolio_optimization_response_uses_camel_case_aliases():
    account = _account_with_cash(liquidation=100_000, cash=10_000)
    positions = [_make_position(symbol="NVDA", market_value=50_000)]

    response = PortfolioOptimizationService().build_portfolio_optimization(
        positions=positions,
        account=account,
    )
    payload = response.model_dump(by_alias=True)

    assert "diversificationScore" in payload
    assert "stockWeights" in payload
    assert "portfolioWeightPct" in payload["stockWeights"][0]
    assert "investedWeightPct" in payload["stockWeights"][0]
    assert "sectorWeights" in payload
    assert "topDrivers" in payload
    assert "rankedSuggestions" in payload
    assert "estimatedScoreImprovement" in payload["rankedSuggestions"][0]
    assert "dataGaps" in payload


def test_portfolio_optimization_empty_portfolio_returns_unavailable():
    account = _account_with_cash(liquidation=100_000, cash=100_000)

    response = PortfolioOptimizationService().build_portfolio_optimization(
        positions=[],
        account=account,
    )

    assert response.stock_weights == []
    assert response.sector_weights == []
    assert response.data_gaps


@pytest.mark.asyncio
async def test_portfolio_optimization_route_uses_cached_snapshot_only():
    class ExplodingPortfolioService:
        def get_enriched_account(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("external account fetch should not be called")

    class FakeSnapshotAdapter:
        def list_recent(self, user_id, *, limit):
            assert user_id == "user-1"
            assert limit == 1
            return [
                PortfolioSnapshotRecord(
                    user_id="user-1",
                    snapshot_date=date.today(),
                    liquidation_value=100_000,
                    cash_balance=20_000,
                    positions=[
                        SnapshotPosition(
                            symbol="NVDA",
                            asset_type="EQUITY",
                            quantity=10,
                            market_value=40_000,
                            weight_pct=40,
                            pnl=0,
                        )
                    ],
                )
            ]

    class FakeMemoryService:
        portfolio_snapshot_adapter = FakeSnapshotAdapter()

    class FakeMetadataService:
        def resolve(self, symbols):
            assert symbols == ["NVDA"]
            return PortfolioOptimizationMetadata(
                sector_by_symbol={"NVDA": "Technology"},
                asset_type_by_symbol={"NVDA": "Equity"},
            )

    response = await get_portfolio_optimization(
        user_id="user-1",
        portfolio_memory_service=FakeMemoryService(),
        metadata_service=FakeMetadataService(),
    )

    assert response.stock_weights[0].symbol == "NVDA"
    assert response.stock_weights[0].portfolio_weight_pct == 40.0
    assert response.sector_weights[0].sector == "Technology"
