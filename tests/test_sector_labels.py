from unittest.mock import MagicMock

import pytest

from app.broker.sector_labels import (
    ETF_SECTOR_LABEL,
    MISC_SECTOR_LABEL,
    normalize_sector_label,
    sector_label_for_holding,
)
from app.models.schwab_models import Position
from app.services.intelligence.portfolio_intelligence_service import (
    PortfolioIntelligenceService,
)
from tests.test_position_prompt_metrics import _make_account, _make_instrument, _make_position


def test_normalize_sector_label_maps_unknown_to_misc():
    assert normalize_sector_label(None) == MISC_SECTOR_LABEL
    assert normalize_sector_label("") == MISC_SECTOR_LABEL
    assert normalize_sector_label("Unknown") == MISC_SECTOR_LABEL
    assert normalize_sector_label("unknown sector") == MISC_SECTOR_LABEL
    assert normalize_sector_label("Technology") == "Technology"


def test_sector_weights_use_misc_for_missing_sector():
    account = _make_account(liquidation_value=100_000)
    positions = [
        _make_position(symbol="AAPL", market_value=50_000),
        _make_position(symbol="MSFT", market_value=50_000),
    ]

    service = PortfolioIntelligenceService(
        peer_comparison_service=MagicMock(),
        enriched_news_service=MagicMock(),
    )
    weights = service._sector_weights(
        positions=positions,
        account=account,
        sector_by_symbol={},
        asset_type_by_symbol={},
    )

    assert len(weights) == 1
    assert weights[0].sector == MISC_SECTOR_LABEL
    assert weights[0].weight_pct == 100.0


def test_sector_label_for_holding_groups_etfs():
    assert (
        sector_label_for_holding(
            symbol="SPY",
            instrument_asset_type="EQUITY",
            sector_by_symbol={"SPY": "Technology"},
            asset_type_by_symbol={"SPY": ETF_SECTOR_LABEL},
        )
        == ETF_SECTOR_LABEL
    )
    assert (
        sector_label_for_holding(
            symbol="SPY",
            instrument_asset_type="ETF",
            sector_by_symbol={"SPY": "Technology"},
            asset_type_by_symbol={},
        )
        == ETF_SECTOR_LABEL
    )
    assert (
        sector_label_for_holding(
            symbol="AAPL",
            instrument_asset_type="EQUITY",
            sector_by_symbol={"AAPL": "Technology"},
            asset_type_by_symbol={},
        )
        == "Technology"
    )


def test_sector_weights_group_etfs_separately_from_equity_sectors():
    account = _make_account(liquidation_value=100_000)
    positions = [
        _make_position(symbol="AAPL", market_value=40_000),
        _make_position(symbol="SPY", market_value=30_000),
        _make_position(symbol="SCHD", market_value=30_000),
    ]
    positions[1] = Position(
        **{
            **positions[1].model_dump(),
            "instrument": _make_instrument(symbol="SPY", asset_type="ETF"),
        }
    )
    positions[2] = Position(
        **{
            **positions[2].model_dump(),
            "instrument": _make_instrument(symbol="SCHD", asset_type="ETF"),
        }
    )

    service = PortfolioIntelligenceService(
        peer_comparison_service=MagicMock(),
        enriched_news_service=MagicMock(),
    )
    weights = service._sector_weights(
        positions=positions,
        account=account,
        sector_by_symbol={
            "AAPL": "Technology",
            "SPY": "Technology",
            "SCHD": "Financial Services",
        },
        asset_type_by_symbol={},
    )

    by_sector = {item.sector: item.weight_pct for item in weights}
    assert by_sector["Technology"] == 40.0
    assert by_sector[ETF_SECTOR_LABEL] == 60.0


def test_sector_weights_include_csp_reserved_cash():
    from tests.test_option_utils import _make_option_position

    def put_for(underlying: str, short_qty: float, strike: float):
        position = _make_option_position(
            symbol=f"{underlying}_061726P{int(strike)}",
            strike_price=strike,
            short_qty=short_qty,
        )
        return position.model_copy(
            update={
                "instrument": position.instrument.model_copy(
                    update={"underlyingSymbol": underlying}
                )
            }
        )

    account = _make_account(liquidation_value=100_000)
    positions = [
        _make_position(symbol="NVDA", market_value=855),
        put_for("NVDA", short_qty=2, strike=170),
        _make_position(symbol="TSM", market_value=2_083),
    ]

    service = PortfolioIntelligenceService(
        peer_comparison_service=MagicMock(),
        enriched_news_service=MagicMock(),
    )
    weights = service._sector_weights(
        positions=positions,
        account=account,
        sector_by_symbol={"NVDA": "Technology", "TSM": "Technology"},
        asset_type_by_symbol={},
    )

    by_sector = {item.sector: item.weight_pct for item in weights}
    assert by_sector["Technology"] == pytest.approx(36.938, rel=1e-3)
