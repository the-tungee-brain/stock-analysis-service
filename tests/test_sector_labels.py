from unittest.mock import MagicMock

from app.broker.sector_labels import MISC_SECTOR_LABEL, normalize_sector_label
from app.services.intelligence.portfolio_intelligence_service import (
    PortfolioIntelligenceService,
)
from tests.test_position_prompt_metrics import _make_account, _make_position


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
    )

    assert len(weights) == 1
    assert weights[0].sector == MISC_SECTOR_LABEL
    assert weights[0].weight_pct == 100.0
