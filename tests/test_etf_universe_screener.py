from app.models.screener_preset_models import ScreenerPreset
from app.screener.etf_universe_screener import _passes_etf_filters


def _etf_preset(**post_filters) -> ScreenerPreset:
    return ScreenerPreset(
        id="test_etf",
        label="Test ETF",
        description="Test",
        fund_universe={"region": "us", "asset_class": "equity"},
        post_filters=post_filters,
    )


def test_etf_price_filter_rejects_below_min():
    preset = _etf_preset(price={"min_price": 25, "max_price": 300})
    info = {"regularMarketPrice": 20.0, "totalAssets": 1_000_000_000}
    assert not _passes_etf_filters(
        info,
        structure={},
        liquidity={},
        price_cfg=preset.post_filters["price"],
        dividend_cfg={},
    )


def test_etf_price_filter_rejects_above_max():
    preset = _etf_preset(price={"min_price": 25, "max_price": 300})
    info = {"regularMarketPrice": 350.0, "totalAssets": 1_000_000_000}
    assert not _passes_etf_filters(
        info,
        structure={},
        liquidity={},
        price_cfg=preset.post_filters["price"],
        dividend_cfg={},
    )


def test_etf_liquidity_min_assets_from_liquidity_block():
    preset = _etf_preset(
        structure={},
        liquidity={"min_total_assets": 500_000_000},
        price={},
    )
    info = {"regularMarketPrice": 100.0, "totalAssets": 100_000_000}
    assert not _passes_etf_filters(
        info,
        structure=preset.post_filters.get("structure") or {},
        liquidity=preset.post_filters["liquidity"],
        price_cfg={},
        dividend_cfg={},
    )
