from unittest.mock import patch

from app.models.strategy_models import (
    InvestmentStrategy,
    WheelStrategyConfig,
    UserInvestmentProfile,
)
from app.screener.equity_query_compiler import compile_equity_query
from app.screener.preset_registry import (
    STRATEGY_COMPANION_PRESET_IDS,
    get_preset,
    preset_for_strategy,
)
from app.services.strategy.strategy_stock_screener_service import (
    StrategyStockScreenerService,
)


def _wheel_profile(*, risk: str = "moderate") -> UserInvestmentProfile:
    return UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        risk_tolerance=risk,
        options_experience="beginner",
        income_vs_growth="balanced",
        wheel=WheelStrategyConfig(wheel_symbols=["AAPL"]),
    )


def test_all_presets_load():
    for strategy in InvestmentStrategy:
        if not StrategyStockScreenerService.supports_stock_screener(strategy):
            continue
        preset = preset_for_strategy(strategy)
        assert preset.id
        assert preset.label
        assert preset.post_filters


def test_equity_presets_compile_to_yfinance_query():
    preset = get_preset("wheel_stock")
    assert preset is not None
    assert preset.equity_query is not None
    query = compile_equity_query(preset.equity_query)
    assert query.operator == "AND"


def test_profile_adjustments_change_wheel_preset():
    conservative = StrategyStockScreenerService.resolve_preset(
        InvestmentStrategy.WHEEL,
        _wheel_profile(risk="conservative"),
    )
    aggressive = StrategyStockScreenerService.resolve_preset(
        InvestmentStrategy.WHEEL,
        _wheel_profile(risk="aggressive"),
    )

    def market_cap_clause(preset):
        for clause in preset.equity_query.clauses:
            if clause.field == "marketCap" and clause.op == "gte":
                return clause.value
        return None

    assert market_cap_clause(conservative) == 10_000_000_000
    assert market_cap_clause(aggressive) == 2_000_000_000


def test_screen_stocks_includes_existing_symbols():
    quotes = [
        {
            "symbol": "AAPL",
            "shortName": "Apple Inc.",
            "marketCap": 3_000_000_000_000,
            "trailingPE": 28.5,
            "dividendYield": 0.004,
            "regularMarketPrice": 190.0,
        },
        {
            "symbol": "MSFT",
            "shortName": "Microsoft Corporation",
            "marketCap": 3_100_000_000_000,
            "trailingPE": 32.1,
            "dividendYield": 0.007,
            "regularMarketPrice": 420.0,
        },
    ]

    def fake_screen(*_args, **_kwargs):
        return {"total": 2, "quotes": quotes}

    service = StrategyStockScreenerService()
    with patch(
        "app.services.strategy.strategy_stock_screener_service.yf.screen",
        side_effect=fake_screen,
    ):
        result = service.screen_stocks(
            profile=_wheel_profile(),
            strategy=InvestmentStrategy.WHEEL,
            page=1,
            page_size=10,
        )

    assert result is not None
    assert [quote.symbol for quote in result.quotes] == ["AAPL", "MSFT"]
    assert result.preset.id == "wheel_stock"


def test_apply_api_overrides_adds_dividend_clause():
    from app.screener.preset_registry import get_preset
    from app.services.strategy.strategy_stock_screener_service import _apply_api_overrides

    preset = get_preset("wheel_stock")
    assert preset is not None
    updated = _apply_api_overrides(preset, {"require_dividend": True})
    fields = [clause.field for clause in updated.equity_query.clauses]
    assert "dividendYield" in fields


def test_filters_from_preset_reflects_overrides():
    from app.services.strategy.strategy_stock_screener_service import filters_from_preset

    preset = StrategyStockScreenerService.resolve_preset(
        InvestmentStrategy.WHEEL,
        _wheel_profile(),
        overrides={"min_market_cap": 50_000_000_000, "max_pe": 20.0},
    )
    filters = filters_from_preset(preset)
    assert filters is not None
    assert filters.min_market_cap == 50_000_000_000
    assert filters.max_pe == 20.0


def test_describe_preset_includes_key_constraints():
    preset = preset_for_strategy(InvestmentStrategy.WHEEL)
    summary = StrategyStockScreenerService.describe_preset(preset)
    assert "market cap" in summary.lower()
    assert "P/E" in summary


def test_etf_companion_presets_registered():
    assert STRATEGY_COMPANION_PRESET_IDS[InvestmentStrategy.WHEEL] == ["wheel_etf"]
    assert STRATEGY_COMPANION_PRESET_IDS[InvestmentStrategy.CSP_INCOME] == ["csp_etf"]
    assert STRATEGY_COMPANION_PRESET_IDS[InvestmentStrategy.COVERED_CALL] == [
        "covered_call_etf"
    ]
    assert STRATEGY_COMPANION_PRESET_IDS[InvestmentStrategy.DIVIDEND] == ["dividend_etf"]


def test_etf_companion_presets_load():
    for preset_id in (
        "wheel_etf",
        "csp_etf",
        "covered_call_etf",
        "dividend_etf",
    ):
        preset = get_preset(preset_id)
        assert preset is not None
        assert preset.equity_query is None
        assert preset.post_filters.get("structure", {}).get("examples_preferred")


def test_wheel_screen_includes_etf_companion_section():
    quotes = [
        {
            "symbol": "AAPL",
            "shortName": "Apple Inc.",
            "marketCap": 3_000_000_000_000,
            "trailingPE": 28.5,
            "dividendYield": 0.004,
            "regularMarketPrice": 190.0,
        },
    ]
    etf_quotes = [
        {
            "symbol": "SPY",
            "shortName": "SPDR S&P 500 ETF",
            "totalAssets": 500_000_000_000,
            "regularMarketPrice": 520.0,
        },
    ]

    def fake_screen(*_args, **_kwargs):
        return {"total": 1, "quotes": quotes}

    def fake_etf_screen(preset, *, limit):
        if preset.id == "wheel_etf":
            from app.screener.etf_universe_screener import _quote_from_info

            return (
                [_quote_from_info("SPY", etf_quotes[0])],
                len(preset.post_filters["structure"]["examples_preferred"]),
            )
        return [], 0

    service = StrategyStockScreenerService()
    with patch(
        "app.services.strategy.strategy_stock_screener_service.yf.screen",
        side_effect=fake_screen,
    ), patch(
        "app.services.strategy.strategy_stock_screener_service.screen_etf_preset",
        side_effect=fake_etf_screen,
    ):
        result = service.screen_stocks(
            profile=_wheel_profile(),
            strategy=InvestmentStrategy.WHEEL,
            page=1,
            page_size=10,
        )

    assert result is not None
    assert len(result.sections) == 1
    assert result.sections[0].preset.id == "wheel_etf"
    assert [quote.symbol for quote in result.sections[0].quotes] == ["SPY"]
