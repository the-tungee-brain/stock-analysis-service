from app.broker.portfolio_diversification import (
    build_portfolio_allocation_precomputed,
    format_diversification_summary_block,
)
from app.models.intelligence_models import SectorWeight
from app.models.strategy_models import (
    InvestmentStrategy,
    UserInvestmentProfile,
    WheelStrategyConfig,
)
from tests.test_position_prompt_metrics import _make_account, _make_position


def _account_with_cash(*, liquidation: float, cash: float):
    account = _make_account(liquidation_value=liquidation)
    current = account.securitiesAccount.currentBalances.model_copy(
        update={"cashBalance": cash}
    )
    securities = account.securitiesAccount.model_copy(
        update={"currentBalances": current}
    )
    return account.model_copy(update={"securitiesAccount": securities})


def test_build_portfolio_allocation_precomputed_returns_structured_plan():
    account = _account_with_cash(liquidation=100_000, cash=15_000)
    positions = [
        _make_position(symbol="NVDA", market_value=35_000),
        _make_position(symbol="AAPL", market_value=25_000),
    ]

    precomputed = build_portfolio_allocation_precomputed(
        positions=positions,
        account=account,
    )

    assert precomputed is not None
    assert precomputed.cash_map.deployable_cash >= 0
    assert len(precomputed.holdings) >= 2
    assert precomputed.holdings[0].symbol == "NVDA"
    assert precomputed.trim_plan
    assert precomputed.cash_map.total_to_redeploy >= precomputed.cash_map.deployable_cash


def test_diversification_summary_includes_cash_map_and_holding_review():
    account = _account_with_cash(liquidation=100_000, cash=15_000)
    positions = [
        _make_position(symbol="NVDA", market_value=35_000),
        _make_position(symbol="AAPL", market_value=25_000),
        _make_position(symbol="MSFT", market_value=15_000),
    ]

    block = format_diversification_summary_block(
        positions=positions,
        account=account,
    )

    assert block is not None
    assert "Portfolio cash map (precomputed" in block
    assert "Holding-by-holding review (precomputed" in block
    assert "Suggested trim plan (precomputed" in block
    assert "NVDA:" in block
    assert "trim ~$" in block
    assert "Total cash available to invest" in block


def test_diversification_summary_includes_top_holdings_and_cash():
    account = _account_with_cash(liquidation=100_000, cash=15_000)
    positions = [
        _make_position(symbol="NVDA", market_value=35_000),
        _make_position(symbol="AAPL", market_value=25_000),
        _make_position(symbol="MSFT", market_value=15_000),
    ]

    block = format_diversification_summary_block(
        positions=positions,
        account=account,
    )

    assert block is not None
    assert "Key numbers" in block
    assert "NVDA" in block
    assert "Cash you can invest today" in block
    assert "Too large — over 30% of portfolio" in block


def test_diversification_summary_respects_profile_single_name_limit():
    account = _account_with_cash(liquidation=100_000, cash=5_000)
    positions = [_make_position(symbol="AAPL", market_value=18_000)]
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        wheel=WheelStrategyConfig(max_single_name_pct=15.0),
    )

    block = format_diversification_summary_block(
        positions=positions,
        account=account,
        profile=profile,
    )

    assert block is not None
    assert "Max per-stock limit (from profile): 15%" in block
    assert "Large — 15–20% of portfolio" in block


def test_diversification_summary_includes_sector_weights():
    account = _make_account(liquidation_value=100_000)
    positions = [_make_position(symbol="NVDA", market_value=40_000)]

    block = format_diversification_summary_block(
        positions=positions,
        account=account,
        sector_weights=[
            SectorWeight(
                sector="Technology",
                weight_pct=45.0,
                symbols=["NVDA", "AAPL"],
            )
        ],
    )

    assert block is not None
    assert "Sector weights" in block
    assert "sector is over 30% of portfolio" in block


def test_diversification_summary_includes_etf_core_gap():
    from app.models.strategy_models import EtfCoreStrategyConfig

    account = _account_with_cash(liquidation=100_000, cash=10_000)
    positions = [
        _make_position(symbol="HOOD", market_value=25_000),
        _make_position(symbol="SCHD", market_value=100),
    ]
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.ETF_CORE,
        etf_core=EtfCoreStrategyConfig(
            target_allocation={"SCHD": 70.0, "BND": 30.0},
        ),
    )

    block = format_diversification_summary_block(
        positions=positions,
        account=account,
        profile=profile,
        etf_fund_metrics={
            "SCHD": {"dividend_yield": "3.50%", "expense_ratio": "0.06%"},
            "BND": {"dividend_yield": "4.20%", "expense_ratio": "0.03%"},
        },
    )

    assert block is not None
    assert "ETF core allocation gap" in block
    assert "ETF core weight in portfolio" in block
    assert "Total in ETF core targets:" in block
    assert "ETF fund metrics (yield and fees)" in block
    assert "dividend yield 3.50%" in block
    assert "expense ratio 0.06%" in block
    assert "SCHD:" in block and "70% target" in block
    assert "BND:" in block and "30% target" in block
    assert "Deployable cash" in block or "Cash you can invest today" in block
    assert "Suggested deploy plan (precomputed" in block


def test_build_portfolio_allocation_includes_csp_in_holding_spending():
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

    account = _account_with_cash(liquidation=100_000, cash=70_000)
    positions = [
        _make_position(symbol="NVDA", market_value=855),
        put_for("NVDA", short_qty=2, strike=170),
        _make_position(symbol="TSM", market_value=2_083),
    ]

    precomputed = build_portfolio_allocation_precomputed(
        positions=positions,
        account=account,
    )

    assert precomputed is not None
    nvda = next(item for item in precomputed.holdings if item.symbol == "NVDA")
    assert nvda.csp_reserved_cash == 34_000.0
    assert nvda.portfolio_spending == 34_855.0
    assert nvda.spending_weight_pct > nvda.weight_pct
    assert "cash-secured puts" in nvda.action_summary.lower()
    assert "portfolio spending" in nvda.action_summary.lower()


def test_diversification_summary_ignores_stale_etf_core_for_wheel_strategy():
    from app.models.strategy_models import EtfCoreStrategyConfig

    account = _account_with_cash(liquidation=100_000, cash=10_000)
    positions = [
        _make_position(symbol="NVDA", market_value=25_000),
        _make_position(symbol="SCHD", market_value=100),
    ]
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        wheel=WheelStrategyConfig(wheel_symbols=["NVDA", "TSM"]),
        etf_core=EtfCoreStrategyConfig(
            target_allocation={"SCHD": 70.0, "BND": 30.0},
        ),
    )

    block = format_diversification_summary_block(
        positions=positions,
        account=account,
        profile=profile,
        etf_fund_metrics={
            "SCHD": {"dividend_yield": "3.50%", "expense_ratio": "0.06%"},
            "BND": {"dividend_yield": "4.20%", "expense_ratio": "0.03%"},
        },
    )

    assert block is not None
    assert "ETF core allocation gap" not in block
    assert "ETF core weight in portfolio" not in block
    assert "Suggested deploy plan (precomputed" not in block
