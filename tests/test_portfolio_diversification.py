from app.broker.portfolio_diversification import format_diversification_summary_block
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
    assert "Top 1 / 3 / 5 weights" in block
    assert "NVDA" in block
    assert "Deployable cash" in block
    assert "CRITICAL (>30%)" in block


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
    assert "Single-name target from profile: 15% max" in block
    assert "ELEVATED (15–20%)" in block


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
    assert "SECTOR CRITICAL (>30%)" in block


def test_diversification_summary_includes_etf_core_gap():
    from app.models.strategy_models import EtfCoreStrategyConfig

    account = _account_with_cash(liquidation=100_000, cash=10_000)
    positions = [
        _make_position(symbol="HOOD", market_value=25_000),
        _make_position(symbol="SCHD", market_value=100),
    ]
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        etf_core=EtfCoreStrategyConfig(
            target_allocation={"SCHD": 70.0, "BND": 30.0},
        ),
    )

    block = format_diversification_summary_block(
        positions=positions,
        account=account,
        profile=profile,
    )

    assert block is not None
    assert "ETF core allocation gap" in block
    assert "SCHD:" in block and "70% target" in block
    assert "BND:" in block and "30% target" in block
    assert "Deployable cash" in block
    assert "Suggested deploy plan (precomputed" in block
