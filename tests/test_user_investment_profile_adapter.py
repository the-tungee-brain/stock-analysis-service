from app.adapters.user.user_investment_profile_adapter import UserInvestmentProfileAdapter
from app.models.strategy_models import (
    DividendStrategyConfig,
    EtfCoreStrategyConfig,
    InvestmentStrategy,
    WheelStrategyConfig,
)


def test_active_strategy_configs_keeps_only_matching_block():
    wheel = WheelStrategyConfig(wheel_symbols=["NVDA"])
    dividend = DividendStrategyConfig(dividend_symbols=["KO"])
    etf_core = EtfCoreStrategyConfig(target_allocation={"VTI": 70.0, "BND": 30.0})

    kept_wheel, kept_dividend, kept_etf = UserInvestmentProfileAdapter._active_strategy_configs(
        InvestmentStrategy.WHEEL,
        wheel=wheel,
        dividend=dividend,
        etf_core=etf_core,
    )
    assert kept_wheel is wheel
    assert kept_dividend is None
    assert kept_etf is None

    kept_wheel, kept_dividend, kept_etf = UserInvestmentProfileAdapter._active_strategy_configs(
        InvestmentStrategy.ETF_CORE,
        wheel=wheel,
        dividend=dividend,
        etf_core=etf_core,
    )
    assert kept_wheel is None
    assert kept_dividend is None
    assert kept_etf is etf_core

    kept_wheel, kept_dividend, kept_etf = UserInvestmentProfileAdapter._active_strategy_configs(
        InvestmentStrategy.DIVIDEND,
        wheel=wheel,
        dividend=dividend,
        etf_core=etf_core,
    )
    assert kept_wheel is None
    assert kept_dividend is dividend
    assert kept_etf is None
