from app.broker.strategy_portfolio_guidance import format_strategy_portfolio_guidance_block
from app.core.prompts import (
    PortfolioContext,
    _portfolio_v1_decision_order,
    _structured_portfolio_analysis_v1_task,
    build_portfolio_prompt,
)
from app.models.strategy_models import InvestmentStrategy, UserInvestmentProfile, WheelStrategyConfig
from tests.test_option_utils import _make_option_position
from tests.test_position_prompt_metrics import _make_account, _make_position


def test_wheel_strategy_guidance_includes_csp_posture():
    account = _make_account(liquidation_value=100_000)
    current = account.securitiesAccount.currentBalances.model_copy(
        update={"cashBalance": 70_000}
    )
    account = account.model_copy(
        update={
            "securitiesAccount": account.securitiesAccount.model_copy(
                update={"currentBalances": current}
            )
        }
    )
    positions = [
        _make_position(symbol="NVDA", market_value=25_000),
        _make_option_position(
            symbol="NVDA  260620P00170000",
            put_call="PUT",
            short_qty=1,
            strike_price=170.0,
        ),
    ]
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        wheel=WheelStrategyConfig(
            wheel_symbols=["AAPL", "MSFT"],
            max_single_name_pct=15.0,
        ),
    )

    block = format_strategy_portfolio_guidance_block(
        profile=profile,
        positions=positions,
        account=account,
    )

    assert block is not None
    assert "Primary strategy: wheel" in block
    assert "CSP reserved cash" in block
    assert "Suggested capital posture (precomputed" in block
    assert "Do NOT mention ETF deploy plans" in block
    assert "NVDA" in block


def test_build_portfolio_prompt_includes_strategy_framework():
    ctx = PortfolioContext(
        account=_make_account(),
        positions=[_make_position(symbol="NVDA", market_value=25_000)],
        strategy_guidance_block="## Primary strategy: wheel",
        primary_strategy=InvestmentStrategy.WHEEL,
    )

    prompt = build_portfolio_prompt(ctx, json_response=True)

    assert "STRATEGY ANALYSIS FRAMEWORK" in prompt
    assert "Primary strategy: wheel" in prompt
    assert "CSP reserved cash" in prompt or "wheel" in prompt.lower()


def test_wheel_capital_posture_recommends_hold_when_csp_dominates():
    account = _make_account(liquidation_value=100_000)
    current = account.securitiesAccount.currentBalances.model_copy(
        update={"cashBalance": 24_000}
    )
    account = account.model_copy(
        update={
            "securitiesAccount": account.securitiesAccount.model_copy(
                update={"currentBalances": current}
            )
        }
    )
    positions = [
        _make_position(symbol="NVDA", market_value=12_000),
        _make_option_position(
            symbol="NVDA  260620P00170000",
            put_call="PUT",
            short_qty=1,
            strike_price=170.0,
        ),
    ]
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        wheel=WheelStrategyConfig(
            wheel_symbols=["NVDA", "MSFT"],
            max_single_name_pct=15.0,
        ),
    )

    block = format_strategy_portfolio_guidance_block(
        profile=profile,
        positions=positions,
        account=account,
    )

    assert block is not None
    assert "Suggested capital posture (precomputed" in block
    assert "pause new cash-secured puts" in block
    assert "CSP reserved" in block or "earmarked" in block


def test_v1_decision_order_differs_by_strategy():
    wheel_order = _portfolio_v1_decision_order(InvestmentStrategy.WHEEL)
    etf_order = _portfolio_v1_decision_order(InvestmentStrategy.ETF_CORE)

    assert "CSP reserves" in wheel_order or "capital posture" in wheel_order
    assert "never tell the user a deploy plan is missing" in wheel_order
    assert "Suggested deploy plan" in etf_order
    assert "Suggested capital posture" in _structured_portfolio_analysis_v1_task(
        InvestmentStrategy.WHEEL
    )
    wheel_task = _structured_portfolio_analysis_v1_task(InvestmentStrategy.WHEEL)
    assert "Never mention ETF deploy plans" in wheel_task
    assert "no precomputed deploy plan" in wheel_task.lower()
    assert _structured_portfolio_analysis_v1_task(InvestmentStrategy.WHEEL) != (
        _structured_portfolio_analysis_v1_task(InvestmentStrategy.ETF_CORE)
    )
