from app.broker.strategy_symbol_alignment import (
    format_strategy_symbol_alignment_block,
    format_symbol_strategy_fit_note,
    strategy_symbol_list,
)
from app.core.prompts import (
    PortfolioContext,
    SymbolContext,
    SYSTEM_MESSAGE,
    SYSTEM_PORTFOLIO_ALLOCATION_MESSAGE,
    build_portfolio_prompt,
    build_symbol_prompt,
)
from app.models.strategy_models import (
    InvestmentStrategy,
    UserInvestmentProfile,
    WheelStrategyConfig,
)
from tests.test_position_prompt_metrics import _make_account, _make_position


def test_strategy_symbol_list_from_wheel_profile():
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        wheel=WheelStrategyConfig(wheel_symbols=["AAPL", "msft"]),
    )
    assert strategy_symbol_list(profile) == ["AAPL", "MSFT"]


def test_alignment_block_flags_held_not_on_strategy_list():
    account = _make_account(liquidation_value=100_000)
    positions = [
        _make_position(symbol="TSM", market_value=20_000),
        _make_position(symbol="AAPL", market_value=15_000),
    ]
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        wheel=WheelStrategyConfig(wheel_symbols=["AAPL", "MSFT"]),
    )

    block = format_strategy_symbol_alignment_block(
        positions=positions,
        account=account,
        profile=profile,
    )

    assert block is not None
    assert "NOT a whitelist" in block
    assert "Held but NOT on strategy list: TSM" in block
    assert "Held and on strategy list: AAPL" in block
    assert "On strategy list but not currently held: MSFT" in block
    assert "suggest adding it to the strategy symbol list" in block


def test_symbol_fit_note_for_off_list_symbol():
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        wheel=WheelStrategyConfig(wheel_symbols=["AAPL"]),
    )

    note = format_symbol_strategy_fit_note(profile, "TSM")

    assert note is not None
    assert "TSM is NOT on your saved strategy symbol list" in note
    assert "suggest adding TSM to your strategy symbol list" in note
    assert "Do NOT treat 'off-list' as a risk" in note


def test_portfolio_prompt_includes_alignment_block():
    ctx = PortfolioContext(
        account=_make_account(),
        positions=[_make_position(symbol="TSM")],
        strategy_alignment_block=(
            "## Strategy symbol list alignment\n"
            "- Held but NOT on strategy list: TSM"
        ),
    )
    prompt = build_portfolio_prompt(ctx)
    assert "STRATEGY SYMBOL LIST ALIGNMENT" in prompt
    assert "Held but NOT on strategy list: TSM" in prompt


def test_symbol_prompt_includes_profile_block():
    ctx = SymbolContext(
        symbol="TSM",
        account=_make_account(),
        positions=[_make_position(symbol="TSM")],
        investment_profile_block=(
            "## Wheel / options preferences\n"
            "- Strategy symbol list (working set — not a ban on other holdings): AAPL\n"
            "## Strategy list status (wheel)\n"
            "TSM is NOT on your saved strategy symbol list."
        ),
    )
    prompt = build_symbol_prompt(ctx)
    assert "INVESTOR PREFERENCES (SAVED PROFILE)" in prompt
    assert "TSM is NOT on your saved strategy symbol list" in prompt


def test_system_messages_include_off_list_guidance():
    assert "working set" in SYSTEM_PORTFOLIO_ALLOCATION_MESSAGE.lower()
    assert "off-list" in SYSTEM_PORTFOLIO_ALLOCATION_MESSAGE.lower()
    assert "Strategy symbol list" in SYSTEM_MESSAGE
