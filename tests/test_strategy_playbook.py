from app.models.strategy_models import (
    InvestmentStrategy,
    UserInvestmentProfile,
    WheelPhase,
    WheelStrategyConfig,
)
from app.services.strategy.strategy_journey_service import StrategyJourneyService
from app.services.strategy.strategy_playbook import (
    build_symbol_statuses,
    pick_focus_symbol,
)


def test_build_symbol_statuses_for_wheel_symbols():
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        wheel=WheelStrategyConfig(wheel_symbols=["AAPL", "MSFT"]),
    )
    statuses = build_symbol_statuses(
        profile=profile,
        strategy=InvestmentStrategy.WHEEL,
        positions=[],
        account=None,
    )
    assert [status.symbol for status in statuses] == ["AAPL", "MSFT"]
    assert all(status.wheel_phase == WheelPhase.READY_FOR_CSP for status in statuses)
    assert all(status.next_action is not None for status in statuses)


def test_pick_focus_symbol_prefers_actionable():
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        wheel=WheelStrategyConfig(wheel_symbols=["AAPL", "MSFT"]),
    )
    statuses = build_symbol_statuses(
        profile=profile,
        strategy=InvestmentStrategy.WHEEL,
        positions=[],
        account=None,
    )
    assert pick_focus_symbol(statuses) in {"AAPL", "MSFT"}


def test_build_recommendations_includes_symbol_statuses():
    from unittest.mock import MagicMock

    profile_adapter = MagicMock()
    journey_adapter = MagicMock()
    service = StrategyJourneyService(profile_adapter, journey_adapter)

    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        wheel=WheelStrategyConfig(wheel_symbols=["AAPL", "MSFT"]),
    )
    profile_adapter.get_by_user_id.return_value = profile
    journey_adapter.get_by_user_and_strategy.return_value = MagicMock(
        steps=[],
        current_step_id=None,
        strategy=InvestmentStrategy.WHEEL,
    )
    journey_adapter.upsert.return_value = journey_adapter.get_by_user_and_strategy.return_value

    recs = service.build_recommendations(
        user_id="user-1",
        schwab_linked=True,
        positions=[],
        account=None,
    )

    assert recs is not None
    assert len(recs.symbol_statuses) == 2
    assert len(recs.next_actions) >= 2
