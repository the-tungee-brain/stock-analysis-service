from app.models.schwab_models import Instrument, Position
from app.models.strategy_models import (
    InvestmentStrategy,
    UserInvestmentProfile,
    WheelPhase,
    WheelStrategyConfig,
)
from app.services.strategy.strategy_journey_service import StrategyJourneyService
from app.services.strategy.strategy_playbook import (
    build_symbol_statuses,
    next_action_for_symbol,
    pick_focus_symbol,
)


def _equity_position(symbol: str, *, shares: float) -> Position:
    return Position(
        shortQuantity=0,
        averagePrice=100,
        currentDayProfitLoss=0,
        currentDayProfitLossPercentage=0,
        longQuantity=shares,
        settledLongQuantity=shares,
        settledShortQuantity=0,
        instrument=Instrument(assetType="EQUITY", cusip="123", symbol=symbol),
        marketValue=shares * 100,
        maintenanceRequirement=0,
        currentDayCost=0,
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


def test_partial_lot_suggests_csp_not_covered_call():
    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        wheel=WheelStrategyConfig(wheel_symbols=["AAPL"]),
    )
    positions = [_equity_position("AAPL", shares=50)]

    statuses = build_symbol_statuses(
        profile=profile,
        strategy=InvestmentStrategy.WHEEL,
        positions=positions,
        account=None,
    )
    assert len(statuses) == 1
    assert statuses[0].wheel_phase == WheelPhase.READY_FOR_CSP
    assert "CSP" in statuses[0].status_label

    action = next_action_for_symbol(
        strategy=InvestmentStrategy.WHEEL,
        symbol="AAPL",
        wheel_phase=WheelPhase.READY_FOR_CSP,
        held=True,
        share_qty=50,
        covered_call_candidates=[
            {"rationale": "Should not use", "strike": 200.0},
        ],
    )
    assert action is not None
    assert "CSP" in action.title
    assert "50" in action.reason


def test_full_lot_suggests_covered_call_when_eligible():
    action = next_action_for_symbol(
        strategy=InvestmentStrategy.WHEEL,
        symbol="AAPL",
        wheel_phase=WheelPhase.ASSIGNED_SHARES,
        held=True,
        share_qty=100,
        covered_call_candidates=[
            {"rationale": "Good OTM call", "strike": 210.0},
        ],
    )
    assert action is not None
    assert "Covered call" in action.title
    assert action.metadata["strike"] == 210.0
