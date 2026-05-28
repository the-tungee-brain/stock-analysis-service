from unittest.mock import MagicMock

from app.models.strategy_models import (
    InvestmentStrategy,
    JourneyStep,
    JourneyStepStatus,
    JourneyStepUpdate,
    UserInvestmentProfile,
    WheelPhase,
    WheelStrategyConfig,
)
from app.services.strategy.strategy_catalog import build_initial_steps
from app.services.strategy.strategy_journey_service import StrategyJourneyService


def test_journey_step_validates_camel_case_json():
    step = JourneyStep.model_validate(
        {
            "stepId": "connect-schwab",
            "title": "Connect Schwab",
            "description": "Link your account.",
            "status": "available",
            "order": 1,
            "completedAt": None,
            "metadata": {},
        }
    )
    assert step.step_id == "connect-schwab"


def test_journey_step_round_trip_uses_camel_case_aliases():
    step = build_initial_steps(InvestmentStrategy.WHEEL)[0]
    payload = step.model_dump(mode="json", by_alias=True)
    restored = JourneyStep.model_validate(payload)
    assert restored.step_id == step.step_id


def test_wheel_config_validates_camel_case_json():
    config = WheelStrategyConfig.model_validate(
        {
            "wheelSymbols": ["AAPL"],
            "targetDeltaMin": 0.2,
            "targetDeltaMax": 0.3,
            "preferredDteDays": 7,
            "maxSingleNamePct": 15,
        }
    )
    assert config.wheel_symbols == ["AAPL"]


def test_build_initial_steps_unlocks_first_only():
    steps = build_initial_steps(InvestmentStrategy.WHEEL)
    assert len(steps) == 8
    assert steps[0].status == JourneyStepStatus.AVAILABLE
    assert all(step.status == JourneyStepStatus.LOCKED for step in steps[1:])


def test_detect_wheel_phase_ready_for_csp():
    phase = StrategyJourneyService.detect_wheel_phase(symbol="AAPL", positions=[])
    assert phase == WheelPhase.READY_FOR_CSP


def test_detect_wheel_phase_partial_lot_suggests_csp_not_cc():
    from app.models.schwab_models import Instrument, Position

    positions = [
        Position(
            shortQuantity=0,
            averagePrice=100,
            currentDayProfitLoss=0,
            currentDayProfitLossPercentage=0,
            longQuantity=50,
            settledLongQuantity=50,
            settledShortQuantity=0,
            instrument=Instrument(
                assetType="EQUITY",
                cusip="123",
                symbol="AAPL",
            ),
            marketValue=5000,
            maintenanceRequirement=0,
            currentDayCost=0,
        )
    ]
    phase = StrategyJourneyService.detect_wheel_phase(symbol="AAPL", positions=positions)
    assert phase == WheelPhase.READY_FOR_CSP
    assert StrategyJourneyService.share_quantity_for_symbol("AAPL", positions) == 50


def test_detect_wheel_phase_full_lot_ready_for_covered_call():
    from app.models.schwab_models import Instrument, Position

    positions = [
        Position(
            shortQuantity=0,
            averagePrice=100,
            currentDayProfitLoss=0,
            currentDayProfitLossPercentage=0,
            longQuantity=100,
            settledLongQuantity=100,
            settledShortQuantity=0,
            instrument=Instrument(
                assetType="EQUITY",
                cusip="123",
                symbol="AAPL",
            ),
            marketValue=10000,
            maintenanceRequirement=0,
            currentDayCost=0,
        )
    ]
    phase = StrategyJourneyService.detect_wheel_phase(symbol="AAPL", positions=positions)
    assert phase == WheelPhase.ASSIGNED_SHARES


def test_update_step_unlocks_next_step():
    profile_adapter = MagicMock()
    journey_adapter = MagicMock()
    service = StrategyJourneyService(profile_adapter, journey_adapter)

    steps = build_initial_steps(InvestmentStrategy.ETF_CORE)
    journey_adapter.get_by_user_and_strategy.return_value = MagicMock(
        steps=steps,
        current_step_id=steps[0].step_id,
        strategy=InvestmentStrategy.ETF_CORE,
    )
    journey_adapter.upsert.side_effect = lambda **kwargs: MagicMock(**kwargs)

    service.update_step(
        user_id="user-1",
        strategy=InvestmentStrategy.ETF_CORE,
        step_id="connect-schwab",
        update=JourneyStepUpdate(status=JourneyStepStatus.COMPLETED),
    )

    upsert_kwargs = journey_adapter.upsert.call_args.kwargs
    updated_steps = upsert_kwargs["steps"]
    assert updated_steps[0].status == JourneyStepStatus.COMPLETED
    assert updated_steps[1].status == JourneyStepStatus.AVAILABLE


def test_build_recommendations_prompts_connect_when_schwab_missing():
    profile_adapter = MagicMock()
    journey_adapter = MagicMock()
    service = StrategyJourneyService(profile_adapter, journey_adapter)

    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        wheel=WheelStrategyConfig(wheel_symbols=["AAPL"]),
    )
    profile_adapter.get_by_user_id.return_value = profile
    journey_adapter.get_by_user_and_strategy.return_value = MagicMock(
        steps=build_initial_steps(InvestmentStrategy.WHEEL),
        current_step_id="connect-schwab",
        strategy=InvestmentStrategy.WHEEL,
    )
    journey_adapter.upsert.return_value = journey_adapter.get_by_user_and_strategy.return_value

    recs = service.build_recommendations(
        user_id="user-1",
        schwab_linked=False,
        positions=[],
        account=None,
    )

    assert recs is not None
    assert recs.next_actions[0].type == "connect"


def test_sync_journey_completes_connect_step():
    profile_adapter = MagicMock()
    journey_adapter = MagicMock()
    service = StrategyJourneyService(profile_adapter, journey_adapter)

    profile = UserInvestmentProfile(
        user_id="user-1",
        primary_strategy=InvestmentStrategy.WHEEL,
        wheel=WheelStrategyConfig(wheel_symbols=["AAPL"]),
    )
    profile_adapter.get_by_user_id.return_value = profile

    steps = build_initial_steps(InvestmentStrategy.WHEEL)
    journey_adapter.get_by_user_and_strategy.return_value = MagicMock(
        steps=steps,
        current_step_id=steps[0].step_id,
        strategy=InvestmentStrategy.WHEEL,
    )
    journey_adapter.upsert.side_effect = lambda **kwargs: MagicMock(**kwargs)

    service.sync_journey_progress(
        user_id="user-1",
        schwab_linked=True,
        positions=[],
    )

    upsert_kwargs = journey_adapter.upsert.call_args.kwargs
    updated_steps = upsert_kwargs["steps"]
    connect = next(step for step in updated_steps if step.step_id == "connect-schwab")
    pick = next(step for step in updated_steps if step.step_id == "pick-underlying")
    assert connect.status == JourneyStepStatus.COMPLETED
    assert pick.status == JourneyStepStatus.COMPLETED
