from datetime import date, datetime, timezone
from unittest.mock import MagicMock

from app.core.prompts import AnalysisAction
from app.models.intelligence_models import (
    IntelligenceSignal,
    PortfolioDigest,
    PortfolioIntelligence,
    ProactiveAlert,
    SectorWeight,
)
from app.models.portfolio_memory_models import SnapshotPosition
from app.models.intelligence_models import (
    OptionsScorecard,
    OptionsStrikeCandidate,
)
from app.models.schwab_models import Instrument, Position
from app.services.intelligence.option_roll_planner_service import OptionRollPlannerService
from app.services.portfolio_memory_service import PortfolioMemoryService


def _make_position(
    *,
    symbol: str = "AAPL",
    asset_type: str = "EQUITY",
    market_value: float = 10000.0,
    long_quantity: float = 100.0,
) -> Position:
    return Position(
        shortQuantity=0.0,
        averagePrice=100.0,
        currentDayProfitLoss=0.0,
        currentDayProfitLossPercentage=0.0,
        longQuantity=long_quantity,
        settledLongQuantity=long_quantity,
        settledShortQuantity=0.0,
        instrument=Instrument(
            assetType=asset_type,
            cusip="037833100",
            symbol=symbol,
        ),
        marketValue=market_value,
        maintenanceRequirement=0.0,
        currentDayCost=0.0,
    )


def test_get_portfolio_changes_detects_weight_shift():
    previous = MagicMock()
    previous.snapshot_date = date(2026, 5, 22)
    previous.liquidation_value = 100000.0
    previous.positions = [
        SnapshotPosition(
            symbol="AAPL",
            asset_type="EQUITY",
            quantity=100,
            market_value=15000,
            weight_pct=15.0,
        ),
        SnapshotPosition(
            symbol="MSFT",
            asset_type="EQUITY",
            quantity=50,
            market_value=10000,
            weight_pct=10.0,
        ),
    ]

    current = MagicMock()
    current.snapshot_date = date(2026, 5, 23)
    current.liquidation_value = 100000.0
    current.positions = [
        SnapshotPosition(
            symbol="AAPL",
            asset_type="EQUITY",
            quantity=100,
            market_value=22000,
            weight_pct=22.0,
        ),
        SnapshotPosition(
            symbol="MSFT",
            asset_type="EQUITY",
            quantity=50,
            market_value=10000,
            weight_pct=10.0,
        ),
    ]

    snapshot_adapter = MagicMock()
    snapshot_adapter.list_recent.return_value = [current, previous]

    service = PortfolioMemoryService(
        portfolio_snapshot_adapter=snapshot_adapter,
        alert_history_adapter=MagicMock(),
    )

    changes = service.get_portfolio_changes(user_id="user-1", compare_days=1)

    assert changes.from_date == date(2026, 5, 22)
    assert changes.to_date == date(2026, 5, 23)
    assert changes.weight_changes[0].symbol == "AAPL"
    assert changes.weight_changes[0].change_pct == 7.0


def test_record_alerts_resolves_missing_fingerprints():
    alert = ProactiveAlert(
        action=AnalysisAction.ASSIGNMENT_RISK,
        label="assignment risk",
        reason="Short put ITM",
        priority=1,
        symbol="AAPL",
    )

    alert_adapter = MagicMock()
    service = PortfolioMemoryService(
        portfolio_snapshot_adapter=MagicMock(),
        alert_history_adapter=alert_adapter,
    )

    service.record_alerts(user_id="user-1", alerts=[alert])

    alert_adapter.upsert_active.assert_called_once()
    alert_adapter.resolve_missing.assert_called_once()
    active_fingerprints = alert_adapter.resolve_missing.call_args.args[1]
    assert len(active_fingerprints) == 1


def test_build_attention_queue_merges_current_and_historical():
    current = ProactiveAlert(
        action=AnalysisAction.RISK_CHECK,
        label="risk check",
        reason="Sector concentration",
        priority=2,
        symbol=None,
    )
    historical = MagicMock()
    historical.action = AnalysisAction.CONCENTRATION_CHECK
    historical.label = "concentration check"
    historical.symbol = "NVDA"
    historical.reason = "NVDA is 25% of portfolio"
    historical.priority = 1
    historical.first_seen_at = datetime(2026, 5, 20, tzinfo=timezone.utc)
    historical.days_active = 3
    historical.id = "alert-1"

    alert_adapter = MagicMock()
    alert_adapter.list_active.return_value = [historical]

    service = PortfolioMemoryService(
        portfolio_snapshot_adapter=MagicMock(),
        alert_history_adapter=alert_adapter,
    )

    queue = service.build_attention_queue(
        user_id="user-1",
        current_alerts=[current],
    )

    assert len(queue) == 2
    assert queue[0].source == "current"
    assert queue[1].source == "historical"
    assert queue[1].alert_id == "alert-1"


def test_build_morning_brief_includes_digest_and_changes():
    digest = PortfolioDigest(
        sector_weights=[
            SectorWeight(sector="Technology", weight_pct=45.0, symbols=["AAPL"])
        ],
        macro_regime="VIX at 18.0",
        earnings_this_week=["AAPL"],
    )
    brief = PortfolioIntelligence(
        signals=[
            IntelligenceSignal(
                kind="concentration",
                severity="warning",
                message="AAPL elevated",
                symbol="AAPL",
            )
        ],
        digest=digest,
        alerts=[
            ProactiveAlert(
                action=AnalysisAction.CONCENTRATION_CHECK,
                label="concentration check",
                reason="AAPL elevated",
                priority=3,
                symbol="AAPL",
            )
        ],
    )

    snapshot_adapter = MagicMock()
    snapshot_adapter.list_recent.return_value = []

    service = PortfolioMemoryService(
        portfolio_snapshot_adapter=snapshot_adapter,
        alert_history_adapter=MagicMock(),
    )

    morning = service.build_morning_brief(
        user_id="user-1",
        portfolio_brief=brief,
        current_alerts=brief.alerts,
    )

    assert morning.macro_regime == "VIX at 18.0"
    assert morning.digest is not None
    assert len(morning.top_alerts) == 1
    assert morning.delivery_ready is True


def test_build_roll_suggestions_for_short_put():
    position = Position(
        shortQuantity=1.0,
        averagePrice=2.0,
        currentDayProfitLoss=0.0,
        currentDayProfitLossPercentage=0.0,
        longQuantity=0.0,
        settledLongQuantity=0.0,
        settledShortQuantity=1.0,
        instrument=Instrument(
            assetType="OPTION",
            cusip="",
            symbol="AAPL  260620P00180000",
            putCall="PUT",
            underlyingSymbol="AAPL",
            strikePrice=180.0,
            expirationDate="2026-06-20",
        ),
        marketValue=-200.0,
        maintenanceRequirement=0.0,
        currentDayCost=0.0,
    )

    scorecard = OptionsScorecard(
        underlying_price=185.0,
        csp_candidates=[
            OptionsStrikeCandidate(
                side="put",
                strike=175.0,
                expiration="2026-07-18",
                delta=-0.25,
                open_interest=500,
                bid=2.5,
                ask=2.7,
                score=0.8,
                rationale="Better candidate",
            )
        ],
    )

    suggestions = OptionRollPlannerService.build_roll_suggestions(
        positions=[position],
        symbol="AAPL",
        option_chain=MagicMock(callExpDateMap={}, putExpDateMap={}),
        scorecard=scorecard,
    )

    assert len(suggestions) == 1
    assert suggestions[0].current_strike == 180.0
    assert suggestions[0].suggested_strike == 175.0
    assert suggestions[0].action == "roll"
