from unittest.mock import MagicMock

from app.api.get_account_positions_route import (
    _warm_portfolio_brief_cache,
    get_account_positions,
)
from app.models.intelligence_models import PortfolioIntelligence


async def test_positions_returns_cached_brief_without_building():
    portfolio_service = MagicMock()
    portfolio_service.get_enriched_account.return_value = {
        "account": MagicMock(
            securitiesAccount=MagicMock(
                accountNumber="123",
                positions=[],
            )
        ),
        "positions": {},
        "cashSecuredPutSummary": None,
        "assignmentRiskSummary": None,
        "portfolioMetrics": MagicMock(model_dump=MagicMock(return_value={})),
    }

    schwab_auth_service = MagicMock()
    schwab_token = MagicMock(access_token="token")
    schwab_auth_service.get_valid_token_by_user_id.return_value = schwab_token

    transaction_service = MagicMock()
    transaction_service.build_recent_activity_summary.return_value = None

    cached_brief = PortfolioIntelligence(signals=[], digest=None, alerts=[])
    portfolio_analysis_service = MagicMock()
    portfolio_analysis_service.try_get_light_cached_portfolio_brief.return_value = (
        cached_brief
    )
    portfolio_analysis_service.build_portfolio_brief_for_positions_load = MagicMock()

    payload = await get_account_positions(
        background_tasks=MagicMock(),
        user_id="user-1",
        portfolio_service=portfolio_service,
        schwab_auth_service=schwab_auth_service,
        transaction_service=transaction_service,
        portfolio_analysis_service=portfolio_analysis_service,
        portfolio_memory_service=MagicMock(),
        refresh=False,
    )

    assert payload["portfolioBrief"] is not None
    assert payload["dataFreshness"]["briefStatus"] == "cached"
    assert payload["dataFreshness"]["positionsSource"] == "schwab"
    assert payload["dataFreshness"]["positionsSyncedAt"]
    portfolio_analysis_service.build_portfolio_brief_for_positions_load.assert_not_called()


async def test_positions_cache_miss_returns_pending_and_schedules_warm():
    portfolio_service = MagicMock()
    portfolio_service.get_enriched_account.return_value = {
        "account": MagicMock(
            securitiesAccount=MagicMock(
                accountNumber="123",
                positions=[],
            )
        ),
        "positions": {},
        "cashSecuredPutSummary": None,
        "assignmentRiskSummary": None,
        "portfolioMetrics": MagicMock(model_dump=MagicMock(return_value={})),
    }

    schwab_auth_service = MagicMock()
    schwab_auth_service.get_valid_token_by_user_id.return_value = MagicMock(
        access_token="token"
    )
    transaction_service = MagicMock()
    transaction_service.build_recent_activity_summary.return_value = None

    portfolio_analysis_service = MagicMock()
    portfolio_analysis_service.try_get_light_cached_portfolio_brief.return_value = None

    background_tasks = MagicMock()
    payload = await get_account_positions(
        background_tasks=background_tasks,
        user_id="user-1",
        portfolio_service=portfolio_service,
        schwab_auth_service=schwab_auth_service,
        transaction_service=transaction_service,
        portfolio_analysis_service=portfolio_analysis_service,
        portfolio_memory_service=MagicMock(),
        refresh=False,
    )

    assert payload["portfolioBrief"] is None
    assert payload["dataFreshness"]["briefStatus"] == "pending"
    background_tasks.add_task.assert_called_once()
    _args, kwargs = background_tasks.add_task.call_args
    assert _args[0] is _warm_portfolio_brief_cache
