from unittest.mock import MagicMock, AsyncMock
import asyncio

from app.core.prompts import AnalysisAction
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from tests.test_position_prompt_metrics import _make_account, _make_position


def test_build_analysis_context_skips_market_fetch_on_followup():
    market_service = MagicMock()
    market_service.get_enriched_quote_snapshot = MagicMock()
    market_service.get_option_chains = MagicMock()

    schwab_auth_service = MagicMock()
    schwab_auth_service.get_valid_token_by_user_id = MagicMock()

    service = PortfolioAnalysisService(
        market_service=market_service,
        schwab_auth_service=schwab_auth_service,
        prompt_enrichment_service=MagicMock(),
        company_research_service=MagicMock(),
        transaction_service=MagicMock(),
        portfolio_intelligence_service=MagicMock(),
        profile_adapter=MagicMock(),
    )

    ctx = asyncio.run(
        service.build_analysis_context(
            user_id="user-1",
            account=_make_account(),
            positions=[_make_position(symbol="NVDA")],
            session_id=None,
            symbol="NVDA",
            user_prompt="let's do that",
            action=AnalysisAction.FREE_FORM,
            include_market_data=False,
        )
    )

    assert ctx.symbol == "NVDA"
    assert ctx.market_snapshot is None
    assert ctx.option_chain is None
    assert ctx.research_context is None
    schwab_auth_service.get_valid_token_by_user_id.assert_not_called()
    market_service.get_enriched_quote_snapshot.assert_not_called()
    market_service.get_option_chains.assert_not_called()


def test_build_analysis_context_loads_market_data_when_requested():
    market_service = MagicMock()
    market_service.get_enriched_quote_snapshot = MagicMock(return_value=[])
    market_service.get_option_chains = MagicMock(return_value=MagicMock())

    schwab_auth_service = MagicMock()
    token = MagicMock(access_token="token")
    schwab_auth_service.get_valid_token_by_user_id = MagicMock(return_value=token)

    prompt_enrichment_service = MagicMock()
    prompt_enrichment_service.build_market_snapshot_markdown = MagicMock(
        return_value="snapshot"
    )
    prompt_enrichment_service.resolve_option_chain_block = MagicMock(
        return_value="options"
    )
    prompt_enrichment_service.format_research_context_block = MagicMock(
        return_value="research"
    )

    company_research_service = MagicMock()
    company_research_service.build_context = MagicMock(return_value=MagicMock())

    service = PortfolioAnalysisService(
        market_service=market_service,
        schwab_auth_service=schwab_auth_service,
        prompt_enrichment_service=prompt_enrichment_service,
        company_research_service=company_research_service,
        transaction_service=MagicMock(),
        portfolio_intelligence_service=MagicMock(),
        profile_adapter=MagicMock(),
    )
    service._build_research_bundle = MagicMock(return_value=("research", "intel", False))
    service._build_recent_transactions_block = MagicMock(return_value=None)
    service.portfolio_intelligence_service.enriched_news_service.ensure_enriched = (
        AsyncMock(return_value=None)
    )

    ctx = asyncio.run(
        service.build_analysis_context(
            user_id="user-1",
            account=_make_account(),
            positions=[_make_position(symbol="NVDA")],
            session_id=None,
            symbol="NVDA",
            user_prompt="Should I trim?",
            action=AnalysisAction.FREE_FORM,
            include_market_data=True,
        )
    )

    assert ctx.market_snapshot == "snapshot"
    assert ctx.intelligence_block == "intel"
    schwab_auth_service.get_valid_token_by_user_id.assert_called_once()
    assert market_service.get_enriched_quote_snapshot.call_count == 2
    market_service.get_option_chains.assert_called_once()
