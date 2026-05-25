from unittest.mock import MagicMock

from app.models.company_research_models import ResearchContext
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from tests.test_position_prompt_metrics import _make_account, _make_position


def test_research_chat_user_message_includes_holdings_and_intelligence():
    ctx = ResearchContext(symbol="AAPL")
    message = PromptEnrichmentService().build_research_chat_user_message(
        ctx=ctx,
        user_prompt="Should I trim?",
        include_context=True,
        holdings_block="SYMBOL | WEIGHT_% | ...",
        intelligence_block="## Precomputed signals\n- [WARNING] Earnings in 3 days",
    )

    content = message["content"]
    assert "RESEARCH DATA FOR AAPL" in content
    assert "YOUR HOLDINGS IN AAPL" in content
    assert "SYMBOL | WEIGHT_%" in content
    assert "PRECOMPUTED INTELLIGENCE" in content
    assert "Earnings in 3 days" in content
    assert "actual positions" in content


def test_build_research_chat_holdings_context_returns_none_without_positions():
    service = PortfolioAnalysisService(
        market_service=MagicMock(),
        prompt_enrichment_service=MagicMock(),
        transaction_service=MagicMock(),
        schwab_auth_service=MagicMock(),
        company_research_service=MagicMock(),
        portfolio_intelligence_service=MagicMock(),
        profile_adapter=MagicMock(),
    )
    account = _make_account()

    holdings, intelligence = service.build_research_chat_holdings_context(
        user_id="user-1",
        symbol="AAPL",
        account=account,
        positions=[_make_position(symbol="MSFT")],
        access_token="token",
    )

    assert holdings is None
    assert intelligence is None
