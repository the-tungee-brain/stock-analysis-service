import json
from pathlib import Path
from unittest.mock import MagicMock

from app.core.prompts import AnalysisAction
from app.models.company_research_models import ResearchContext
from app.models.schwab_option_chain_models import OptionChain
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from tests.test_position_prompt_metrics import _make_account, _make_position

FIXTURE = Path(__file__).parent / "fixtures" / "schwab_option_chain_sample.json"


def test_research_chat_user_message_includes_holdings_intelligence_and_options():
    ctx = ResearchContext(symbol="AAPL")
    message = PromptEnrichmentService().build_research_chat_user_message(
        ctx=ctx,
        user_prompt="Should I trim?",
        include_context=True,
        holdings_block="SYMBOL | WEIGHT_% | ...",
        intelligence_block="## Precomputed signals\n- [WARNING] Earnings in 3 days",
        option_chain_block="Held option contracts\n| Strike | delta |",
    )

    content = message["content"]
    assert "RESEARCH DATA FOR AAPL" in content
    assert "YOUR HOLDINGS IN AAPL" in content
    assert "SYMBOL | WEIGHT_%" in content
    assert "PRECOMPUTED INTELLIGENCE" in content
    assert "Earnings in 3 days" in content
    assert "OPTION DATA (HELD CONTRACTS + CHAIN)" in content
    assert "Held option contracts" in content
    assert "actual positions" in content
    assert "bid/ask" in content


def test_build_research_chat_holdings_context_skips_portfolio_blocks_without_schwab():
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

    holdings, intelligence, option_chain = service.build_research_chat_holdings_context(
        user_id="user-1",
        symbol="AAPL",
        account=account,
        positions=[_make_position(symbol="MSFT")],
        access_token=None,
    )

    assert holdings is None
    assert intelligence is None
    assert option_chain is None


def test_build_research_chat_holdings_context_loads_options_without_symbol_positions():
    prompt_enrichment_service = MagicMock()
    prompt_enrichment_service.format_intelligence_block.return_value = (
        "## Options scorecard\n- CSP $190 put"
    )
    prompt_enrichment_service.has_actionable_options_scorecard.return_value = True
    prompt_enrichment_service.resolve_option_chain_block.return_value = (
        "Held option contracts\nUnderlying: AAPL @ $200"
    )

    market_service = MagicMock()
    market_service.get_enriched_quote_snapshot.return_value = {}

    service = PortfolioAnalysisService(
        market_service=market_service,
        prompt_enrichment_service=prompt_enrichment_service,
        transaction_service=MagicMock(),
        schwab_auth_service=MagicMock(),
        company_research_service=MagicMock(),
        portfolio_intelligence_service=MagicMock(),
        profile_adapter=MagicMock(),
    )
    service.build_symbol_intelligence = MagicMock(return_value=MagicMock())
    service._load_symbol_option_chain = MagicMock(
        return_value=OptionChain.model_validate(json.loads(FIXTURE.read_text()))
    )
    service._build_macro_market_context_block = MagicMock(return_value=None)
    account = _make_account()

    holdings, intelligence, option_chain = service.build_research_chat_holdings_context(
        user_id="user-1",
        symbol="AAPL",
        account=account,
        positions=[_make_position(symbol="MSFT")],
        access_token="token",
    )

    assert holdings is None
    assert intelligence is not None
    assert "Options scorecard" in intelligence
    assert option_chain is not None
    assert "Held option contracts" in option_chain
    prompt_enrichment_service.resolve_option_chain_block.assert_called_once()
    call_kwargs = prompt_enrichment_service.resolve_option_chain_block.call_args.kwargs
    assert call_kwargs["action"] is AnalysisAction.FREE_FORM
    assert call_kwargs["has_options_scorecard"] is True


def test_build_research_chat_holdings_context_includes_macro_headlines():
    portfolio_intelligence_service = MagicMock()
    portfolio_intelligence_service.build_macro_market_context_block.return_value = (
        "## Market headlines (general, last 24h)\n- Fed holds rates steady (Reuters)"
    )
    portfolio_intelligence_service.build_symbol_intelligence.return_value = MagicMock()

    prompt_enrichment_service = MagicMock()
    prompt_enrichment_service.format_intelligence_block.return_value = (
        "## Precomputed signals\n- [INFO] Size OK"
    )
    prompt_enrichment_service.has_actionable_options_scorecard.return_value = False
    prompt_enrichment_service.resolve_option_chain_block.return_value = (
        "Underlying: AAPL @ $200.12"
    )

    market_service = MagicMock()
    market_service.get_enriched_quote_snapshot.return_value = {}

    service = PortfolioAnalysisService(
        market_service=market_service,
        prompt_enrichment_service=prompt_enrichment_service,
        transaction_service=MagicMock(),
        schwab_auth_service=MagicMock(),
        company_research_service=MagicMock(),
        portfolio_intelligence_service=portfolio_intelligence_service,
        profile_adapter=MagicMock(),
    )
    service.build_symbol_intelligence = MagicMock(return_value=MagicMock())
    service._load_symbol_option_chain = MagicMock(return_value=MagicMock())
    service._build_macro_market_context_block = MagicMock(
        side_effect=portfolio_intelligence_service.build_macro_market_context_block
    )
    account = _make_account()

    holdings, intelligence, option_chain = service.build_research_chat_holdings_context(
        user_id="user-1",
        symbol="AAPL",
        account=account,
        positions=[_make_position(symbol="AAPL", market_value=10_000)],
        access_token="token",
    )

    assert holdings is not None
    assert intelligence is not None
    assert option_chain is not None
    assert "Fed holds rates steady" in intelligence
    assert "Precomputed signals" in intelligence
    service._build_macro_market_context_block.assert_called_once()
