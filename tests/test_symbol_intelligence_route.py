from datetime import date, timedelta
from unittest.mock import MagicMock

from app.models.intelligence_models import SymbolIntelligence
from app.services.portfolio_analysis_service import (
    INTELLIGENCE_OPTION_LOOKAHEAD_DAYS,
    INTELLIGENCE_OPTION_STRIKE_COUNT,
    PortfolioAnalysisService,
)
from tests.test_position_prompt_metrics import _make_account, _make_position


def test_build_symbol_intelligence_returns_symbol_on_research_failure():
    company_research_service = MagicMock()
    company_research_service.build_context.side_effect = RuntimeError("fail")

    service = PortfolioAnalysisService(
        market_service=MagicMock(),
        prompt_enrichment_service=MagicMock(),
        transaction_service=MagicMock(),
        schwab_auth_service=MagicMock(),
        company_research_service=company_research_service,
        portfolio_intelligence_service=MagicMock(),
        profile_adapter=MagicMock(),
    )

    result = service.build_symbol_intelligence(
        user_id="user-1",
        symbol="AAPL",
    )

    assert result == SymbolIntelligence(symbol="AAPL", partial=True)


def test_build_symbol_intelligence_delegates_to_intelligence_service():
    ctx = MagicMock()
    company_research_service = MagicMock()
    company_research_service.build_context.return_value = ctx

    portfolio_intelligence_service = MagicMock()
    portfolio_intelligence_service.attach_enriched_news.return_value = ctx
    expected = SymbolIntelligence(symbol="AAPL", signals=[])
    portfolio_intelligence_service.build_symbol_intelligence.return_value = expected

    market_service = MagicMock()
    market_service.get_option_chains.return_value = MagicMock()
    quote_snapshot = MagicMock()
    quote_snapshot.implied_vol = 0.285
    market_service.get_enriched_quote_snapshot.return_value = {"AAPL": quote_snapshot}

    transaction_service = MagicMock()
    transaction_service.get_filled_orders_by_symbol.return_value = []

    service = PortfolioAnalysisService(
        market_service=market_service,
        prompt_enrichment_service=MagicMock(),
        transaction_service=transaction_service,
        schwab_auth_service=MagicMock(),
        company_research_service=company_research_service,
        portfolio_intelligence_service=portfolio_intelligence_service,
        profile_adapter=MagicMock(),
    )

    account = _make_account()
    positions = [_make_position(symbol="AAPL")]

    result = service.build_symbol_intelligence(
        user_id="user-1",
        symbol="AAPL",
        account=account,
        positions=positions,
        access_token="token",
        include_options=True,
    )

    assert result == expected
    portfolio_intelligence_service.build_symbol_intelligence.assert_called_once()
    _, delegate_kwargs = (
        portfolio_intelligence_service.build_symbol_intelligence.call_args
    )
    assert delegate_kwargs["underlying_iv_percent"] == 0.285
    market_service.get_option_chains.assert_called_once()
    market_service.get_enriched_quote_snapshot.assert_called_once_with(
        access_token="token",
        symbols=["AAPL"],
    )
    _, kwargs = market_service.get_option_chains.call_args
    assert kwargs["strike_count"] == INTELLIGENCE_OPTION_STRIKE_COUNT
    today = date.today()
    assert kwargs["from_date"] == today.isoformat()
    assert kwargs["to_date"] == (
        today + timedelta(days=INTELLIGENCE_OPTION_LOOKAHEAD_DAYS)
    ).isoformat()
