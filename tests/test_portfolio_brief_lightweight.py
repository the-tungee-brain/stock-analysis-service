from unittest.mock import MagicMock

from app.adapters.cache.portfolio_brief_cache import PortfolioBriefCache
from app.models.intelligence_models import PortfolioIntelligence, PortfolioDigest
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from tests.test_position_prompt_metrics import _make_account, _make_position


def test_portfolio_brief_cache_roundtrip():
    redis_client = MagicMock()
    redis_client.get.return_value = None

    cache = PortfolioBriefCache(redis_client=redis_client)
    account = _make_account()
    positions = [_make_position(symbol="AAPL", market_value=50_000)]
    fingerprint = cache.fingerprint(positions, account)
    brief = PortfolioIntelligence(
        signals=[],
        digest=PortfolioDigest(sector_weights=[]),
        alerts=[],
    )

    cache.put(user_id="user-1", fingerprint=fingerprint, brief=brief)

    redis_client.setex.assert_called_once()
    key, ttl, payload = redis_client.setex.call_args[0]
    assert key == f"portfolio:brief:full:user-1:{fingerprint}"
    assert ttl == cache.ttl_seconds
    assert "signals" in payload


def test_build_portfolio_brief_lightweight_skips_full_research():
    portfolio_intelligence_service = MagicMock()
    portfolio_intelligence_service.attach_enriched_news.side_effect = lambda ctx: ctx
    portfolio_intelligence_service.build_portfolio_intelligence.return_value = (
        PortfolioIntelligence(signals=[], digest=None, alerts=[])
    )

    company_research_service = MagicMock()
    lightweight_ctx = MagicMock(snapshot=MagicMock(sector="Technology"))
    company_research_service.build_lightweight_context.return_value = lightweight_ctx

    market_service = MagicMock()
    market_service.get_enriched_quote_snapshot.return_value = {}

    service = PortfolioAnalysisService(
        market_service=market_service,
        schwab_auth_service=MagicMock(),
        prompt_enrichment_service=MagicMock(),
        company_research_service=company_research_service,
        transaction_service=MagicMock(),
        portfolio_intelligence_service=portfolio_intelligence_service,
        profile_adapter=MagicMock(),
    )

    account = _make_account()
    positions = [_make_position(symbol="AAPL", market_value=50_000)]

    service.build_portfolio_brief(
        user_id="user-1",
        account=account,
        positions=positions,
        access_token="token",
        lightweight=True,
    )

    company_research_service.build_lightweight_context.assert_called()
    company_research_service.build_context.assert_not_called()


def test_build_portfolio_brief_for_positions_load_uses_cache():
    cache = MagicMock()
    cached_brief = PortfolioIntelligence(signals=[], digest=None, alerts=[])
    cache.fingerprint.return_value = "fp123"
    cache.get.return_value = cached_brief

    company_research_service = MagicMock()

    service = PortfolioAnalysisService(
        market_service=MagicMock(),
        schwab_auth_service=MagicMock(),
        prompt_enrichment_service=MagicMock(),
        company_research_service=company_research_service,
        transaction_service=MagicMock(),
        portfolio_intelligence_service=MagicMock(),
        profile_adapter=MagicMock(),
        portfolio_brief_cache=cache,
    )

    account = _make_account()
    positions = [_make_position(symbol="AAPL", market_value=50_000)]

    brief = service.build_portfolio_brief_for_positions_load(
        user_id="user-1",
        account=account,
        positions=positions,
        access_token="token",
    )

    assert brief is cached_brief
    cache.get.assert_called_once_with(
        user_id="user-1",
        fingerprint="fp123",
        variant=PortfolioBriefCache.VARIANT_LIGHT,
    )
    company_research_service.build_lightweight_context.assert_not_called()
    company_research_service.build_context.assert_not_called()


def test_build_portfolio_brief_with_cache_uses_full_variant():
    cache = MagicMock()
    cached_brief = PortfolioIntelligence(signals=[], digest=None, alerts=[])
    cache.fingerprint.return_value = "fp456"
    cache.get.return_value = cached_brief

    company_research_service = MagicMock()

    service = PortfolioAnalysisService(
        market_service=MagicMock(),
        schwab_auth_service=MagicMock(),
        prompt_enrichment_service=MagicMock(),
        company_research_service=company_research_service,
        transaction_service=MagicMock(),
        portfolio_intelligence_service=MagicMock(),
        profile_adapter=MagicMock(),
        portfolio_brief_cache=cache,
    )

    account = _make_account()
    positions = [_make_position(symbol="AAPL", market_value=50_000)]

    brief = service.build_portfolio_brief_with_cache(
        user_id="user-1",
        account=account,
        positions=positions,
        access_token="token",
    )

    assert brief is cached_brief
    cache.get.assert_called_once_with(
        user_id="user-1",
        fingerprint="fp456",
        variant=PortfolioBriefCache.VARIANT_FULL,
    )
    company_research_service.build_context.assert_not_called()


def test_build_portfolio_brief_full_context_does_not_opt_into_company_news():
    portfolio_intelligence_service = MagicMock()
    portfolio_intelligence_service.attach_enriched_news.side_effect = lambda ctx: ctx
    portfolio_intelligence_service.build_portfolio_intelligence.return_value = (
        PortfolioIntelligence(signals=[], digest=None, alerts=[])
    )

    company_research_service = MagicMock()
    ctx = MagicMock()
    ctx.snapshot = MagicMock(sector="Technology")
    ctx.asset_type = "EQUITY"
    company_research_service.build_context.return_value = ctx

    market_service = MagicMock()
    market_service.get_enriched_quote_snapshot.return_value = {}

    service = PortfolioAnalysisService(
        market_service=market_service,
        schwab_auth_service=MagicMock(),
        prompt_enrichment_service=MagicMock(),
        company_research_service=company_research_service,
        transaction_service=MagicMock(),
        portfolio_intelligence_service=portfolio_intelligence_service,
        profile_adapter=MagicMock(),
    )

    account = _make_account()
    positions = [_make_position(symbol="AAPL", market_value=50_000)]

    service.build_portfolio_brief(
        user_id="user-1",
        account=account,
        positions=positions,
        access_token="token",
        lightweight=False,
    )

    company_research_service.build_context.assert_called_once_with(symbol="AAPL")
