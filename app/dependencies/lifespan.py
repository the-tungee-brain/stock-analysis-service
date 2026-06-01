import os
from contextlib import asynccontextmanager

import oracledb
import redis
import requests
from fastapi import FastAPI
from openai import OpenAI

from app.adapters.cache.dividend_history_cache import DividendHistoryCache
from app.adapters.cache.enriched_news_cache import EnrichedNewsCache
from app.adapters.cache.finnhub_response_cache import FinnhubResponseCache
from app.adapters.cache.app_user_cache import AppUserCache
from app.adapters.cache.portfolio_brief_cache import PortfolioBriefCache
from app.adapters.cache.research_context_cache import ResearchContextCache

from app.adapters.chat.chat_messages_adapter import ChatMessagesAdapter
from app.adapters.chat.chat_sessions_adapter import ChatSessionsAdapter
from app.adapters.finnhub.finnhub_adapter import FinnhubAdapter
from app.adapters.llm.openai_adapter import OpenAIAdapter
from app.adapters.schwab.schwab_auth import SchwabAuth
from app.adapters.schwab.schwab_auth_access_token_adapter import (
    SchwabAuthAccessTokenAdapter,
)
from app.adapters.schwab.schwab_market_adapter import SchwabMarketAdapter
from app.adapters.schwab.schwab_redis_token_manager import SchwabRedisTokenManager
from app.adapters.cache.recent_orders_cache import RecentOrdersCache
from app.adapters.cache.llm_output_cache import LLMOutputCache
from app.adapters.schwab.schwab_trader_adapter import SchwabTraderAdapter
from app.adapters.user.app_user_adapter import AppUserAdapter
from app.adapters.user.waitlist_adapter import WaitlistAdapter
from app.adapters.user.user_investment_profile_adapter import (
    UserInvestmentProfileAdapter,
)
from app.adapters.user.user_strategy_journey_adapter import UserStrategyJourneyAdapter
from app.adapters.user.watchlist_adapter import WatchlistAdapter
from app.adapters.market.ticker_symbol_adapter import TickerSymbolAdapter
from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.adapters.email.email_adapter import EmailAdapter
from app.adapters.portfolio.alert_history_adapter import AlertHistoryAdapter
from app.adapters.portfolio.morning_brief_delivery_adapter import (
    MorningBriefDeliveryAdapter,
)
from app.adapters.portfolio.portfolio_snapshot_adapter import PortfolioSnapshotAdapter

from app.builders.app_user_builder import AppUserBuilder
from app.builders.waitlist_builder import WaitlistBuilder
from app.builders.chat_messages_builder import ChatMessagesBuilder
from app.builders.chat_sessions_builder import ChatSessionsBuilder
from app.builders.finnhub_builder import FinnhubBuilder
from app.builders.news_analytics_builder import NewsAnalyticsBuilder
from app.builders.prompt_builder import PromptBuilder
from app.builders.schwab_auth_builder import SchwabAuthBuilder
from app.builders.schwab_market_builder import SchwabMarketBuilder
from app.builders.schwab_trader_builder import SchwabTraderBuilder
from app.builders.performance_builder import PerformanceBuilder
from app.builders.earnings_builder import EarningsBuilder
from app.builders.ticker_symbol_builder import TickerSymbolBuilder
from app.builders.fundamentals_builder import FundamentalsBuilder
from app.builders.yfinance_analysis_builder import YFinanceAnalysisBuilder
from app.builders.yfinance_funds_builder import YFinanceFundsBuilder
from app.builders.yfinance_financials_builder import YFinanceFinancialsBuilder

from app.core.llm_config import settings
from app.core.access_control import max_active_users

from app.services.chat_service import ChatService
from app.services.company_profile_service import CompanyProfileService
from app.services.company_research_service import CompanyResearchService
from app.services.llm_service import LLMService
from app.services.market_service import MarketService
from app.services.news_service import NewsService
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.portfolio_service import PortfolioService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.schwab_auth_service import SchwabAuthService
from app.services.user_service import UserService
from app.services.account_deletion_service import AccountDeletionService
from app.services.ticker_service import TickerService
from app.services.watchlist_service import WatchlistService
from app.services.transaction_service import TransactionService
from app.adapters.sec.sec_edgar_adapter import SecEdgarAdapter
from app.adapters.securitiesdb.securitiesdb_adapter import SecuritiesDbAdapter
from app.builders.sec_cik_builder import SecCikBuilder
from app.builders.sec_financials_builder import SecFinancialsBuilder
from app.builders.sec_ratios_builder import SecRatiosBuilder
from app.services.sec_research_service import SecResearchService
from app.services.earnings_service import EarningsService
from app.services.dividend_research_service import DividendResearchService
from app.services.etf_research_service import EtfResearchService
from app.services.enriched_news_service import EnrichedNewsService
from app.services.intelligence.peer_comparison_service import PeerComparisonService
from app.services.intelligence.portfolio_intelligence_service import (
    PortfolioIntelligenceService,
)
from app.services.morning_brief_delivery_service import MorningBriefDeliveryService
from app.services.portfolio_memory_service import PortfolioMemoryService
from app.services.portfolio_news_service import PortfolioNewsService
from app.services.research_overview_service import ResearchOverviewService
from app.services.strategy.strategy_journey_service import StrategyJourneyService
from app.services.strategy.strategy_stock_screener_service import (
    StrategyStockScreenerService,
)
from app.services.strategy.strategy_stock_suggestion_service import (
    StrategyStockSuggestionService,
)
from app.services.strategy.wheel_backtest_service import WheelBacktestService


def get_redis_client() -> redis.Redis:
    host = os.getenv("REDIS_HOST")
    port = int(os.getenv("REDIS_PORT"))
    password = os.getenv("REDIS_PASSWORD")

    return redis.Redis(
        host=host,
        port=port,
        password=password,
        decode_responses=True,
    )


def get_schwab_trader_builder(session: requests.Session) -> SchwabTraderBuilder:
    base_uri = os.getenv("SCHWAB_TRADER_API_URI")
    adapter = SchwabTraderAdapter(session=session, base_uri=base_uri)
    return SchwabTraderBuilder(schwab_trader_adapter=adapter)


def get_powerpocketdb_client() -> oracledb.ConnectionPool:
    oracledb.defaults.fetch_lobs = False

    user = os.getenv("POWERPOCKETDB_USER")
    password = os.getenv("POWERPOCKETDB_PASSWORD")
    dsn = os.getenv("POWERPOCKETDB_TP_TNS")

    return oracledb.create_pool(
        user=user,
        password=password,
        dsn=dsn,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    schwab_client_id = os.getenv("SCHWAB_CLIENT_ID")
    schwab_client_secret = os.getenv("SCHWAB_CLIENT_SECRET")
    schwab_redirect_uri = os.getenv("SCHWAB_REDIRECT_URI")
    schwab_oauth_uri = os.getenv("SCHWAB_OAUTH_URI")
    schwab_market_uri = os.getenv("SCHWAB_MARKET_URI")
    finnhub_api_key = os.getenv("FINNHUB_API_KEY")

    session = requests.Session()
    redis_client = get_redis_client()
    powerpocketdb_client = get_powerpocketdb_client()
    settings.validate()
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    finnhub_response_cache = FinnhubResponseCache(redis_client=redis_client)
    finnhub_adapter = FinnhubAdapter(
        api_key=finnhub_api_key,
        response_cache=finnhub_response_cache,
    )
    schwab_market_adapter = SchwabMarketAdapter(
        session=session, base_uri=schwab_market_uri
    )
    schwab_auth_access_token_adapter = SchwabAuthAccessTokenAdapter(
        client=powerpocketdb_client
    )
    app_user_adapter = AppUserAdapter(client=powerpocketdb_client)
    waitlist_adapter = WaitlistAdapter(client=powerpocketdb_client)
    user_investment_profile_adapter = UserInvestmentProfileAdapter(
        client=powerpocketdb_client
    )
    user_strategy_journey_adapter = UserStrategyJourneyAdapter(
        client=powerpocketdb_client
    )
    watchlist_adapter = WatchlistAdapter(client=powerpocketdb_client)
    schwab_auth = SchwabAuth(
        client_id=schwab_client_id,
        client_secret=schwab_client_secret,
        redirect_uri=schwab_redirect_uri,
    )
    schwab_redis_token_manager = SchwabRedisTokenManager(redis_client=redis_client)
    research_context_cache = ResearchContextCache(redis_client=redis_client)
    dividend_history_cache = DividendHistoryCache(redis_client=redis_client)
    portfolio_brief_cache = PortfolioBriefCache(redis_client=redis_client)
    app_user_cache = AppUserCache(redis_client=redis_client)
    enriched_news_cache = EnrichedNewsCache(redis_client=redis_client)
    recent_orders_cache = RecentOrdersCache(redis_client=redis_client)
    llm_output_cache = LLMOutputCache(redis_client=redis_client)
    openai_adapter = OpenAIAdapter(client=openai_client)
    chat_messages_adapter = ChatMessagesAdapter(client=powerpocketdb_client)
    chat_sessions_adapter = ChatSessionsAdapter(client=powerpocketdb_client)
    yfinance_adapter = YFinanceAdapter()
    ticker_symbol_adapter = TickerSymbolAdapter(client=powerpocketdb_client)
    portfolio_snapshot_adapter = PortfolioSnapshotAdapter(client=powerpocketdb_client)
    alert_history_adapter = AlertHistoryAdapter(client=powerpocketdb_client)
    morning_brief_delivery_adapter = MorningBriefDeliveryAdapter(
        client=powerpocketdb_client
    )
    email_adapter = EmailAdapter(session=session)

    finnhub_builder = FinnhubBuilder(finnhub_adapter=finnhub_adapter)
    schwab_market_builder = SchwabMarketBuilder(
        schwab_market_adapter=schwab_market_adapter
    )
    schwab_auth_builder = SchwabAuthBuilder(
        schwab_auth=schwab_auth,
        schwab_auth_access_token_adapter=schwab_auth_access_token_adapter,
        schwab_redis_token_manager=schwab_redis_token_manager,
    )
    schwab_trader_builder = get_schwab_trader_builder(session)
    app_user_builder = AppUserBuilder(
        app_user_adapter=app_user_adapter,
        app_user_cache=app_user_cache,
    )
    waitlist_builder = WaitlistBuilder(waitlist_adapter=waitlist_adapter)
    news_analytics_builder = NewsAnalyticsBuilder(openai_adapter=openai_adapter)
    prompt_builder = PromptBuilder(openai_adapter=openai_adapter)
    chat_messages_builder = ChatMessagesBuilder(
        chat_messages_adapter=chat_messages_adapter
    )
    chat_sessions_builder = ChatSessionsBuilder(
        chat_sessions_adapter=chat_sessions_adapter
    )
    performance_builder = PerformanceBuilder(market_data_adapter=yfinance_adapter)
    fundamentals_builder = FundamentalsBuilder(market_data_adapter=yfinance_adapter)
    yfinance_financials_builder = YFinanceFinancialsBuilder(
        yfinance_adapter=yfinance_adapter
    )
    yfinance_analysis_builder = YFinanceAnalysisBuilder(
        yfinance_adapter=yfinance_adapter
    )
    yfinance_funds_builder = YFinanceFundsBuilder(yfinance_adapter=yfinance_adapter)
    earnings_builder = EarningsBuilder(
        yfinance_adapter=yfinance_adapter,
        finnhub_adapter=finnhub_adapter,
        yfinance_analysis_builder=yfinance_analysis_builder,
    )
    sec_edgar_adapter = SecEdgarAdapter.from_env(session=session)
    securitiesdb_adapter = SecuritiesDbAdapter.from_env(session=session)
    sec_cik_builder = SecCikBuilder(sec_edgar_adapter=sec_edgar_adapter)
    sec_financials_builder = SecFinancialsBuilder()
    sec_ratios_builder = SecRatiosBuilder()
    sec_research_service = SecResearchService(
        sec_edgar_adapter=sec_edgar_adapter,
        sec_cik_builder=sec_cik_builder,
        sec_financials_builder=sec_financials_builder,
        sec_ratios_builder=sec_ratios_builder,
    )
    sec_cik_builder._load_ticker_map()
    ticker_symbol_builder = TickerSymbolBuilder(
        ticker_symbol_adapter=ticker_symbol_adapter
    )
    etf_research_service = EtfResearchService(
        securitiesdb_adapter=securitiesdb_adapter,
        fundamentals_builder=fundamentals_builder,
    )
    dividend_research_service = DividendResearchService(
        securitiesdb_adapter=securitiesdb_adapter,
        dividend_history_cache=dividend_history_cache,
    )

    news_service = NewsService(
        finnhub_builder=finnhub_builder,
        yfinance_adapter=yfinance_adapter,
    )
    market_service = MarketService(
        schwab_market_builder=schwab_market_builder,
        performance_builder=performance_builder,
    )
    prompt_enrichment_service = PromptEnrichmentService()
    earnings_service = EarningsService(
        earnings_builder=earnings_builder,
        finnhub_builder=finnhub_builder,
        news_service=news_service,
    )
    llm_service = LLMService(
        openai_adapter=openai_adapter,
        news_analytics_builder=news_analytics_builder,
        prompt_builder=prompt_builder,
        llm_output_cache=llm_output_cache,
    )
    portfolio_service = PortfolioService(schwab_trader_builder=schwab_trader_builder)
    schwab_auth_service = SchwabAuthService(
        schwab_oauth_uri=schwab_oauth_uri,
        schwab_client_id=schwab_client_id,
        schwab_redirect_uri=schwab_redirect_uri,
        schwab_auth_builder=schwab_auth_builder,
    )
    user_service = UserService(
        app_user_builder=app_user_builder,
        waitlist_builder=waitlist_builder,
        max_active_users=max_active_users(),
    )
    chat_service = ChatService(
        chat_sessions_builder=chat_sessions_builder,
        chat_messages_builder=chat_messages_builder,
    )
    account_deletion_service = AccountDeletionService(
        schwab_auth_service=schwab_auth_service,
        chat_sessions_builder=chat_sessions_builder,
        chat_messages_builder=chat_messages_builder,
        app_user_adapter=app_user_adapter,
        user_investment_profile_adapter=user_investment_profile_adapter,
        user_strategy_journey_adapter=user_strategy_journey_adapter,
        watchlist_adapter=watchlist_adapter,
        alert_history_adapter=alert_history_adapter,
        portfolio_snapshot_adapter=portfolio_snapshot_adapter,
        morning_brief_delivery_adapter=morning_brief_delivery_adapter,
        waitlist_adapter=waitlist_adapter,
        recent_orders_cache=recent_orders_cache,
        portfolio_brief_cache=portfolio_brief_cache,
    )
    company_profile_service = CompanyProfileService(
        finnhub_builder=finnhub_builder,
        yfinance_adapter=yfinance_adapter,
        ticker_symbol_builder=ticker_symbol_builder,
    )
    enriched_news_service = EnrichedNewsService(
        enriched_news_cache=enriched_news_cache,
        news_service=news_service,
        prompt_enrichment_service=prompt_enrichment_service,
        llm_service=llm_service,
    )
    company_research_service = CompanyResearchService(
        company_profile_service=company_profile_service,
        market_service=market_service,
        news_service=news_service,
        fundamentals_builder=fundamentals_builder,
        sec_research_service=sec_research_service,
        earnings_service=earnings_service,
        research_context_cache=research_context_cache,
        enriched_news_service=enriched_news_service,
        ticker_symbol_builder=ticker_symbol_builder,
        etf_research_service=etf_research_service,
        yfinance_financials_builder=yfinance_financials_builder,
    )
    peer_comparison_service = PeerComparisonService(
        yfinance_adapter=yfinance_adapter,
        performance_builder=performance_builder,
    )
    portfolio_intelligence_service = PortfolioIntelligenceService(
        peer_comparison_service=peer_comparison_service,
        enriched_news_service=enriched_news_service,
        news_service=news_service,
        llm_output_cache=llm_output_cache,
    )
    transaction_service = TransactionService(
        schwab_trader_builder=schwab_trader_builder,
        recent_orders_cache=recent_orders_cache,
    )
    portfolio_analysis_service = PortfolioAnalysisService(
        market_service=market_service,
        schwab_auth_service=schwab_auth_service,
        prompt_enrichment_service=prompt_enrichment_service,
        company_research_service=company_research_service,
        transaction_service=transaction_service,
        portfolio_intelligence_service=portfolio_intelligence_service,
        profile_adapter=user_investment_profile_adapter,
        portfolio_brief_cache=portfolio_brief_cache,
    )
    ticker_service = TickerService(ticker_symbol_builder=ticker_symbol_builder)
    watchlist_service = WatchlistService(
        watchlist_adapter=watchlist_adapter,
        ticker_service=ticker_service,
        finnhub_builder=finnhub_builder,
    )
    research_overview_service = ResearchOverviewService(
        company_research_service=company_research_service,
        company_profile_service=company_profile_service,
        market_service=market_service,
        ticker_service=ticker_service,
        portfolio_service=portfolio_service,
        schwab_auth_service=schwab_auth_service,
        portfolio_analysis_service=portfolio_analysis_service,
        yfinance_analysis_builder=yfinance_analysis_builder,
        yfinance_funds_builder=yfinance_funds_builder,
        etf_research_service=etf_research_service,
        prompt_enrichment_service=prompt_enrichment_service,
        llm_service=llm_service,
    )
    portfolio_memory_service = PortfolioMemoryService(
        portfolio_snapshot_adapter=portfolio_snapshot_adapter,
        alert_history_adapter=alert_history_adapter,
    )
    portfolio_news_service = PortfolioNewsService(yfinance_adapter=yfinance_adapter)
    morning_brief_delivery_service = MorningBriefDeliveryService(
        app_user_adapter=app_user_adapter,
        delivery_adapter=morning_brief_delivery_adapter,
        email_adapter=email_adapter,
        portfolio_analysis_service=portfolio_analysis_service,
        portfolio_service=portfolio_service,
        transaction_service=transaction_service,
        schwab_auth_service=schwab_auth_service,
        portfolio_memory_service=portfolio_memory_service,
    )
    strategy_journey_service = StrategyJourneyService(
        profile_adapter=user_investment_profile_adapter,
        journey_adapter=user_strategy_journey_adapter,
    )
    strategy_stock_suggestion_service = StrategyStockSuggestionService(
        prompt_enrichment_service=prompt_enrichment_service,
        llm_service=llm_service,
    )
    strategy_stock_screener_service = StrategyStockScreenerService()
    wheel_backtest_service = WheelBacktestService(yfinance_adapter=yfinance_adapter)

    try:
        from models.prediction_service import load_deployed_model

        app.state.pattern_loaded_model = load_deployed_model()
    except FileNotFoundError:
        app.state.pattern_loaded_model = None

    app.state.http_session = session
    app.state.redis_client = redis_client
    app.state.yfinance_adapter = yfinance_adapter
    app.state.yfinance_financials_builder = yfinance_financials_builder
    app.state.yfinance_analysis_builder = yfinance_analysis_builder
    app.state.yfinance_funds_builder = yfinance_funds_builder
    app.state.news_service = news_service
    app.state.prompt_enrichment_service = prompt_enrichment_service
    app.state.market_service = market_service
    app.state.llm_service = llm_service
    app.state.portfolio_service = portfolio_service
    app.state.schwab_redis_token_manager = schwab_redis_token_manager
    app.state.schwab_auth_service = schwab_auth_service
    app.state.user_service = user_service
    from app.core import paid_access

    paid_access.bind_user_service(user_service)
    app.state.account_deletion_service = account_deletion_service
    app.state.portfolio_analysis_service = portfolio_analysis_service
    app.state.chat_service = chat_service
    app.state.company_profile_service = company_profile_service
    app.state.company_research_service = company_research_service
    app.state.portfolio_intelligence_service = portfolio_intelligence_service
    app.state.enriched_news_service = enriched_news_service
    app.state.earnings_service = earnings_service
    app.state.sec_research_service = sec_research_service
    app.state.etf_research_service = etf_research_service
    app.state.dividend_research_service = dividend_research_service
    app.state.ticker_service = ticker_service
    app.state.watchlist_service = watchlist_service
    app.state.research_overview_service = research_overview_service
    app.state.transaction_service = transaction_service
    app.state.recent_orders_cache = recent_orders_cache
    app.state.portfolio_memory_service = portfolio_memory_service
    app.state.portfolio_news_service = portfolio_news_service
    app.state.morning_brief_delivery_service = morning_brief_delivery_service
    app.state.strategy_journey_service = strategy_journey_service
    app.state.strategy_stock_suggestion_service = strategy_stock_suggestion_service
    app.state.strategy_stock_screener_service = strategy_stock_screener_service
    app.state.wheel_backtest_service = wheel_backtest_service

    try:
        yield
    finally:
        redis_client.close()
        session.close()
