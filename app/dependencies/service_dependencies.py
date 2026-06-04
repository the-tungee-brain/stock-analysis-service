from fastapi import Request
from app.builders.yfinance_analysis_builder import YFinanceAnalysisBuilder
from app.builders.yfinance_funds_builder import YFinanceFundsBuilder
from app.builders.yfinance_financials_builder import YFinanceFinancialsBuilder
from app.services.llm_service import LLMService
from app.services.market_service import MarketService
from app.services.schwab_auth_service import SchwabAuthService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.user_service import UserService
from app.services.portfolio_service import PortfolioService
from app.services.news_service import NewsService
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.chat_service import ChatService
from app.services.company_profile_service import CompanyProfileService
from app.services.company_research_service import CompanyResearchService
from app.services.ticker_service import TickerService
from app.services.sec_research_service import SecResearchService
from app.services.transaction_service import TransactionService
from app.services.earnings_service import EarningsService
from app.services.dividend_research_service import DividendResearchService
from app.services.etf_research_service import EtfResearchService
from app.services.enriched_news_service import EnrichedNewsService
from app.services.intelligence.portfolio_intelligence_service import (
    PortfolioIntelligenceService,
)
from app.services.morning_brief_delivery_service import MorningBriefDeliveryService
from app.services.portfolio_memory_service import PortfolioMemoryService
from app.services.portfolio_news_service import PortfolioNewsService
from app.services.strategy.strategy_journey_service import StrategyJourneyService
from app.services.strategy.strategy_stock_screener_service import (
    StrategyStockScreenerService,
)
from app.services.strategy.strategy_stock_suggestion_service import (
    StrategyStockSuggestionService,
)
from app.notifications.composite_service import CompositeNotificationService
from app.services.strategy.momentum_breakout_alert_refresh_service import (
    MomentumBreakoutAlertRefreshService,
)
from app.services.strategy.momentum_breakout_alert_service import (
    MomentumBreakoutAlertService,
)
from app.services.strategy.momentum_breakout_research_service import (
    MomentumBreakoutResearchService,
)
from app.services.strategy.wheel_backtest_service import WheelBacktestService
from app.services.account_deletion_service import AccountDeletionService
from app.services.research_overview_service import ResearchOverviewService
from app.services.watchlist_service import WatchlistService


def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service


def get_portfolio_service(request: Request) -> PortfolioService:
    return request.app.state.portfolio_service


def get_schwab_auth_service(request: Request) -> SchwabAuthService:
    return request.app.state.schwab_auth_service


def get_user_service(request: Request) -> UserService:
    return request.app.state.user_service


def get_market_service(request: Request) -> MarketService:
    return request.app.state.market_service


def get_prompt_enrichment_service(request: Request) -> PromptEnrichmentService:
    return request.app.state.prompt_enrichment_service


def get_news_service(request: Request) -> NewsService:
    return request.app.state.news_service


def get_portfolio_analysis_service(request: Request) -> PortfolioAnalysisService:
    return request.app.state.portfolio_analysis_service


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_company_profile_service(request: Request) -> CompanyProfileService:
    return request.app.state.company_profile_service


def get_company_research_service(request: Request) -> CompanyResearchService:
    return request.app.state.company_research_service


def get_yfinance_financials_builder(request: Request) -> YFinanceFinancialsBuilder:
    return request.app.state.yfinance_financials_builder


def get_yfinance_analysis_builder(request: Request) -> YFinanceAnalysisBuilder:
    return request.app.state.yfinance_analysis_builder


def get_yfinance_funds_builder(request: Request) -> YFinanceFundsBuilder:
    return request.app.state.yfinance_funds_builder


def get_ticker_service(request: Request) -> TickerService:
    return request.app.state.ticker_service


def get_sec_research_service(request: Request) -> SecResearchService:
    return request.app.state.sec_research_service


def get_transaction_service(request: Request) -> TransactionService:
    return request.app.state.transaction_service


def get_earnings_service(request: Request) -> EarningsService:
    return request.app.state.earnings_service


def get_portfolio_intelligence_service(request: Request) -> PortfolioIntelligenceService:
    return request.app.state.portfolio_intelligence_service


def get_enriched_news_service(request: Request) -> EnrichedNewsService:
    return request.app.state.enriched_news_service


def get_etf_research_service(request: Request) -> EtfResearchService:
    return request.app.state.etf_research_service


def get_dividend_research_service(request: Request) -> DividendResearchService:
    return request.app.state.dividend_research_service


def get_portfolio_memory_service(request: Request) -> PortfolioMemoryService:
    return request.app.state.portfolio_memory_service


def get_portfolio_news_service(request: Request) -> PortfolioNewsService:
    return request.app.state.portfolio_news_service


def get_morning_brief_delivery_service(
    request: Request,
) -> MorningBriefDeliveryService:
    return request.app.state.morning_brief_delivery_service


def get_strategy_journey_service(request: Request) -> StrategyJourneyService:
    return request.app.state.strategy_journey_service


def get_strategy_stock_suggestion_service(
    request: Request,
) -> StrategyStockSuggestionService:
    return request.app.state.strategy_stock_suggestion_service


def get_strategy_stock_screener_service(
    request: Request,
) -> StrategyStockScreenerService:
    return request.app.state.strategy_stock_screener_service


def get_account_deletion_service(request: Request) -> AccountDeletionService:
    return request.app.state.account_deletion_service


def get_wheel_backtest_service(request: Request) -> WheelBacktestService:
    return request.app.state.wheel_backtest_service


def get_momentum_breakout_research_service(
    request: Request,
) -> MomentumBreakoutResearchService:
    return request.app.state.momentum_breakout_research_service


def get_momentum_breakout_alert_service(
    request: Request,
) -> MomentumBreakoutAlertService:
    return request.app.state.momentum_breakout_alert_service


def get_momentum_breakout_alert_refresh_service(
    request: Request,
) -> MomentumBreakoutAlertRefreshService:
    return request.app.state.momentum_breakout_alert_refresh_service


def get_momentum_breakout_notification_service(
    request: Request,
) -> CompositeNotificationService:
    return request.app.state.momentum_breakout_notification_service


def get_research_overview_service(request: Request) -> ResearchOverviewService:
    return request.app.state.research_overview_service


def get_watchlist_service(request: Request) -> WatchlistService:
    return request.app.state.watchlist_service
