from fastapi import Request
from app.services.llm_service import LLMService
from app.services.market_service import MarketService
from app.services.schwab_auth_service import SchwabAuthService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.user_service import UserService
from app.services.portfolio_service import PortfolioService
from app.services.news_service import NewsService
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.chat_service import ChatService


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
