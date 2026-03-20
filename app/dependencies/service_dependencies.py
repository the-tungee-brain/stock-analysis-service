from fastapi import Request
from app.services.llm_service import LLMService
from app.services.market_service import MarketService
from app.services.schwab_auth_service import SchwabAuthService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.services.user_service import UserService
from app.services.portfolio_service import PortfolioService


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
