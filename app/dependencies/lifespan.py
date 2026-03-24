import os
from contextlib import asynccontextmanager

import redis
import requests
from fastapi import FastAPI

from app.adapters.schwab.schwab_redis_token_manager import SchwabRedisTokenManager
from app.adapters.schwab.schwab_trader_adapter import SchwabTraderAdapter
from app.builders.schwab_trader_builder import SchwabTraderBuilder
from app.services.llm_service import LLMService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService
from app.builders.schwab_auth_builder import SchwabAuthBuilder
from app.adapters.schwab.schwab_auth import SchwabAuth
from app.adapters.schwab.schwab_auth_access_token_adapter import (
    SchwabAuthAccessTokenAdapter,
)
from app.adapters.user.app_user_adapter import AppUserAdapter
from app.builders.app_user_builder import AppUserBuilder
from app.services.user_service import UserService
from app.adapters.llm.openai_adapter import OpenAIAdapter
from app.core.llm_config import settings
from openai import OpenAI
import oracledb
from app.adapters.schwab.schwab_market_adapter import SchwabMarketAdapter
from app.builders.schwab_market_builder import SchwabMarketBuilder
from app.services.market_service import MarketService
from app.services.prompt_enrichment_service import PromptEnrichmentService
from app.adapters.finnhub.finnhub_adapter import FinnhubAdapter
from app.builders.finnhub_builder import FinnhubBuilder
from app.services.news_service import NewsService


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
    openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    finnhub_adapter = FinnhubAdapter(api_key=finnhub_api_key)
    schwab_market_adapter = SchwabMarketAdapter(
        session=session, base_uri=schwab_market_uri
    )
    schwab_auth_access_token_adapter = SchwabAuthAccessTokenAdapter(
        client=powerpocketdb_client
    )
    app_user_adapter = AppUserAdapter(client=powerpocketdb_client)
    schwab_auth = SchwabAuth(
        client_id=schwab_client_id,
        client_secret=schwab_client_secret,
        redirect_uri=schwab_redirect_uri,
    )
    schwab_redis_token_manager = SchwabRedisTokenManager(redis_client=redis_client)
    openai_adapter = OpenAIAdapter(client=openai_client)

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
    app_user_builder = AppUserBuilder(app_user_adapter=app_user_adapter)

    news_service = NewsService(finnhub_builder=finnhub_builder)
    market_service = MarketService(schwab_market_builder=schwab_market_builder)
    prompt_enrichment_service = PromptEnrichmentService()
    llm_service = LLMService(openai_adapter=openai_adapter)
    portfolio_service = PortfolioService(schwab_trader_builder=schwab_trader_builder)
    schwab_auth_service = SchwabAuthService(
        schwab_oauth_uri=schwab_oauth_uri,
        schwab_client_id=schwab_client_id,
        schwab_redirect_uri=schwab_redirect_uri,
        schwab_auth_builder=schwab_auth_builder,
    )
    user_service = UserService(app_user_builder=app_user_builder)

    app.state.http_session = session
    app.state.redis_client = redis_client
    app.state.news_service = news_service
    app.state.prompt_enrichment_service = prompt_enrichment_service
    app.state.market_service = market_service
    app.state.llm_service = llm_service
    app.state.portfolio_service = portfolio_service
    app.state.schwab_redis_token_manager = schwab_redis_token_manager
    app.state.schwab_auth_service = schwab_auth_service
    app.state.user_service = user_service

    try:
        yield
    finally:
        redis_client.close()
        session.close()
