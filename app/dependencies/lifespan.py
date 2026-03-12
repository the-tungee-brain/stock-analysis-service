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


def get_redis_client() -> redis.Redis:
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    session = requests.Session()
    redis_client = get_redis_client()

    schwab_trader_builder = get_schwab_trader_builder(session)
    llm_service = LLMService()
    portfolio_service = PortfolioService(schwab_trader_builder=schwab_trader_builder)
    schwab_redis_token_manager = SchwabRedisTokenManager(redis_client)
    schwab_auth_service = SchwabAuthService(
        schwab_redis_token_manager=schwab_redis_token_manager
    )

    app.state.http_session = session
    app.state.redis_client = redis_client
    app.state.llm_service = llm_service
    app.state.portfolio_service = portfolio_service
    app.state.schwab_redis_token_manager = schwab_redis_token_manager
    app.state.schwab_auth_service = schwab_auth_service

    try:
        yield
    finally:
        redis_client.close()
        session.close()
