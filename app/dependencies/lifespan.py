from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.adapters.schwab.schwab_redis_token_manager import SchwabRedisTokenManager
from app.services.llm_service import LLMService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService
from app.adapters.schwab.schwab_trader_adapter import SchwabTraderAdapter
from app.builders.schwab_trader_builder import SchwabTraderBuilder
import redis
import requests
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    session = requests.Session()
    schwab_trader_adapter = SchwabTraderAdapter(
        session=session, base_uri=os.getenv("SCHWAB_TRADER_API_URI")
    )
    schwab_trader_builder = SchwabTraderBuilder(
        schwab_trader_adapter=schwab_trader_adapter
    )
    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)
    llm_service = LLMService()
    portfolio_service = PortfolioService(schwab_trader_builder=schwab_trader_builder)
    schwab_redis_token_manager = SchwabRedisTokenManager(redis_client)
    schwab_auth_service = SchwabAuthService(
        schwab_redis_token_manager=schwab_redis_token_manager
    )

    app.state.redis_client = redis_client
    app.state.llm_service = llm_service
    app.state.portfolio_service = portfolio_service
    app.state.schwab_redis_token_manager = schwab_redis_token_manager
    app.state.schwab_auth_service = schwab_auth_service

    yield
    redis_client.close()
