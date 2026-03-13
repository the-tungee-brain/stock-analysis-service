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
import oracledb


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
    user = os.getenv("POWERPOCKETDB_USER")
    password = os.getenv("POWERPOCKETDB_PASSWORD")
    dsn = os.getenv("POWERPOCKETDB_TP_TNS")

    if not all([user, password, dsn]):
        raise ValueError("Missing ORACLE_USER, ORACLE_PASSWORD, or ORACLE_DSN env vars")

    return oracledb.create_pool(user=user, password=password, dsn=dsn)


@asynccontextmanager
async def lifespan(app: FastAPI):
    schwab_client_id = os.getenv("SCHWAB_CLIENT_ID")
    schwab_client_secret = os.getenv("SCHWAB_CLIENT_SECRET")
    schwab_redirect_uri = os.getenv("SCHWAB_REDIRECT_URI")
    schwab_oauth_uri = os.getenv("SCHWAB_OAUTH_URI")

    session = requests.Session()
    redis_client = get_redis_client()
    powerpocketdb_client = get_powerpocketdb_client()

    schwab_auth_access_token_adapter = SchwabAuthAccessTokenAdapter(
        client=powerpocketdb_client
    )
    app_user_adapter = AppUserAdapter(client=powerpocketdb_client)
    schwab_auth = SchwabAuth(
        client_id=schwab_client_id,
        client_secret=schwab_client_secret,
        redirect_uri=schwab_redirect_uri,
    )
    schwab_redis_token_manager = SchwabRedisTokenManager(redis_client)

    schwab_auth_builder = SchwabAuthBuilder(
        schwab_auth=schwab_auth,
        schwab_auth_access_token_adapter=schwab_auth_access_token_adapter,
        schwab_redis_token_manager=schwab_redis_token_manager,
    )
    schwab_trader_builder = get_schwab_trader_builder(session)
    app_user_builder = AppUserBuilder(app_user_adapter=app_user_adapter)

    llm_service = LLMService()
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
