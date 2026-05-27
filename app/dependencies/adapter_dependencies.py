from fastapi import Request

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.adapters.schwab.schwab_redis_token_manager import SchwabRedisTokenManager


def get_schwab_redis_token_manager(request: Request) -> SchwabRedisTokenManager:
    return request.app.state.schwab_redis_token_manager


def get_yfinance_adapter(request: Request) -> YFinanceAdapter:
    return request.app.state.yfinance_adapter
