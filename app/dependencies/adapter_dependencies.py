from fastapi import Request
from app.adapters.schwab.schwab_redis_token_manager import SchwabRedisTokenManager


def get_schwab_redis_token_manager(request: Request) -> SchwabRedisTokenManager:
    return request.app.state.schwab_redis_token_manager
