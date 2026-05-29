from unittest.mock import MagicMock

from app.adapters.cache.app_user_cache import AppUserCache
from app.models.user_models import AppUserItem
from datetime import datetime, timezone


def test_app_user_cache_round_trip():
    redis_client = MagicMock()
    redis_client.get.return_value = None

    cache = AppUserCache(redis_client=redis_client, ttl_seconds=60)
    user = AppUserItem(
        id="id-1",
        identity_sub="sub-1",
        identity_provider="google",
        email="user@example.com",
        full_name="User",
        avatar_url=None,
        created_at=datetime.now(timezone.utc),
        last_login_at=datetime.now(timezone.utc),
    )

    cache.put("sub-1", user)

    redis_client.setex.assert_called_once()
    key, ttl, payload = redis_client.setex.call_args[0]
    assert "sub-1" in key
    assert ttl == 60

    redis_client.get.return_value = payload
    loaded = cache.get("sub-1")
    assert loaded is not None
    assert loaded.email == "user@example.com"
