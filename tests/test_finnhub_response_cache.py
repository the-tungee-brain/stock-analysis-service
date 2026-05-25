import json
from unittest.mock import MagicMock

import pytest
import requests

from app.adapters.cache.finnhub_response_cache import FinnhubResponseCache
from app.adapters.finnhub.finnhub_adapter import FinnhubAdapter
from app.adapters.finnhub.finnhub_circuit import FinnhubUnavailableError


def test_finnhub_response_cache_roundtrip():
    redis_client = MagicMock()
    stored: dict[str, str] = {}

    redis_client.setex = lambda key, ttl, value: stored.update({key: value})
    redis_client.get = lambda key: stored.get(key)

    cache = FinnhubResponseCache(redis_client=redis_client)
    cache.put(
        endpoint="company_news",
        cache_key="AAPL:2026-05-18:2026-05-25",
        value=[{"headline": "Test"}],
    )

    loaded = cache.get(
        endpoint="company_news",
        cache_key="AAPL:2026-05-18:2026-05-25",
    )
    assert loaded == [{"headline": "Test"}]
    assert json.loads(stored["finnhub:company_news:AAPL:2026-05-18:2026-05-25"]) == [
        {"headline": "Test"}
    ]


def test_finnhub_adapter_uses_response_cache():
    redis_client = MagicMock()
    stored: dict[str, str] = {}

    redis_client.setex = lambda key, ttl, value: stored.update({key: value})
    redis_client.get = lambda key: stored.get(key)

    cache = FinnhubResponseCache(redis_client=redis_client)
    adapter = FinnhubAdapter(
        api_key="test-key",
        response_cache=cache,
        rate_limiter=None,
    )
    adapter.finnhub_client = MagicMock()
    adapter.finnhub_client.quote.return_value = {"c": 100.0}

    first = adapter.get_quote("AAPL")
    second = adapter.get_quote("AAPL")

    assert first == {"c": 100.0}
    assert second == {"c": 100.0}
    adapter.finnhub_client.quote.assert_called_once()


def test_finnhub_adapter_skips_upstream_when_circuit_open():
    adapter = FinnhubAdapter(
        api_key="test-key",
        circuit_cooldown_seconds=60,
        rate_limiter=None,
    )
    adapter.finnhub_client = MagicMock()
    adapter.finnhub_client.quote.side_effect = requests.exceptions.ConnectTimeout(
        "timeout",
        request=MagicMock(),
    )

    with pytest.raises(requests.exceptions.ConnectTimeout):
        adapter.get_quote("AAPL")

    adapter.finnhub_client.quote.reset_mock()

    with pytest.raises(FinnhubUnavailableError):
        adapter.get_quote("AAPL")

    adapter.finnhub_client.quote.assert_not_called()
