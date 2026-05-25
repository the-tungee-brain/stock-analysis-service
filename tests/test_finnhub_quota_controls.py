import threading
import time
from unittest.mock import MagicMock

import pytest

from app.adapters.finnhub.finnhub_circuit import FinnhubUnavailableError
from app.adapters.finnhub.finnhub_rate_limiter import FinnhubRateLimiter
from app.services.news_service import NewsService, finnhub_press_releases_enabled


def test_finnhub_rate_limiter_blocks_when_burst_exceeds_limit():
    limiter = FinnhubRateLimiter(
        max_requests=2,
        window_seconds=1.0,
        max_wait_seconds=2.0,
    )

    limiter.acquire()
    limiter.acquire()

    started = time.monotonic()
    limiter.acquire()
    elapsed = time.monotonic() - started

    assert elapsed >= 0.5


def test_finnhub_rate_limiter_raises_when_wait_exceeded():
    limiter = FinnhubRateLimiter(
        max_requests=1,
        window_seconds=60.0,
        max_wait_seconds=0.05,
    )

    limiter.acquire()

    with pytest.raises(FinnhubUnavailableError):
        limiter.acquire()


def test_finnhub_rate_limiter_is_thread_safe():
    limiter = FinnhubRateLimiter(
        max_requests=5,
        window_seconds=1.0,
        max_wait_seconds=2.0,
    )
    errors: list[Exception] = []

    def worker() -> None:
        try:
            for _ in range(2):
                limiter.acquire()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors


def test_finnhub_rate_limiter_from_env_disabled_when_zero(monkeypatch):
    monkeypatch.setenv("FINNHUB_RATE_LIMIT_PER_MINUTE", "0")
    assert FinnhubRateLimiter.from_env() is None


def test_press_releases_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FINNHUB_PRESS_RELEASES", raising=False)
    assert not finnhub_press_releases_enabled()

    finnhub_builder = MagicMock()
    service = NewsService(finnhub_builder=finnhub_builder)

    response = service.get_press_releases("AAPL")

    assert response.root == []
    finnhub_builder.get_press_releases.assert_not_called()


def test_press_releases_enabled_when_flag_set(monkeypatch):
    monkeypatch.setenv("FINNHUB_PRESS_RELEASES", "1")
    assert finnhub_press_releases_enabled()
