from __future__ import annotations

import os
import time
from collections import deque
from threading import Lock

from app.adapters.finnhub.finnhub_circuit import FinnhubUnavailableError


class FinnhubRateLimiter:
    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        max_wait_seconds: float = 30.0,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_wait_seconds = max_wait_seconds
        self._timestamps: deque[float] = deque()
        self._lock = Lock()

    @classmethod
    def from_env(cls) -> FinnhubRateLimiter | None:
        limit = int(os.getenv("FINNHUB_RATE_LIMIT_PER_MINUTE", "50"))
        if limit <= 0:
            return None
        max_wait = float(os.getenv("FINNHUB_RATE_LIMIT_MAX_WAIT_SECONDS", "30"))
        return cls(max_requests=limit, max_wait_seconds=max_wait)

    def acquire(self) -> None:
        deadline = time.monotonic() + self.max_wait_seconds

        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.window_seconds:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return

                wait_seconds = self.window_seconds - (now - self._timestamps[0])

            if time.monotonic() + wait_seconds > deadline:
                raise FinnhubUnavailableError("Finnhub rate limit wait exceeded")

            time.sleep(min(max(wait_seconds, 0.0), 0.1))
