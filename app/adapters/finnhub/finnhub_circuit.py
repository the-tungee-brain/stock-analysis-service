from __future__ import annotations

import time
from threading import Lock


class FinnhubUnavailableError(Exception):
    """Raised when Finnhub calls are skipped because the circuit is open."""


class FinnhubCircuitBreaker:
    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        self._opened_at: float | None = None
        self._lock = Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                self._opened_at = None
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._opened_at = time.monotonic()
