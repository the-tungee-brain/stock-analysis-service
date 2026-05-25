from __future__ import annotations

import os
import time
from threading import Lock


class FinnhubUnavailableError(Exception):
    """Raised when Finnhub calls are skipped because the circuit is open."""


class FinnhubCircuitBreaker:
    def __init__(
        self,
        cooldown_seconds: float,
        *,
        failure_threshold: int = 3,
    ):
        self.cooldown_seconds = cooldown_seconds
        self.failure_threshold = max(1, failure_threshold)
        self._opened_at: float | None = None
        self._consecutive_failures = 0
        self._lock = Lock()

    @classmethod
    def from_env(cls, cooldown_seconds: float) -> FinnhubCircuitBreaker:
        threshold = int(os.getenv("FINNHUB_CIRCUIT_FAILURE_THRESHOLD", "3"))
        return cls(cooldown_seconds, failure_threshold=threshold)

    def allow_request(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                self._opened_at = None
                self._consecutive_failures = 0
                return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._opened_at = None
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._opened_at = time.monotonic()
