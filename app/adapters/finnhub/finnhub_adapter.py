from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TypeVar

import finnhub
import requests
from finnhub.exceptions import FinnhubAPIException

from app.adapters.finnhub.finnhub_circuit import (
    FinnhubCircuitBreaker,
    FinnhubUnavailableError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class FinnhubAdapter:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float | None = None,
        circuit_cooldown_seconds: float | None = None,
    ):
        timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("FINNHUB_TIMEOUT_SECONDS", "2")
        )
        cooldown = float(
            circuit_cooldown_seconds
            if circuit_cooldown_seconds is not None
            else os.getenv("FINNHUB_CIRCUIT_COOLDOWN_SECONDS", "120")
        )
        self._circuit = FinnhubCircuitBreaker(cooldown_seconds=cooldown)
        self.finnhub_client = finnhub.Client(api_key=api_key)
        self.finnhub_client.DEFAULT_TIMEOUT = timeout

    def _call(self, label: str, fn: Callable[[], T]) -> T:
        if not self._circuit.allow_request():
            raise FinnhubUnavailableError("Finnhub circuit open")

        try:
            result = fn()
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            FinnhubAPIException,
        ) as exc:
            self._circuit.record_failure()
            logger.warning("Finnhub %s unavailable: %s", label, exc)
            raise
        else:
            self._circuit.record_success()
            return result

    def get_company_news(self, symbol: str, _from: str, to: str):
        return self._call(
            "company_news",
            lambda: self.finnhub_client.company_news(
                symbol=symbol, _from=_from, to=to
            ),
        )

    def get_company_profile(self, symbol: str):
        return self._call(
            "company_profile",
            lambda: self.finnhub_client.company_profile2(symbol=symbol),
        )

    def get_quote(self, symbol: str):
        return self._call(
            "quote",
            lambda: self.finnhub_client.quote(symbol=symbol),
        )

    def get_company_earnings(self, symbol: str, limit: int | None = None):
        return self._call(
            "company_earnings",
            lambda: self.finnhub_client.company_earnings(
                symbol=symbol, limit=limit
            ),
        )

    def get_earnings_calendar(
        self,
        _from: str,
        to: str,
        symbol: str = "",
        international: bool = False,
    ):
        return self._call(
            "earnings_calendar",
            lambda: self.finnhub_client.earnings_calendar(
                _from=_from,
                to=to,
                symbol=symbol,
                international=international,
            ),
        )

    def get_transcripts_list(self, symbol: str):
        return self._call(
            "transcripts_list",
            lambda: self.finnhub_client.transcripts_list(symbol=symbol),
        )

    def get_transcript(self, transcript_id: str):
        return self._call(
            "transcript",
            lambda: self.finnhub_client.transcripts(_id=transcript_id),
        )

    def get_press_releases(self, symbol: str, _from: str, to: str):
        return self._call(
            "press_releases",
            lambda: self.finnhub_client.press_releases(
                symbol=symbol, _from=_from, to=to
            ),
        )

    def get_stock_peers(self, symbol: str) -> list[str]:
        return self._call(
            "stock_peers",
            lambda: self.finnhub_client.stock_peers(symbol=symbol),
        )
