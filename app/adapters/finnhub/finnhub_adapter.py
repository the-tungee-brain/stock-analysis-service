from __future__ import annotations

import logging
import os
import types
from collections.abc import Callable
from typing import TypeVar

import finnhub
import requests
from finnhub.exceptions import FinnhubAPIException
from requests.adapters import HTTPAdapter

from app.adapters.cache.finnhub_response_cache import FinnhubResponseCache
from app.adapters.finnhub.finnhub_circuit import (
    FinnhubCircuitBreaker,
    FinnhubUnavailableError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_TIMEOUT_SECONDS = 15.0
DEFAULT_API_URL = "https://finnhub.io/api/v1"


def _normalized_request(self, method: str, path: str, **kwargs):
    uri = f"{self.API_URL.rstrip('/')}/{path.lstrip('/')}"
    kwargs["timeout"] = kwargs.get("timeout", self.DEFAULT_TIMEOUT)
    kwargs["params"] = self._format_params(kwargs.get("params", {}))

    response = getattr(self._session, method)(uri, **kwargs)
    return self._handle_response(response)


class FinnhubAdapter:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float | None = None,
        circuit_cooldown_seconds: float | None = None,
        response_cache: FinnhubResponseCache | None = None,
    ):
        timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv("FINNHUB_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
        )
        cooldown = float(
            circuit_cooldown_seconds
            if circuit_cooldown_seconds is not None
            else os.getenv("FINNHUB_CIRCUIT_COOLDOWN_SECONDS", "120")
        )
        self._circuit = FinnhubCircuitBreaker.from_env(cooldown_seconds=cooldown)
        self._cache = response_cache
        self.finnhub_client = finnhub.Client(api_key=api_key)
        self.finnhub_client.API_URL = os.getenv(
            "FINNHUB_API_URL", DEFAULT_API_URL
        ).rstrip("/")
        self.finnhub_client.DEFAULT_TIMEOUT = timeout
        self.finnhub_client._request = types.MethodType(
            _normalized_request,
            self.finnhub_client,
        )
        no_retry = HTTPAdapter(max_retries=0)
        self.finnhub_client._session.mount("https://", no_retry)
        self.finnhub_client._session.mount("http://", no_retry)

    @staticmethod
    def _cache_key(*parts: str) -> str:
        return ":".join(part.strip().upper() for part in parts if part)

    def _cached_call(
        self,
        endpoint: str,
        cache_key: str,
        label: str,
        fn: Callable[[], T],
    ) -> T:
        if self._cache is not None:
            cached = self._cache.get(endpoint=endpoint, cache_key=cache_key)
            if cached is not None:
                return cached

        result = self._call(label, fn)

        if self._cache is not None:
            self._cache.put(endpoint=endpoint, cache_key=cache_key, value=result)

        return result

    def _call(self, label: str, fn: Callable[[], T]) -> T:
        if not self._circuit.allow_request():
            raise FinnhubUnavailableError("Finnhub circuit open")

        try:
            result = fn()
        except FinnhubAPIException as exc:
            if exc.status_code != 429:
                self._circuit.record_failure()
            logger.warning("Finnhub %s unavailable: %s", label, exc)
            raise
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as exc:
            self._circuit.record_failure()
            logger.warning("Finnhub %s unavailable: %s", label, exc)
            raise
        else:
            self._circuit.record_success()
            return result

    def get_company_news(self, symbol: str, _from: str, to: str):
        cache_key = self._cache_key(symbol, _from, to)
        return self._cached_call(
            "company_news",
            cache_key,
            "company_news",
            lambda: self.finnhub_client.company_news(
                symbol=symbol, _from=_from, to=to
            ),
        )

    def invalidate_company_news(self, symbol: str, _from: str, to: str) -> None:
        if self._cache is None:
            return
        self._cache.delete(
            "company_news",
            self._cache_key(symbol, _from, to),
        )

    def get_general_news(self, category: str = "general", min_id: int = 0):
        cache_key = self._cache_key(category, str(min_id))
        return self._cached_call(
            "general_news",
            cache_key,
            "general_news",
            lambda: self.finnhub_client.general_news(category, min_id=min_id),
        )

    def get_company_profile(self, symbol: str):
        cache_key = self._cache_key(symbol)
        return self._cached_call(
            "company_profile",
            cache_key,
            "company_profile",
            lambda: self.finnhub_client.company_profile2(symbol=symbol),
        )

    def get_quote(self, symbol: str):
        cache_key = self._cache_key(symbol)
        return self._cached_call(
            "quote",
            cache_key,
            "quote",
            lambda: self.finnhub_client.quote(symbol=symbol),
        )

    def get_company_earnings(self, symbol: str, limit: int | None = None):
        cache_key = self._cache_key(symbol, str(limit or ""))
        return self._cached_call(
            "company_earnings",
            cache_key,
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
        cache_key = self._cache_key(symbol, _from, to, str(international))
        return self._cached_call(
            "earnings_calendar",
            cache_key,
            "earnings_calendar",
            lambda: self.finnhub_client.earnings_calendar(
                _from=_from,
                to=to,
                symbol=symbol,
                international=international,
            ),
        )

    def get_transcripts_list(self, symbol: str):
        cache_key = self._cache_key(symbol)
        return self._cached_call(
            "transcripts_list",
            cache_key,
            "transcripts_list",
            lambda: self.finnhub_client.transcripts_list(symbol=symbol),
        )

    def get_transcript(self, transcript_id: str):
        cache_key = self._cache_key(transcript_id)
        return self._cached_call(
            "transcript",
            cache_key,
            "transcript",
            lambda: self.finnhub_client.transcripts(_id=transcript_id),
        )

    def get_press_releases(self, symbol: str, _from: str, to: str):
        cache_key = self._cache_key(symbol, _from, to)
        return self._cached_call(
            "press_releases",
            cache_key,
            "press_releases",
            lambda: self.finnhub_client.press_releases(
                symbol=symbol, _from=_from, to=to
            ),
        )

    def get_stock_peers(self, symbol: str) -> list[str]:
        cache_key = self._cache_key(symbol)
        return self._cached_call(
            "stock_peers",
            cache_key,
            "stock_peers",
            lambda: self.finnhub_client.stock_peers(symbol=symbol),
        )
