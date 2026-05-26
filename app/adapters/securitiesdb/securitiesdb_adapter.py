import os
from typing import Any

import requests

from app.adapters.sec.sec_edgar_adapter import TTLCache


class SecuritiesDbAdapter:
    BASE_URL = "https://securitiesdb.com/api/v1"

    def __init__(
        self,
        session: requests.Session,
        *,
        cache_ttl_seconds: int = 21_600,
    ) -> None:
        self.session = session
        self._cache = TTLCache(ttl_seconds=cache_ttl_seconds)

    @classmethod
    def from_env(cls, session: requests.Session) -> "SecuritiesDbAdapter":
        ttl = int(os.getenv("SECURITIESDB_CACHE_TTL_SECONDS", "21600"))
        return cls(session=session, cache_ttl_seconds=ttl)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        api_key = os.getenv("SECURITIESDB_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_etf_holdings(self, ticker: str) -> dict[str, Any] | None:
        symbol = ticker.strip().upper()
        if not symbol:
            return None

        cache_key = f"etf_holdings:{symbol}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        url = f"{self.BASE_URL}/etfs/{symbol}/holdings"
        timeout = float(os.getenv("SECURITIESDB_TIMEOUT_SECONDS", "15"))
        response = self.session.get(url, headers=self._headers(), timeout=timeout)

        if response.status_code == 404:
            return None
        response.raise_for_status()

        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, dict):
            return None

        result = {
            "meta": payload.get("meta") if isinstance(payload.get("meta"), dict) else {},
            "data": data,
        }
        self._cache.set(cache_key, result)
        return result
