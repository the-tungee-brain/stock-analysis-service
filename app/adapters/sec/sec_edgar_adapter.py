import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests


class TTLCache:
    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if not entry:
            return None
        if datetime.now(timezone.utc) > entry["expires"]:
            del self._store[key]
            return None
        return entry["value"]

    def set(self, key: str, value: Any) -> None:
        self._store[key] = {
            "value": value,
            "expires": datetime.now(timezone.utc) + timedelta(seconds=self._ttl),
        }


class SecEdgarAdapter:
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    def __init__(self, session: requests.Session, user_agent: str) -> None:
        self.session = session
        self.user_agent = user_agent
        self._ticker_cache = TTLCache(ttl_seconds=86_400)
        self._submissions_cache = TTLCache(ttl_seconds=21_600)
        self._facts_cache = TTLCache(ttl_seconds=21_600)
        self._last_request_at = 0.0

    @classmethod
    def from_env(cls, session: requests.Session) -> "SecEdgarAdapter":
        user_agent = os.getenv(
            "SEC_USER_AGENT",
            "PowerPocket stock-analysis contact@example.com",
        )
        return cls(session=session, user_agent=user_agent)

    @staticmethod
    def format_cik(cik: int | str) -> str:
        return str(int(cik)).zfill(10)

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

    def _throttle(self) -> None:
        # SEC fair-access guidance: stay at or below ~10 requests/second.
        min_interval = 0.11
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _get_json(self, url: str, cache: TTLCache | None = None) -> Any:
        if cache is not None:
            cached = cache.get(url)
            if cached is not None:
                return cached

        self._throttle()
        response = self.session.get(url, headers=self._headers(), timeout=30)
        response.raise_for_status()
        data = response.json()

        if cache is not None:
            cache.set(url, data)
        return data

    def get_company_tickers(self) -> dict[str, Any]:
        return self._get_json(self.TICKERS_URL, self._ticker_cache)

    def get_submissions(self, cik: int | str) -> dict[str, Any]:
        cik_str = self.format_cik(cik)
        url = self.SUBMISSIONS_URL.format(cik=cik_str)
        return self._get_json(url, self._submissions_cache)

    def get_company_facts(self, cik: int | str) -> dict[str, Any]:
        cik_str = self.format_cik(cik)
        url = self.COMPANY_FACTS_URL.format(cik=cik_str)
        return self._get_json(url, self._facts_cache)
