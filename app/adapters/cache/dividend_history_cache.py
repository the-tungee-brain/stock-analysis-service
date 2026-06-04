import os
from typing import Optional

import redis

from app.models.dividend_research_models import DividendHistoryContext
from app.core.latency_observability import observe_dependency, record_dependency_latency


class DividendHistoryCache:
    DEFAULT_TTL_SECONDS = 600

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "dividend:history",
        ttl_seconds: int | None = None,
    ) -> None:
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds or int(
            os.getenv(
                "DIVIDEND_HISTORY_CACHE_TTL_SECONDS",
                str(self.DEFAULT_TTL_SECONDS),
            )
        )

    @staticmethod
    def build_cache_key(
        *,
        shares: float,
        investment_usd: float | None,
        share_price: float | None,
        reinvest_dividends: bool,
        price_cagr_pct: float | None,
        project_years: int | None,
        dividend_cagr_pct: float | None,
        history_start_year: int | None = None,
        annual_contribution_usd: float = 0.0,
    ) -> str:
        parts = [f"sh:{shares:.2f}"]

        if investment_usd is not None and investment_usd > 0:
            parts.append(f"inv:{investment_usd:.2f}")
        if share_price is not None and share_price > 0:
            parts.append(f"px:{share_price:.2f}")

        parts.append(f"years:{project_years or 10}")
        parts.append("drip" if reinvest_dividends else "cash")

        if price_cagr_pct is not None:
            parts.append(f"pcagr:{price_cagr_pct}")
        else:
            parts.append("pcagr:auto")

        if dividend_cagr_pct is not None:
            parts.append(f"dcagr:{dividend_cagr_pct}")
        else:
            parts.append("dcagr:auto")

        if history_start_year is not None:
            parts.append(f"hist:{history_start_year}")
        else:
            parts.append("hist:auto")

        if annual_contribution_usd > 0:
            parts.append(f"contrib:{annual_contribution_usd:.2f}")

        return "|".join(parts)

    def _redis_key(self, symbol: str, cache_key: str) -> str:
        return f"{self.key_prefix}:{symbol.strip().upper()}:{cache_key}"

    def get(self, symbol: str, cache_key: str) -> Optional[DividendHistoryContext]:
        with observe_dependency("redis"):
            raw = self.redis_client.get(self._redis_key(symbol, cache_key))
        if not raw:
            record_dependency_latency("dividend_history_cache", 0.0, cache_status="miss")
            return None
        record_dependency_latency("dividend_history_cache", 0.0, cache_status="hit")
        return DividendHistoryContext.model_validate_json(raw)

    def put(
        self,
        symbol: str,
        cache_key: str,
        context: DividendHistoryContext,
    ) -> None:
        with observe_dependency("redis"):
            self.redis_client.setex(
                self._redis_key(symbol, cache_key),
                self.ttl_seconds,
                context.model_dump_json(),
            )
