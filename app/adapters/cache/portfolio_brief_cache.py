from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import redis

from app.models.intelligence_models import PortfolioIntelligence
from app.models.schwab_models import Position, SchwabAccounts
from app.core.latency_observability import observe_dependency, record_dependency_latency


class PortfolioBriefCache:
    DEFAULT_TTL_SECONDS = 900
    VARIANT_FULL = "full"
    VARIANT_LIGHT = "light"

    def __init__(
        self,
        redis_client: redis.Redis,
        key_prefix: str = "portfolio:brief",
        ttl_seconds: int | None = None,
    ) -> None:
        self.redis_client = redis_client
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds or int(
            os.getenv(
                "PORTFOLIO_BRIEF_CACHE_TTL_SECONDS",
                str(self.DEFAULT_TTL_SECONDS),
            )
        )

    @staticmethod
    def fingerprint(
        positions: list[Position],
        account: SchwabAccounts,
    ) -> str:
        liquidation = account.securitiesAccount.currentBalances.liquidationValue
        parts: list[str] = [f"liq:{round(liquidation, 2)}"]

        for position in sorted(
            positions,
            key=lambda item: (
                item.instrument.underlyingSymbol or item.instrument.symbol or ""
            ).upper(),
        ):
            instrument = position.instrument
            symbol = (instrument.underlyingSymbol or instrument.symbol or "").upper()
            if not symbol:
                continue
            asset = instrument.assetType or "EQUITY"
            parts.append(
                f"{symbol}:{asset}:{round(position.longQuantity, 4)}:"
                f"{round(position.shortQuantity, 4)}:{round(abs(position.marketValue), 2)}"
            )

        digest = hashlib.sha256("|".join(parts).encode()).hexdigest()
        return digest[:20]

    def _redis_key(
        self,
        *,
        user_id: str,
        fingerprint: str,
        variant: str = VARIANT_FULL,
    ) -> str:
        return f"{self.key_prefix}:{variant}:{user_id}:{fingerprint}"

    def get(
        self,
        *,
        user_id: str,
        fingerprint: str,
        variant: str = VARIANT_FULL,
    ) -> Optional[PortfolioIntelligence]:
        with observe_dependency("redis"):
            raw = self.redis_client.get(
                self._redis_key(
                    user_id=user_id,
                    fingerprint=fingerprint,
                    variant=variant,
                )
            )
        if not raw:
            record_dependency_latency("portfolio_brief_cache", 0.0, cache_status="miss")
            return None
        record_dependency_latency("portfolio_brief_cache", 0.0, cache_status="hit")
        try:
            payload = raw.decode() if isinstance(raw, bytes) else raw
            return PortfolioIntelligence.model_validate_json(payload)
        except (TypeError, json.JSONDecodeError, ValueError):
            return None

    def put(
        self,
        *,
        user_id: str,
        fingerprint: str,
        brief: PortfolioIntelligence,
        variant: str = VARIANT_FULL,
    ) -> None:
        if self.ttl_seconds <= 0:
            return
        with observe_dependency("redis"):
            self.redis_client.setex(
                self._redis_key(
                    user_id=user_id,
                    fingerprint=fingerprint,
                    variant=variant,
                ),
                self.ttl_seconds,
                brief.model_dump_json(by_alias=True),
            )

    def invalidate_user(self, *, user_id: str) -> int:
        pattern = f"{self.key_prefix}:*:{user_id}:*"
        deleted = 0
        for key in self.redis_client.scan_iter(match=pattern):
            with observe_dependency("redis"):
                deleted += int(self.redis_client.delete(key))
        return deleted
