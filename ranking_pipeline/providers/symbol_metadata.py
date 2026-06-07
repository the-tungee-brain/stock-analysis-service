"""Provider-first symbol metadata resolver for ranking universe screening."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

import oracledb
import yfinance as yf

from app.adapters.market.provider_symbol_profile_adapter import ProviderSymbolProfileAdapter
from app.adapters.market.yfinance_bootstrap import configure_yfinance, yfinance_fetch_lock
from ranking_pipeline.storage.oracle_screening import build_oracle_pool

logger = logging.getLogger(__name__)


def _truthy_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _optional_int(value: Any) -> int | None:
    parsed = _optional_float(value)
    return int(parsed) if parsed is not None else None


@dataclass(frozen=True, slots=True)
class SymbolMetadata:
    symbol: str
    market_cap: float | None = None
    name: str | None = None
    exchange_name: str | None = None
    sector: str | None = None
    asset_type: str | None = None
    quote_type: str | None = None
    source: str = "missing"

    def missing_fields(self, required_fields: Iterable[str]) -> list[str]:
        return [field for field in required_fields if getattr(self, field, None) is None]


class OracleProviderSymbolMetadataStore:
    def __init__(self, pool: oracledb.ConnectionPool, *, provider: str = "yahoo") -> None:
        self._pool = pool
        self._provider = provider.strip().lower()

    def get_many(self, symbols: list[str]) -> dict[str, SymbolMetadata]:
        symbol_keys = list(dict.fromkeys(sym.strip().upper() for sym in symbols if sym.strip()))
        if not symbol_keys:
            return {}
        out: dict[str, SymbolMetadata] = {}
        for start in range(0, len(symbol_keys), 500):
            chunk = symbol_keys[start : start + 500]
            out.update(self._get_chunk(chunk))
        return out

    def _get_chunk(self, symbols: list[str]) -> dict[str, SymbolMetadata]:
        params: dict[str, Any] = {"provider": self._provider}
        placeholders: list[str] = []
        for idx, symbol in enumerate(symbols):
            key = f"symbol_{idx}"
            params[key] = symbol
            placeholders.append(f":{key}")
        sql = f"""
            SELECT symbol, market_cap, name, exchange_name, sector, asset_type, quote_type
            FROM PROVIDER_SYMBOL_PROFILE
            WHERE provider = :provider
              AND status = 'available'
              AND symbol IN ({", ".join(placeholders)})
        """
        with self._pool.acquire() as conn:
            rows = conn.cursor().execute(sql, params).fetchall()
        return {
            str(row[0]).strip().upper(): SymbolMetadata(
                symbol=str(row[0]).strip().upper(),
                market_cap=_optional_float(row[1]),
                name=str(row[2]).strip() if row[2] else None,
                exchange_name=str(row[3]).strip() if row[3] else None,
                sector=str(row[4]).strip() if row[4] else None,
                asset_type=str(row[5]).strip() if row[5] else None,
                quote_type=str(row[6]).strip() if row[6] else None,
                source=f"oracle:{self._provider}",
            )
            for row in rows
        }


class ProviderFirstSymbolMetadataResolver:
    def __init__(
        self,
        store: OracleProviderSymbolMetadataStore,
        *,
        profile_writer: ProviderSymbolProfileAdapter | None = None,
        allow_yfinance_fallback: bool | None = None,
    ) -> None:
        self._store = store
        self._profile_writer = profile_writer
        self._allow_yfinance_fallback = (
            _truthy_env("RANKING_YFINANCE_METADATA_FALLBACK", default=True)
            if allow_yfinance_fallback is None
            else allow_yfinance_fallback
        )

    def resolve_many(
        self,
        symbols: list[str],
        *,
        required_fields: Iterable[str] = ("market_cap",),
    ) -> dict[str, SymbolMetadata]:
        symbol_keys = list(dict.fromkeys(sym.strip().upper() for sym in symbols if sym.strip()))
        found = self._store.get_many(symbol_keys)
        resolved: dict[str, SymbolMetadata] = {}
        for symbol in symbol_keys:
            metadata = found.get(symbol) or SymbolMetadata(symbol=symbol)
            missing = metadata.missing_fields(required_fields)
            if missing:
                if not self._allow_yfinance_fallback:
                    logger.warning(
                        "Provider metadata missing for %s fields=%s; yfinance fallback disabled",
                        symbol,
                        ",".join(missing),
                    )
                else:
                    logger.warning(
                        "Provider metadata missing for %s fields=%s; falling back to yfinance",
                        symbol,
                        ",".join(missing),
                    )
                    fallback = self._fetch_yfinance_metadata(symbol)
                    if fallback is not None:
                        metadata = fallback
            resolved[symbol] = metadata
        return resolved

    def _fetch_yfinance_metadata(self, symbol: str) -> SymbolMetadata | None:
        configure_yfinance()
        try:
            with yfinance_fetch_lock():
                info = yf.Ticker(symbol).info or {}
        except Exception as exc:
            logger.warning("yfinance metadata fallback failed for %s: %s", symbol, exc)
            return None
        if not info:
            return None
        if self._profile_writer is not None:
            try:
                self._profile_writer.upsert_success(
                    "yahoo",
                    symbol,
                    info,
                    fetched_at=datetime.now(timezone.utc),
                )
            except Exception:
                logger.warning("Provider metadata backfill write failed for %s", symbol, exc_info=True)
        return SymbolMetadata(
            symbol=symbol,
            market_cap=_optional_float(info.get("marketCap")),
            name=(info.get("longName") or info.get("shortName")),
            exchange_name=(info.get("exchange") or info.get("fullExchangeName")),
            sector=info.get("sector"),
            asset_type=(info.get("typeDisp") or info.get("quoteType")),
            quote_type=info.get("quoteType"),
            source="yfinance:fallback",
        )


def build_provider_first_symbol_metadata_resolver() -> ProviderFirstSymbolMetadataResolver:
    pool = build_oracle_pool()
    return ProviderFirstSymbolMetadataResolver(
        OracleProviderSymbolMetadataStore(pool),
        profile_writer=ProviderSymbolProfileAdapter(pool),
    )
