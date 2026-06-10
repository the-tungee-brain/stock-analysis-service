from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import oracledb

from app.core.latency_observability import observe_dependency
from app.http.json_sanitizer import sanitize_json_value
from app.models.provider_symbol_profile_models import (
    ProviderSymbolProfile,
    ProviderSymbolProfileMetadata,
)
from app.utils.dividend_yield import normalize_dividend_yield_pct


class ProviderSymbolProfileAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "PROVIDER_SYMBOL_PROFILE"
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with observe_dependency("oracle"):
            con = self.client.acquire()
            try:
                cur = con.cursor()
                self._ensure_table(cur)
                for column_name, column_type in self._required_additive_columns():
                    self._ensure_column(cur, column_name, column_type)
                con.commit()
            finally:
                self.client.release(con)

    def _ensure_table(self, cur: oracledb.Cursor) -> None:
        cur.execute(
            """
            SELECT 1
            FROM user_tables
            WHERE table_name = :table_name
            """,
            {"table_name": self.table_name},
        )
        if cur.fetchone() is not None:
            return
        try:
            cur.execute(
                f"""
                CREATE TABLE {self.table_name} (
                    provider        VARCHAR2(32) NOT NULL,
                    symbol          VARCHAR2(32) NOT NULL,
                    status          VARCHAR2(24) DEFAULT 'available' NOT NULL,
                    fetched_at      TIMESTAMP WITH TIME ZONE NOT NULL,
                    updated_at      TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP NOT NULL,
                    name            VARCHAR2(512),
                    currency        VARCHAR2(16),
                    exchange_name   VARCHAR2(128),
                    quote_type      VARCHAR2(64),
                    asset_type      VARCHAR2(64),
                    sector          VARCHAR2(256),
                    industry        VARCHAR2(256),
                    country         VARCHAR2(128),
                    website         VARCHAR2(1024),
                    current_price   NUMBER,
                    previous_close  NUMBER,
                    market_cap      NUMBER,
                    total_assets    NUMBER,
                    volume          NUMBER,
                    avg_volume      NUMBER,
                    trailing_pe     NUMBER,
                    forward_pe      NUMBER,
                    price_to_book   NUMBER,
                    dividend_yield  NUMBER,
                    dividend_yield_pct NUMBER,
                    raw_dividend_yield NUMBER,
                    raw_dividend_yield_source VARCHAR2(128),
                    dividend_rate   NUMBER,
                    expense_ratio   NUMBER,
                    beta            NUMBER,
                    raw_json        CLOB,
                    CONSTRAINT provider_symbol_profile_pk PRIMARY KEY (provider, symbol)
                )
                """
            )
        except oracledb.DatabaseError as exc:
            if not self._is_ddl_already_applied(exc):
                raise

    def _ensure_column(
        self,
        cur: oracledb.Cursor,
        column_name: str,
        column_type: str,
    ) -> None:
        cur.execute(
            """
            SELECT 1
            FROM user_tab_columns
            WHERE table_name = :table_name
              AND column_name = :column_name
            """,
            {"table_name": self.table_name, "column_name": column_name},
        )
        if cur.fetchone() is not None:
            return
        try:
            cur.execute(
                f"ALTER TABLE {self.table_name} ADD ({column_name.lower()} {column_type})"
            )
        except oracledb.DatabaseError as exc:
            if not self._is_ddl_already_applied(exc):
                raise

    @staticmethod
    def _required_additive_columns() -> tuple[tuple[str, str], ...]:
        return (
            ("DIVIDEND_YIELD_PCT", "NUMBER"),
            ("RAW_DIVIDEND_YIELD", "NUMBER"),
            ("RAW_DIVIDEND_YIELD_SOURCE", "VARCHAR2(128)"),
        )

    @staticmethod
    def _is_ddl_already_applied(exc: oracledb.DatabaseError) -> bool:
        error = exc.args[0] if exc.args else None
        code = getattr(error, "code", None)
        message = str(exc)
        return code in {955, 1430} or "ORA-00955" in message or "ORA-01430" in message

    @staticmethod
    def _read_lob(value: Any) -> Any:
        if hasattr(value, "read"):
            return value.read()
        return value

    @staticmethod
    def _normalize_provider(provider: str) -> str:
        return provider.strip().lower()

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return symbol.strip().upper()

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return parsed

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        return int(parsed)

    def get(self, provider: str, symbol: str) -> ProviderSymbolProfile | None:
        provider_key = self._normalize_provider(provider)
        symbol_key = self._normalize_symbol(symbol)
        if not provider_key or not symbol_key:
            return None

        sql = f"""
            SELECT provider, symbol, fetched_at, raw_json
            FROM {self.table_name}
            WHERE provider = :provider
              AND symbol = :symbol
              AND status = 'available'
        """

        with observe_dependency("oracle"):
            con = self.client.acquire()
            try:
                cur = con.cursor()
                cur.execute(sql, {"provider": provider_key, "symbol": symbol_key})
                row = cur.fetchone()
                if row is None:
                    return None
                raw_json = self._read_lob(row[3])
                payload = json.loads(raw_json or "{}")
                if not isinstance(payload, dict) or not payload:
                    return None
                return ProviderSymbolProfile(
                    provider=str(row[0]),
                    symbol=str(row[1]),
                    fetched_at=row[2],
                    raw_json=payload,
                )
            finally:
                self.client.release(con)

    def list_metadata(
        self,
        symbols: list[str],
    ) -> list[ProviderSymbolProfileMetadata]:
        symbol_keys = sorted(
            {
                self._normalize_symbol(symbol)
                for symbol in symbols
                if self._normalize_symbol(symbol)
            }
        )
        if not symbol_keys:
            return []

        placeholders = ", ".join(
            f":symbol_{index}" for index, _ in enumerate(symbol_keys)
        )
        sql = f"""
            SELECT provider, symbol, status, fetched_at,
                   sector, industry, asset_type, quote_type
            FROM {self.table_name}
            WHERE symbol IN ({placeholders})
        """
        params = {
            f"symbol_{index}": symbol
            for index, symbol in enumerate(symbol_keys)
        }

        with observe_dependency("oracle"):
            con = self.client.acquire()
            try:
                cur = con.cursor()
                cur.execute(sql, params)
                return [
                    ProviderSymbolProfileMetadata(
                        provider=str(row[0]),
                        symbol=str(row[1]),
                        status=str(row[2]),
                        fetched_at=row[3],
                        sector=self._optional_str(row[4]),
                        industry=self._optional_str(row[5]),
                        asset_type=self._optional_str(row[6]),
                        quote_type=self._optional_str(row[7]),
                    )
                    for row in cur.fetchall()
                ]
            finally:
                self.client.release(con)

    def upsert_success(
        self,
        provider: str,
        symbol: str,
        info: dict[str, Any],
        *,
        fetched_at: datetime | None = None,
    ) -> None:
        provider_key = self._normalize_provider(provider)
        symbol_key = self._normalize_symbol(symbol)
        if not provider_key or not symbol_key or not info:
            return

        normalized = self._normalized_fields(info)
        payload = sanitize_json_value(info)
        raw_json = json.dumps(payload, sort_keys=True, default=str)
        resolved_fetched_at = fetched_at or datetime.now(timezone.utc)

        sql = f"""
            MERGE INTO {self.table_name} t
            USING (
                SELECT
                    :provider       AS provider,
                    :symbol         AS symbol,
                    :status         AS status,
                    :fetched_at     AS fetched_at,
                    :name           AS name,
                    :currency       AS currency,
                    :exchange_name  AS exchange_name,
                    :quote_type     AS quote_type,
                    :asset_type     AS asset_type,
                    :sector         AS sector,
                    :industry       AS industry,
                    :country        AS country,
                    :website        AS website,
                    :current_price  AS current_price,
                    :previous_close AS previous_close,
                    :market_cap     AS market_cap,
                    :total_assets   AS total_assets,
                    :volume         AS volume,
                    :avg_volume     AS avg_volume,
                    :trailing_pe    AS trailing_pe,
                    :forward_pe     AS forward_pe,
                    :price_to_book  AS price_to_book,
                    :dividend_yield AS dividend_yield,
                    :dividend_yield_pct AS dividend_yield_pct,
                    :raw_dividend_yield AS raw_dividend_yield,
                    :raw_dividend_yield_source AS raw_dividend_yield_source,
                    :dividend_rate  AS dividend_rate,
                    :expense_ratio  AS expense_ratio,
                    :beta           AS beta,
                    :raw_json       AS raw_json
                FROM dual
            ) s
            ON (t.provider = s.provider AND t.symbol = s.symbol)
            WHEN MATCHED THEN
                UPDATE SET
                    t.status         = s.status,
                    t.fetched_at     = s.fetched_at,
                    t.updated_at     = systimestamp,
                    t.name           = s.name,
                    t.currency       = s.currency,
                    t.exchange_name  = s.exchange_name,
                    t.quote_type     = s.quote_type,
                    t.asset_type     = s.asset_type,
                    t.sector         = s.sector,
                    t.industry       = s.industry,
                    t.country        = s.country,
                    t.website        = s.website,
                    t.current_price  = s.current_price,
                    t.previous_close = s.previous_close,
                    t.market_cap     = s.market_cap,
                    t.total_assets   = s.total_assets,
                    t.volume         = s.volume,
                    t.avg_volume     = s.avg_volume,
                    t.trailing_pe    = s.trailing_pe,
                    t.forward_pe     = s.forward_pe,
                    t.price_to_book  = s.price_to_book,
                    t.dividend_yield = s.dividend_yield,
                    t.dividend_yield_pct = s.dividend_yield_pct,
                    t.raw_dividend_yield = s.raw_dividend_yield,
                    t.raw_dividend_yield_source = s.raw_dividend_yield_source,
                    t.dividend_rate  = s.dividend_rate,
                    t.expense_ratio  = s.expense_ratio,
                    t.beta           = s.beta,
                    t.raw_json       = s.raw_json
            WHEN NOT MATCHED THEN
                INSERT (
                    provider,
                    symbol,
                    status,
                    fetched_at,
                    updated_at,
                    name,
                    currency,
                    exchange_name,
                    quote_type,
                    asset_type,
                    sector,
                    industry,
                    country,
                    website,
                    current_price,
                    previous_close,
                    market_cap,
                    total_assets,
                    volume,
                    avg_volume,
                    trailing_pe,
                    forward_pe,
                    price_to_book,
                    dividend_yield,
                    dividend_yield_pct,
                    raw_dividend_yield,
                    raw_dividend_yield_source,
                    dividend_rate,
                    expense_ratio,
                    beta,
                    raw_json
                )
                VALUES (
                    s.provider,
                    s.symbol,
                    s.status,
                    s.fetched_at,
                    systimestamp,
                    s.name,
                    s.currency,
                    s.exchange_name,
                    s.quote_type,
                    s.asset_type,
                    s.sector,
                    s.industry,
                    s.country,
                    s.website,
                    s.current_price,
                    s.previous_close,
                    s.market_cap,
                    s.total_assets,
                    s.volume,
                    s.avg_volume,
                    s.trailing_pe,
                    s.forward_pe,
                    s.price_to_book,
                    s.dividend_yield,
                    s.dividend_yield_pct,
                    s.raw_dividend_yield,
                    s.raw_dividend_yield_source,
                    s.dividend_rate,
                    s.expense_ratio,
                    s.beta,
                    s.raw_json
                )
        """
        params = {
            "provider": provider_key,
            "symbol": symbol_key,
            "status": "available",
            "fetched_at": resolved_fetched_at,
            "raw_json": raw_json,
            **normalized,
        }

        with observe_dependency("oracle"):
            con = self.client.acquire()
            try:
                cur = con.cursor()
                cur.setinputsizes(raw_json=oracledb.DB_TYPE_CLOB)
                cur.execute(sql, params)
                con.commit()
            finally:
                self.client.release(con)

    def upsert_success_many(
        self,
        provider: str,
        items: list[tuple[str, dict[str, Any]]],
        *,
        fetched_at: datetime | None = None,
    ) -> int:
        provider_key = self._normalize_provider(provider)
        if not provider_key or not items:
            return 0

        resolved_fetched_at = fetched_at or datetime.now(timezone.utc)
        params: list[dict[str, Any]] = []
        for symbol, info in items:
            symbol_key = self._normalize_symbol(symbol)
            if not symbol_key or not info:
                continue
            payload = sanitize_json_value(info)
            params.append(
                {
                    "provider": provider_key,
                    "symbol": symbol_key,
                    "status": "available",
                    "fetched_at": resolved_fetched_at,
                    "raw_json": json.dumps(payload, sort_keys=True, default=str),
                    **self._normalized_fields(info),
                }
            )
        if not params:
            return 0

        sql = f"""
            MERGE INTO {self.table_name} t
            USING (
                SELECT
                    :provider       AS provider,
                    :symbol         AS symbol,
                    :status         AS status,
                    :fetched_at     AS fetched_at,
                    :name           AS name,
                    :currency       AS currency,
                    :exchange_name  AS exchange_name,
                    :quote_type     AS quote_type,
                    :asset_type     AS asset_type,
                    :sector         AS sector,
                    :industry       AS industry,
                    :country        AS country,
                    :website        AS website,
                    :current_price  AS current_price,
                    :previous_close AS previous_close,
                    :market_cap     AS market_cap,
                    :total_assets   AS total_assets,
                    :volume         AS volume,
                    :avg_volume     AS avg_volume,
                    :trailing_pe    AS trailing_pe,
                    :forward_pe     AS forward_pe,
                    :price_to_book  AS price_to_book,
                    :dividend_yield AS dividend_yield,
                    :dividend_yield_pct AS dividend_yield_pct,
                    :raw_dividend_yield AS raw_dividend_yield,
                    :raw_dividend_yield_source AS raw_dividend_yield_source,
                    :dividend_rate  AS dividend_rate,
                    :expense_ratio  AS expense_ratio,
                    :beta           AS beta,
                    :raw_json       AS raw_json
                FROM dual
            ) s
            ON (t.provider = s.provider AND t.symbol = s.symbol)
            WHEN MATCHED THEN
                UPDATE SET
                    t.status         = s.status,
                    t.fetched_at     = s.fetched_at,
                    t.updated_at     = systimestamp,
                    t.name           = s.name,
                    t.currency       = s.currency,
                    t.exchange_name  = s.exchange_name,
                    t.quote_type     = s.quote_type,
                    t.asset_type     = s.asset_type,
                    t.sector         = s.sector,
                    t.industry       = s.industry,
                    t.country        = s.country,
                    t.website        = s.website,
                    t.current_price  = s.current_price,
                    t.previous_close = s.previous_close,
                    t.market_cap     = s.market_cap,
                    t.total_assets   = s.total_assets,
                    t.volume         = s.volume,
                    t.avg_volume     = s.avg_volume,
                    t.trailing_pe    = s.trailing_pe,
                    t.forward_pe     = s.forward_pe,
                    t.price_to_book  = s.price_to_book,
                    t.dividend_yield = s.dividend_yield,
                    t.dividend_yield_pct = s.dividend_yield_pct,
                    t.raw_dividend_yield = s.raw_dividend_yield,
                    t.raw_dividend_yield_source = s.raw_dividend_yield_source,
                    t.dividend_rate  = s.dividend_rate,
                    t.expense_ratio  = s.expense_ratio,
                    t.beta           = s.beta,
                    t.raw_json       = s.raw_json
            WHEN NOT MATCHED THEN
                INSERT (
                    provider, symbol, status, fetched_at, updated_at, name, currency,
                    exchange_name, quote_type, asset_type, sector, industry, country,
                    website, current_price, previous_close, market_cap, total_assets,
                    volume, avg_volume, trailing_pe, forward_pe, price_to_book,
                    dividend_yield, dividend_yield_pct, raw_dividend_yield,
                    raw_dividend_yield_source, dividend_rate, expense_ratio, beta, raw_json
                )
                VALUES (
                    s.provider, s.symbol, s.status, s.fetched_at, systimestamp,
                    s.name, s.currency, s.exchange_name, s.quote_type, s.asset_type,
                    s.sector, s.industry, s.country, s.website, s.current_price,
                    s.previous_close, s.market_cap, s.total_assets, s.volume,
                    s.avg_volume, s.trailing_pe, s.forward_pe, s.price_to_book,
                    s.dividend_yield, s.dividend_yield_pct, s.raw_dividend_yield,
                    s.raw_dividend_yield_source, s.dividend_rate, s.expense_ratio,
                    s.beta, s.raw_json
                )
        """

        with observe_dependency("oracle"):
            con = self.client.acquire()
            try:
                cur = con.cursor()
                cur.setinputsizes(raw_json=oracledb.DB_TYPE_CLOB)
                cur.executemany(sql, params)
                con.commit()
            finally:
                self.client.release(con)
        return len(params)

    def _normalized_fields(self, info: dict[str, Any]) -> dict[str, Any]:
        raw_dividend_yield = self._optional_float(info.get("dividendYield"))
        asset_type = self._optional_str(info.get("typeDisp") or info.get("quoteType"))
        normalized_dividend_yield = normalize_dividend_yield_pct(
            raw_dividend_yield,
            asset_type=asset_type,
            source="yfinance.info.dividendYield",
        )
        return {
            "name": self._optional_str(info.get("longName") or info.get("shortName")),
            "currency": self._optional_str(info.get("currency")),
            "exchange_name": self._optional_str(
                info.get("exchange") or info.get("fullExchangeName")
            ),
            "quote_type": self._optional_str(info.get("quoteType")),
            "asset_type": asset_type,
            "sector": self._optional_str(info.get("sector")),
            "industry": self._optional_str(info.get("industry")),
            "country": self._optional_str(info.get("country")),
            "website": self._optional_str(info.get("website")),
            "current_price": self._optional_float(
                info.get("currentPrice") or info.get("regularMarketPrice")
            ),
            "previous_close": self._optional_float(
                info.get("regularMarketPreviousClose") or info.get("previousClose")
            ),
            "market_cap": self._optional_int(info.get("marketCap")),
            "total_assets": self._optional_int(info.get("totalAssets")),
            "volume": self._optional_int(
                info.get("volume") or info.get("regularMarketVolume")
            ),
            "avg_volume": self._optional_int(
                info.get("averageVolume") or info.get("averageDailyVolume10Day")
            ),
            "trailing_pe": self._optional_float(info.get("trailingPE")),
            "forward_pe": self._optional_float(info.get("forwardPE")),
            "price_to_book": self._optional_float(info.get("priceToBook")),
            "dividend_yield": raw_dividend_yield,
            "dividend_yield_pct": (
                normalized_dividend_yield.dividend_yield_pct
                if normalized_dividend_yield is not None
                else None
            ),
            "raw_dividend_yield": raw_dividend_yield,
            "raw_dividend_yield_source": (
                "yfinance.info.dividendYield"
                if raw_dividend_yield is not None
                else None
            ),
            "dividend_rate": self._optional_float(
                info.get("dividendRate") or info.get("trailingAnnualDividendRate")
            ),
            "expense_ratio": self._optional_float(
                info.get("annualReportExpenseRatio")
            ),
            "beta": self._optional_float(info.get("beta")),
        }
