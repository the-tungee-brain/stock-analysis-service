from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime
from threading import Lock
from typing import Any

import pandas as pd
import yfinance as yf

from app.broker.fiscal_period import (
    fiscal_quarter_and_year,
    fiscal_quarter_and_year_for_earnings_report,
    fiscal_year_end_month_from_info,
    format_fiscal_period,
)

logger = logging.getLogger(__name__)


class YFinanceAdapter:
    PEER_INFO_KEYS = ("recommendedSymbols", "recommended_symbols")
    INFO_TTL_SECONDS = int(os.getenv("YFINANCE_INFO_CACHE_TTL_SECONDS", "900"))
    HISTORY_TTL_SECONDS = int(os.getenv("YFINANCE_HISTORY_CACHE_TTL_SECONDS", "300"))
    EARNINGS_TTL_SECONDS = int(os.getenv("YFINANCE_EARNINGS_CACHE_TTL_SECONDS", "3600"))

    def __init__(self) -> None:
        self._info_cache: dict[str, tuple[float, dict]] = {}
        self._history_cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._earnings_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self._lock = Lock()

    def _get_cached(self, cache: dict, key: str, ttl_seconds: int):
        with self._lock:
            entry = cache.get(key)
            if entry is None:
                return None
            if time.time() - entry[0] >= ttl_seconds:
                del cache[key]
                return None
            return entry[1]

    def _set_cached(self, cache: dict, key: str, value) -> None:
        with self._lock:
            cache[key] = (time.time(), value)

    def get_daily_closes_1y(self, symbol: str) -> pd.Series:
        hist = self.get_history(symbol, period="1y", interval="1d")
        return hist["Close"] if "Close" in hist.columns else pd.Series(dtype=float)

    def get_ticker_info(self, symbol: str) -> dict:
        symbol_upper = symbol.strip().upper()
        cached = self._get_cached(self._info_cache, symbol_upper, self.INFO_TTL_SECONDS)
        if cached is not None:
            return dict(cached)

        ticker = yf.Ticker(symbol_upper)
        info = ticker.info or {}
        self._set_cached(self._info_cache, symbol_upper, info)
        return info

    def get_history(self, symbol: str, *, period: str, interval: str) -> pd.DataFrame:
        symbol_upper = symbol.strip().upper()
        cache_key = f"{symbol_upper}|{period}|{interval}"
        cached = self._get_cached(
            self._history_cache,
            cache_key,
            self.HISTORY_TTL_SECONDS,
        )
        if cached is not None:
            return cached.copy()

        ticker = yf.Ticker(symbol_upper)
        hist = ticker.history(period=period, interval=interval)
        self._set_cached(self._history_cache, cache_key, hist)
        return hist.copy()

    def get_stock_chart_payload(
        self,
        symbol: str,
        *,
        period: str = "1mo",
        interval: str = "1d",
    ) -> dict:
        symbol_upper = symbol.strip().upper()
        hist = self.get_history(symbol_upper, period=period, interval=interval)
        if hist.empty:
            raise ValueError("No data found")

        index_is_datetime = isinstance(hist.index, pd.DatetimeIndex)
        if index_is_datetime:
            date_values = hist.index
        else:
            date_values = pd.to_datetime(hist.index, errors="coerce")
            if date_values.isna().all():
                date_values = hist.index

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [column for column in required_cols if column not in hist.columns]
        if missing:
            raise RuntimeError(
                f"Missing expected OHLCV columns: {', '.join(missing)}"
            )

        data = []
        for index, (_, row) in enumerate(hist.iterrows()):
            date_value = date_values[index]
            if isinstance(date_value, pd.Timestamp):
                date_str = date_value.isoformat()
            else:
                date_str = str(date_value)

            data.append(
                {
                    "date": date_str,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                }
            )

        info = self.get_ticker_info(symbol_upper)
        return {
            "symbol": symbol_upper,
            "name": info.get("longName", symbol_upper),
            "currency": info.get("currency", "USD"),
            "data": data,
        }

    def get_recommended_peers(self, symbol: str, *, limit: int = 10) -> list[str]:
        info = self.get_ticker_info(symbol=symbol)
        if not info:
            return []

        peers: list[str] = []
        seen: set[str] = set()
        symbol_upper = symbol.strip().upper()

        for key in self.PEER_INFO_KEYS:
            raw = info.get(key)
            if not isinstance(raw, list):
                continue
            for item in raw:
                if not isinstance(item, str):
                    continue
                peer = item.strip().upper()
                if not peer or peer == symbol_upper or peer in seen:
                    continue
                seen.add(peer)
                peers.append(peer)
                if len(peers) >= limit:
                    return peers

        return peers

    def get_earnings_bundle(self, symbol: str, *, limit: int = 8) -> dict[str, Any]:
        """
        Cached earnings payload for list/detail builders.

        Keys: surprises (list[dict]), upcoming (dict|None), revenue_by_period (dict[str, float]).
        """
        symbol_upper = symbol.strip().upper()
        cache_key = f"{symbol_upper}|{limit}"
        cached = self._get_cached(
            self._earnings_cache,
            cache_key,
            self.EARNINGS_TTL_SECONDS,
        )
        if cached is not None:
            return dict(cached)

        info = self.get_ticker_info(symbol_upper)
        fy_end_month = fiscal_year_end_month_from_info(info)
        ticker = yf.Ticker(symbol_upper)

        surprises = self._fetch_earnings_surprises(
            ticker,
            limit=limit,
            fiscal_year_end_month=fy_end_month,
        )
        revenue_by_period = self._fetch_quarterly_revenue(ticker)
        reported_periods = {
            (item["quarter"], item["year"])
            for item in surprises
            if item.get("actual") is not None
            and item.get("quarter") is not None
            and item.get("year") is not None
        }
        latest_reported = self._latest_reported_date(surprises)
        upcoming = self._fetch_upcoming_earnings(
            ticker,
            info=info,
            fiscal_year_end_month=fy_end_month,
            reported_periods=reported_periods,
            latest_reported_date=latest_reported,
        )

        bundle = {
            "surprises": surprises,
            "upcoming": upcoming,
            "revenue_by_period": revenue_by_period,
        }
        self._set_cached(self._earnings_cache, cache_key, bundle)
        return bundle

    def _fetch_earnings_surprises(
        self,
        ticker: yf.Ticker,
        *,
        limit: int,
        fiscal_year_end_month: int | None,
    ) -> list[dict[str, Any]]:
        try:
            history = ticker.get_earnings_history()
        except Exception:
            logger.exception("yfinance earnings_history failed")
            return []
        if history is None or history.empty:
            return []

        rows: list[dict[str, Any]] = []
        for period_end, row in history.iterrows():
            report_date = self._index_to_date(period_end)
            if report_date is None:
                continue
            quarter, year = fiscal_quarter_and_year(
                report_date,
                fiscal_year_end_month=fiscal_year_end_month,
            )
            actual = self._optional_float(row.get("epsActual"))
            estimate = self._optional_float(row.get("epsEstimate"))
            surprise_pct = self._normalize_surprise_percent(row.get("surprisePercent"))
            rows.append(
                {
                    "period": report_date.isoformat(),
                    "quarter": quarter,
                    "year": year,
                    "fiscalPeriod": format_fiscal_period(quarter, year),
                    "actual": actual,
                    "estimate": estimate,
                    "surprisePercent": surprise_pct,
                }
            )

        rows.sort(key=lambda item: item["period"], reverse=True)
        return rows[:limit]

    def _fetch_upcoming_earnings(
        self,
        ticker: yf.Ticker,
        *,
        info: dict,
        fiscal_year_end_month: int | None,
        reported_periods: set[tuple[int, int]],
        latest_reported_date: date | None,
    ) -> dict[str, Any] | None:
        calendar = getattr(ticker, "calendar", None)
        if not isinstance(calendar, dict) or not calendar:
            return None

        report_date = self._next_earnings_date(
            calendar.get("Earnings Date"),
            after=latest_reported_date,
        )
        if report_date is None:
            return None

        quarter, year = fiscal_quarter_and_year_for_earnings_report(
            report_date,
            fiscal_year_end_month=fiscal_year_end_month,
        )
        if quarter is not None and year is not None:
            if (quarter, year) in reported_periods:
                return None

        timing = self._timing_from_info(info, report_date=report_date)
        return {
            "period": report_date.isoformat(),
            "quarter": quarter,
            "year": year,
            "fiscalPeriod": format_fiscal_period(quarter, year),
            "estimate": self._optional_float(calendar.get("Earnings Average")),
            "revenueEstimate": self._optional_float(calendar.get("Revenue Average")),
            "timing": timing,
        }

    def _fetch_quarterly_revenue(self, ticker: yf.Ticker) -> dict[str, float]:
        try:
            income = ticker.get_income_stmt(freq="quarterly")
        except Exception:
            logger.debug("yfinance quarterly income_stmt unavailable", exc_info=True)
            return {}
        if income is None or income.empty:
            return {}

        revenue_row = None
        for label in ("Total Revenue", "Revenue"):
            if label in income.index:
                revenue_row = income.loc[label]
                break
        if revenue_row is None:
            return {}

        by_period: dict[str, float] = {}
        for column, value in revenue_row.items():
            period_end = self._index_to_date(column)
            if period_end is None:
                continue
            parsed = self._optional_float(value)
            if parsed is not None:
                by_period[period_end.isoformat()] = parsed
        return by_period

    @staticmethod
    def _latest_reported_date(surprises: list[dict[str, Any]]) -> date | None:
        dates: list[date] = []
        for item in surprises:
            if item.get("actual") is None:
                continue
            parsed = YFinanceAdapter._period_to_date(item.get("period"))
            if parsed is not None:
                dates.append(parsed)
        return max(dates) if dates else None

    @staticmethod
    def _period_to_date(period: Any) -> date | None:
        if not period:
            return None
        try:
            return date.fromisoformat(str(period)[:10])
        except ValueError:
            return None

    @staticmethod
    def _next_earnings_date(
        raw: Any,
        *,
        after: date | None = None,
    ) -> date | None:
        if raw is None:
            return None
        candidates = raw if isinstance(raw, list) else [raw]
        today = date.today()
        min_date = today
        if after is not None and after > min_date:
            min_date = after
        parsed_dates: list[date] = []
        for item in candidates:
            parsed = YFinanceAdapter._index_to_date(item)
            if parsed is None or parsed <= min_date:
                continue
            if after is not None and (parsed - after).days <= 45:
                continue
            parsed_dates.append(parsed)
        if not parsed_dates:
            return None
        # Yahoo may list a near-term stale date and the real next report further out.
        return max(parsed_dates)

    @staticmethod
    def _timing_from_info(info: dict, *, report_date: date) -> str | None:
        for key in ("earningsTimestamp", "earningsTimestampStart"):
            raw = info.get(key)
            if raw is None:
                continue
            try:
                ts = datetime.fromtimestamp(float(raw))
            except (TypeError, ValueError, OSError, OverflowError):
                continue
            if ts.date() == report_date:
                if ts.hour < 12:
                    return "bmo"
                return "amc"
        return None

    @staticmethod
    def _index_to_date(value: Any) -> date | None:
        if value is None:
            return None
        if isinstance(value, pd.Timestamp):
            return value.date()
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        try:
            parsed = pd.Timestamp(value)
            if pd.isna(parsed):
                return None
            return parsed.date()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_surprise_percent(value: Any) -> float | None:
        parsed = YFinanceAdapter._optional_float(value)
        if parsed is None:
            return None
        if abs(parsed) <= 1.5:
            return parsed * 100.0
        return parsed
