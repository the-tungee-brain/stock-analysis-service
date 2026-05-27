import os
import time
from threading import Lock

import pandas as pd
import yfinance as yf


class YFinanceAdapter:
    PEER_INFO_KEYS = ("recommendedSymbols", "recommended_symbols")
    INFO_TTL_SECONDS = int(os.getenv("YFINANCE_INFO_CACHE_TTL_SECONDS", "900"))
    HISTORY_TTL_SECONDS = int(os.getenv("YFINANCE_HISTORY_CACHE_TTL_SECONDS", "300"))

    def __init__(self) -> None:
        self._info_cache: dict[str, tuple[float, dict]] = {}
        self._history_cache: dict[str, tuple[float, pd.DataFrame]] = {}
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
