from __future__ import annotations

import logging
import os

import yfinance as yf

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.builders.finnhub_builder import FinnhubBuilder
from app.models.company_research_models import ResearchSnapshot

logger = logging.getLogger(__name__)

FINNHUB_STOCK_LOGO_URL = (
    "https://static2.finnhub.io/file/publicdatany/finnhubimage/stock_logo/{symbol}.png"
)


class CompanyProfileService:
    def __init__(
        self,
        finnhub_builder: FinnhubBuilder,
        yfinance_adapter: YFinanceAdapter | None = None,
    ):
        self.finnhub_builder = finnhub_builder
        self.yfinance_adapter = yfinance_adapter

    def get_snapshot(self, symbol: str) -> ResearchSnapshot:
        symbol_upper = symbol.strip().upper()

        snapshot = self._snapshot_from_yfinance(symbol_upper)
        if snapshot is not None:
            return snapshot

        snapshot = self._snapshot_from_finnhub(symbol_upper)
        if snapshot is not None:
            return snapshot

        raise ValueError(f"Unable to load company snapshot for {symbol_upper}")

    def get_peers(self, symbol: str) -> list[str]:
        symbol_upper = symbol.strip().upper()

        peers = self._peers_from_yfinance(symbol_upper)
        if peers:
            return peers

        try:
            peers = self.finnhub_builder.get_peers(symbol=symbol)
        except Exception:
            logger.warning("Finnhub peers unavailable for %s", symbol, exc_info=True)
            return []

        return [peer for peer in peers if peer != symbol_upper]

    def _peers_from_yfinance(self, symbol: str) -> list[str]:
        if self.yfinance_adapter is None:
            return []

        try:
            peers = self.yfinance_adapter.get_recommended_peers(symbol=symbol)
        except Exception:
            logger.warning(
                "Yahoo Finance peers unavailable for %s", symbol, exc_info=True
            )
            return []

        symbol_upper = symbol.upper()
        return [peer for peer in peers if peer != symbol_upper]

    def _snapshot_from_finnhub(self, symbol: str) -> ResearchSnapshot | None:
        try:
            profile = self.finnhub_builder.get_company_profile(symbol=symbol)
            if profile is None:
                return None
            quote = self.finnhub_builder.get_quote(symbol=symbol)
        except Exception:
            logger.warning("Finnhub snapshot unavailable for %s", symbol, exc_info=True)
            return None

        if profile is None or quote is None or getattr(quote, "c", None) in (None, 0):
            return None

        range_52w = self._format_52w_range(symbol)
        change_pct = self._compute_change_pct(
            current=getattr(quote, "c", None),
            prev_close=getattr(quote, "pc", None),
        )

        return ResearchSnapshot(
            symbol=profile.ticker or symbol,
            name=profile.name,
            sector=profile.finnhubIndustry,
            country=profile.country,
            price=quote.c,
            changePct=change_pct,
            marketCap=self._format_market_cap_millions(profile.marketCapitalization),
            range52w=range_52w,
            logo=self._normalize_logo_url(symbol, str(profile.logo)),
            weburl=profile.weburl,
        )

    def _snapshot_from_yfinance(self, symbol: str) -> ResearchSnapshot | None:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            history = ticker.history(period="5d", interval="1d")
        except Exception:
            logger.warning("Yahoo Finance snapshot unavailable for %s", symbol, exc_info=True)
            return None

        price = self._price_from_yfinance(info, history)
        if price is None:
            return None

        prev_close = self._previous_close_from_yfinance(info, history, price)
        change_pct = self._compute_change_pct(current=price, prev_close=prev_close)
        website = info.get("website") or f"https://finance.yahoo.com/quote/{symbol}"
        logo = self._normalize_logo_url(symbol, info.get("logo_url"))

        return ResearchSnapshot(
            symbol=symbol,
            name=(
                info.get("longName")
                or info.get("shortName")
                or symbol
            ),
            sector=info.get("sector") or info.get("industry") or "Unknown",
            country=info.get("country") or "Unknown",
            price=price,
            changePct=change_pct,
            marketCap=self._format_market_cap_absolute(info.get("marketCap")),
            range52w=self._format_52w_range(symbol),
            weburl=website,
            logo=logo,
        )

    @staticmethod
    def _price_from_yfinance(info: dict, history) -> float | None:
        if history is not None and not history.empty:
            return float(history["Close"].iloc[-1])

        for key in ("currentPrice", "regularMarketPrice", "previousClose"):
            value = info.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    @staticmethod
    def _previous_close_from_yfinance(info: dict, history, price: float) -> float | None:
        if history is not None and len(history.index) >= 2:
            return float(history["Close"].iloc[-2])

        previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        if previous_close is not None:
            try:
                return float(previous_close)
            except (TypeError, ValueError):
                pass
        return price

    def _format_52w_range(self, symbol: str) -> str | None:
        try:
            low_52w, high_52w = self.get_52w_range_yf(symbol=symbol)
        except Exception:
            return None
        return f"${low_52w:.0f} – ${high_52w:.0f}"

    def _compute_change_pct(
        self, current: float | None, prev_close: float | None
    ) -> float:
        if current is None or prev_close in (None, 0):
            return 0.0
        return (current / prev_close - 1.0) * 100.0

    def get_52w_range_yf(self, symbol: str) -> tuple[float, float]:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y", interval="1d")

        if hist.empty:
            raise ValueError(f"No historical data for {symbol}")

        high_52w = float(hist["High"].max())
        low_52w = float(hist["Low"].min())
        return low_52w, high_52w

    def format_52w_range(self, symbol: str) -> str:
        low, high = self.get_52w_range_yf(symbol)
        return f"${low:.2f} – ${high:.2f}"

    @staticmethod
    def _stock_logo_url(symbol: str) -> str:
        return FINNHUB_STOCK_LOGO_URL.format(symbol=symbol.strip().upper())

    @staticmethod
    def _normalize_logo_url(symbol: str, candidate: str | None) -> str:
        if candidate:
            value = candidate.strip()
            lower = value.lower()
            if lower.startswith(("http://", "https://")) and (
                ".png" in lower
                or ".jpg" in lower
                or ".jpeg" in lower
                or ".svg" in lower
                or ".webp" in lower
                or ".gif" in lower
                or "finnhubimage" in lower
                or "/logo" in lower
            ):
                return value
        return CompanyProfileService._stock_logo_url(symbol)

    @staticmethod
    def _format_market_cap_millions(mc: float) -> str:
        if mc >= 1_000_000:
            return f"{mc / 1_000_000:.1f}T"
        if mc >= 1_000:
            return f"{mc / 1_000:.1f}B"
        return f"{mc:.0f}M"

    @staticmethod
    def _format_market_cap_absolute(mc: float | None) -> str:
        if mc is None:
            return "N/A"
        try:
            value = float(mc)
        except (TypeError, ValueError):
            return "N/A"
        if value >= 1_000_000_000_000:
            return f"{value / 1_000_000_000_000:.1f}T"
        if value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}B"
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        return f"{value:.0f}"
