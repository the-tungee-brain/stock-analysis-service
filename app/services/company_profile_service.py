from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.builders.finnhub_builder import FinnhubBuilder
from app.models.company_research_models import ResearchSnapshot

if TYPE_CHECKING:
    from app.builders.ticker_symbol_builder import TickerSymbolBuilder

logger = logging.getLogger(__name__)

FINNHUB_STOCK_LOGO_URL = (
    "https://static2.finnhub.io/file/publicdatany/finnhubimage/stock_logo/{symbol}.png"
)

# Finnhub's CDN predates some ticker changes (e.g. FB → META).
FINNHUB_LOGO_SYMBOL_ALIASES: dict[str, str] = {
    "META": "FB",
}


class CompanyProfileService:
    def __init__(
        self,
        finnhub_builder: FinnhubBuilder,
        yfinance_adapter: YFinanceAdapter | None = None,
        ticker_symbol_builder: TickerSymbolBuilder | None = None,
    ):
        self.finnhub_builder = finnhub_builder
        self.yfinance_adapter = yfinance_adapter
        self.ticker_symbol_builder = ticker_symbol_builder

    def get_snapshot(self, symbol: str) -> ResearchSnapshot:
        symbol_upper = symbol.strip().upper()

        snapshot = self._snapshot_from_yfinance(symbol_upper)
        if snapshot is None:
            snapshot = self._snapshot_from_finnhub(symbol_upper)

        if snapshot is None:
            raise ValueError(f"Unable to load company snapshot for {symbol_upper}")

        return self._apply_ticker_logo(snapshot)

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
            logger.warning("Yahoo Finance peers unavailable for %s", symbol)
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
            logo=None,
            weburl=profile.weburl,
            dividendYieldPct=None,
            peRatio=None,
            volume=None,
            avgVolume=None,
            expenseRatioPct=None,
        )

    def _snapshot_from_yfinance(self, symbol: str) -> ResearchSnapshot | None:
        if self.yfinance_adapter is None:
            return None

        info = self.yfinance_adapter.get_ticker_info(symbol)
        history = self.yfinance_adapter.get_history(
            symbol, period="5d", interval="1d"
        )

        price = self._price_from_yfinance(info, history)
        if price is None:
            return None

        prev_close = self._previous_close_from_yfinance(info, history, price)
        change_pct = self._compute_change_pct(current=price, prev_close=prev_close)
        website = info.get("website") or f"https://finance.yahoo.com/quote/{symbol}"
        is_etf = self._is_etf_info(info)

        if is_etf:
            sector = self._etf_sector_label(info)
            country = self._etf_country_label(info)
            market_cap = self._format_market_cap_absolute(
                info.get("totalAssets") or info.get("marketCap")
            )
        else:
            sector = info.get("sector") or info.get("industry") or "Unknown"
            country = info.get("country") or "Unknown"
            market_cap = self._format_market_cap_absolute(info.get("marketCap"))

        key_stats = self._key_stats_from_yfinance(info, is_etf=is_etf)

        return ResearchSnapshot(
            symbol=symbol,
            name=(
                info.get("longName")
                or info.get("shortName")
                or symbol
            ),
            sector=sector,
            country=country,
            price=price,
            changePct=change_pct,
            marketCap=market_cap,
            range52w=self._format_52w_range(symbol),
            weburl=website,
            logo=None,
            **key_stats,
        )

    def _apply_ticker_logo(self, snapshot: ResearchSnapshot) -> ResearchSnapshot:
        logo = self._resolve_stock_logo(snapshot.symbol)
        if logo == snapshot.logo:
            return snapshot
        return snapshot.model_copy(update={"logo": logo})

    def _resolve_stock_logo(self, symbol: str) -> str | None:
        item = None
        if self.ticker_symbol_builder is not None:
            try:
                item = self.ticker_symbol_builder.get_by_symbol(symbol=symbol)
            except Exception:
                logger.warning(
                    "Ticker logo lookup failed for %s", symbol, exc_info=True
                )

        if item is not None and item.asset_type == "ETF":
            return None

        if item is not None and item.logo_url:
            return item.logo_url.strip()

        return self._finnhub_stock_logo_url(symbol)

    @staticmethod
    def _finnhub_stock_logo_url(symbol: str) -> str:
        symbol_upper = symbol.strip().upper()
        finnhub_symbol = FINNHUB_LOGO_SYMBOL_ALIASES.get(symbol_upper, symbol_upper)
        return FINNHUB_STOCK_LOGO_URL.format(symbol=finnhub_symbol)

    @staticmethod
    def _key_stats_from_yfinance(info: dict, *, is_etf: bool) -> dict:
        return {
            "dividendYieldPct": CompanyProfileService._normalize_dividend_yield_pct(
                info.get("dividendYield")
            ),
            "peRatio": CompanyProfileService._optional_float(info.get("trailingPE")),
            "volume": CompanyProfileService._optional_int(
                info.get("volume") or info.get("regularMarketVolume")
            ),
            "avgVolume": CompanyProfileService._optional_int(
                info.get("averageVolume")
                or info.get("averageDailyVolume10Day")
            ),
            "expenseRatioPct": (
                CompanyProfileService._normalize_expense_ratio_pct(info)
                if is_etf
                else None
            ),
        }

    @staticmethod
    def _optional_float(value) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if parsed != parsed:  # NaN
            return None
        return round(parsed, 2)

    @staticmethod
    def _optional_int(value) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            return None
        if parsed < 0:
            return None
        return parsed

    @staticmethod
    def _normalize_dividend_yield_pct(value) -> float | None:
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if parsed != parsed or parsed <= 0:
            return None
        if abs(parsed) < 1:
            return round(parsed * 100, 2)
        return round(parsed, 2)

    @staticmethod
    def _normalize_expense_ratio_pct(info: dict) -> float | None:
        annual = info.get("annualReportExpenseRatio")
        if isinstance(annual, (int, float)) and annual > 0:
            return round(abs(annual) * 100, 2)

        for key in ("netExpenseRatio", "expenseRatio"):
            value = info.get(key)
            if value is None or not isinstance(value, (int, float)) or value <= 0:
                continue
            abs_value = abs(float(value))
            if abs_value < 0.01:
                return round(abs_value * 100, 2)
            return round(abs_value, 2)
        return None

    @staticmethod
    def _is_etf_info(info: dict) -> bool:
        quote_type = str(info.get("quoteType") or "").upper()
        if quote_type == "ETF":
            return True
        legal_type = str(info.get("legalType") or "").lower()
        return "exchange traded fund" in legal_type

    @staticmethod
    def _etf_sector_label(info: dict) -> str:
        for key in ("category", "fundFamily", "industry", "sector"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "Exchange-traded fund"

    @staticmethod
    def _etf_country_label(info: dict) -> str:
        country = info.get("country")
        if isinstance(country, str) and country.strip():
            return country.strip()

        exchange = str(info.get("exchange") or info.get("fullExchangeName") or "").upper()
        us_exchanges = {
            "NYQ",
            "NMS",
            "NGM",
            "NCM",
            "ASE",
            "PCX",
            "BTS",
            "SNP",
            "NYQ",
            "NASDAQ",
            "NYSE",
            "ARCA",
        }
        if exchange in us_exchanges or "NEW YORK" in exchange or "NASDAQ" in exchange:
            return "United States"

        return "United States"

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
        if self.yfinance_adapter is None:
            raise ValueError("Yahoo Finance adapter not configured")
        return self.yfinance_adapter.get_52w_range(symbol)

    def format_52w_range(self, symbol: str) -> str:
        low, high = self.get_52w_range_yf(symbol)
        return f"${low:.2f} – ${high:.2f}"

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
