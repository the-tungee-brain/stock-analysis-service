from __future__ import annotations

import logging
from datetime import date
from operator import attrgetter

import requests
from finnhub.exceptions import FinnhubAPIException

from app.adapters.finnhub.finnhub_adapter import FinnhubAdapter
from app.adapters.finnhub.finnhub_circuit import FinnhubUnavailableError
from app.models.finnhub_company_profile_models import CompanyProfile
from app.models.finnhub_news_models import NewsResponse
from app.models.finnhub_quote_models import Quote

logger = logging.getLogger(__name__)


class FinnhubBuilder:
    def __init__(self, finnhub_adapter: FinnhubAdapter):
        self.finnhub_adapter = finnhub_adapter

    def get_company_news(self, symbol: str, _from: date, to: date) -> NewsResponse:
        _from_str = _from.strftime("%Y-%m-%d")
        to_str = to.strftime("%Y-%m-%d")
        try:
            raw_news_response = self.finnhub_adapter.get_company_news(
                symbol=symbol, _from=_from_str, to=to_str
            )
            news_response = NewsResponse.model_validate(raw_news_response)
        except (
            FinnhubUnavailableError,
            FinnhubAPIException,
            requests.exceptions.RequestException,
        ) as exc:
            logger.warning(
                "Finnhub company news unavailable for %s: %s",
                symbol,
                exc,
            )
            return NewsResponse(root=[])
        except Exception:
            logger.warning(
                "Finnhub company news unavailable for %s", symbol, exc_info=True
            )
            return NewsResponse(root=[])

        news_response.root.sort(key=attrgetter("datetime"), reverse=True)
        return news_response

    def invalidate_company_news_cache(
        self, symbol: str, *, _from: date, to: date
    ) -> None:
        self.finnhub_adapter.invalidate_company_news(
            symbol=symbol,
            _from=_from.strftime("%Y-%m-%d"),
            to=to.strftime("%Y-%m-%d"),
        )

    def get_market_news(
        self,
        *,
        category: str = "general",
        min_id: int = 0,
    ) -> NewsResponse:
        try:
            raw = self.finnhub_adapter.get_general_news(
                category=category,
                min_id=min_id,
            )
            news_response = NewsResponse.model_validate(raw or [])
        except (
            FinnhubUnavailableError,
            FinnhubAPIException,
            requests.exceptions.RequestException,
        ) as exc:
            logger.warning("Finnhub market news unavailable: %s", exc)
            return NewsResponse(root=[])
        except Exception:
            logger.warning("Finnhub market news unavailable", exc_info=True)
            return NewsResponse(root=[])

        news_response.root.sort(key=attrgetter("datetime"), reverse=True)
        return news_response

    def get_company_profile(self, symbol: str) -> CompanyProfile | None:
        try:
            raw_company_profile = self.finnhub_adapter.get_company_profile(
                symbol=symbol
            )
            if not raw_company_profile:
                return None
            return CompanyProfile.model_validate(raw_company_profile)
        except Exception:
            logger.warning(
                "Finnhub company profile unavailable for %s", symbol, exc_info=True
            )
            return None

    def get_quote(self, symbol: str) -> Quote | None:
        try:
            raw_quote = self.finnhub_adapter.get_quote(symbol=symbol)
            if not raw_quote:
                return None
            return Quote.model_validate(raw_quote)
        except (
            FinnhubUnavailableError,
            FinnhubAPIException,
            requests.exceptions.RequestException,
        ) as exc:
            logger.warning("Finnhub quote unavailable for %s: %s", symbol, exc)
            return None
        except Exception:
            logger.warning("Finnhub quote unavailable for %s", symbol, exc_info=True)
            return None

    def get_peers(self, symbol: str) -> list[str]:
        try:
            raw_peers = self.finnhub_adapter.get_stock_peers(symbol=symbol)
        except Exception:
            logger.warning("Finnhub peers unavailable for %s", symbol, exc_info=True)
            return []
        if not isinstance(raw_peers, list):
            return []
        return [
            peer.upper()
            for peer in raw_peers
            if isinstance(peer, str) and peer.strip()
        ]

    def get_press_releases(self, symbol: str, _from: date, to: date) -> NewsResponse:
        _from_str = _from.strftime("%Y-%m-%d")
        to_str = to.strftime("%Y-%m-%d")
        try:
            raw = self.finnhub_adapter.get_press_releases(
                symbol=symbol, _from=_from_str, to=to_str
            )
        except Exception as exc:
            logger.warning(
                "Finnhub press releases unavailable for %s: %s",
                symbol,
                exc,
            )
            return NewsResponse(root=[])
        if not raw:
            return NewsResponse(root=[])
        news_response = NewsResponse.model_validate(raw)
        news_response.root.sort(key=attrgetter("datetime"), reverse=True)
        return news_response
