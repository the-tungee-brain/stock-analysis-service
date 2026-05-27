from datetime import date, datetime, timedelta, timezone
import os

from app.builders.finnhub_builder import FinnhubBuilder
from app.models.finnhub_news_models import NewsResponse

MARKET_NEWS_DISPLAY_LIMIT = int(os.getenv("MARKET_NEWS_LIMIT", "5"))
MARKET_NEWS_PROMPT_LIMIT = int(os.getenv("MARKET_NEWS_PROMPT_LIMIT", "3"))
MARKET_NEWS_LOOKBACK_HOURS = int(os.getenv("MARKET_NEWS_LOOKBACK_HOURS", "24"))
COMPANY_NEWS_DISPLAY_LIMIT = int(os.getenv("COMPANY_NEWS_LIMIT", "20"))


def finnhub_press_releases_enabled() -> bool:
    return os.getenv("FINNHUB_PRESS_RELEASES", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


class NewsService:
    def __init__(self, finnhub_builder: FinnhubBuilder):
        self.finnhub_builder = finnhub_builder

    def get_company_news(self, symbol: str, lookback_days: int = 1) -> NewsResponse:
        today = date.today()
        start = today - timedelta(days=lookback_days)

        try:
            news_response = self.finnhub_builder.get_company_news(
                symbol=symbol,
                _from=start,
                to=today,
            )
        except Exception:
            return NewsResponse(root=[])

        news_response.root = news_response.root[:COMPANY_NEWS_DISPLAY_LIMIT]
        return news_response

    def get_market_news(
        self,
        *,
        category: str = "general",
        limit: int | None = None,
        lookback_hours: int | None = None,
    ) -> NewsResponse:
        display_limit = limit or MARKET_NEWS_DISPLAY_LIMIT
        hours = lookback_hours or MARKET_NEWS_LOOKBACK_HOURS
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        try:
            news_response = self.finnhub_builder.get_market_news(category=category)
        except Exception:
            return NewsResponse(root=[])

        recent: list = []
        for item in news_response.root:
            published = item.datetime
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if published >= cutoff:
                recent.append(item)

        news_response.root = recent[:display_limit]
        return news_response

    def invalidate_company_news_cache(
        self, symbol: str, lookback_days: int = 7
    ) -> None:
        today = date.today()
        start = today - timedelta(days=lookback_days)
        self.finnhub_builder.invalidate_company_news_cache(
            symbol=symbol,
            _from=start,
            to=today,
        )

    def get_press_releases(self, symbol: str, lookback_days: int = 30) -> NewsResponse:
        if not finnhub_press_releases_enabled():
            return NewsResponse(root=[])

        today = date.today()
        start = today - timedelta(days=lookback_days)

        try:
            releases = self.finnhub_builder.get_press_releases(
                symbol=symbol,
                _from=start,
                to=today,
            )
        except Exception:
            return NewsResponse(root=[])

        releases.root = releases.root[:5]
        return releases
