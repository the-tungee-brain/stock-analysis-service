from datetime import date, timedelta
import os

from app.builders.finnhub_builder import FinnhubBuilder
from app.models.finnhub_news_models import NewsResponse


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

        news_response.root = news_response.root[:10]
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
