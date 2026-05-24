from app.builders.finnhub_builder import FinnhubBuilder
from app.models.finnhub_news_models import NewsResponse
from datetime import date, timedelta


class NewsService:
    def __init__(self, finnhub_builder: FinnhubBuilder):
        self.finnhub_builder = finnhub_builder

    def get_company_news(self, symbol: str, lookback_days: int = 1) -> NewsResponse:
        today = date.today()
        start = today - timedelta(days=lookback_days)

        news_response = self.finnhub_builder.get_company_news(
            symbol=symbol,
            _from=start,
            to=today,
        )

        news_response.root = news_response.root[:10]

        return news_response

    def get_press_releases(self, symbol: str, lookback_days: int = 30) -> NewsResponse:
        today = date.today()
        start = today - timedelta(days=lookback_days)

        releases = self.finnhub_builder.get_press_releases(
            symbol=symbol,
            _from=start,
            to=today,
        )

        releases.root = releases.root[:5]
        return releases
