from app.builders.finnhub_builder import FinnhubBuilder
from app.models.finnhub_news_models import NewsResponse
from datetime import date, timedelta


class NewsService:
    def __init__(self, finnhub_builder: FinnhubBuilder):
        self.finnhub_builder = finnhub_builder

    def get_company_news(self, symbol: str) -> NewsResponse:
        today = date.today()
        return self.finnhub_builder.get_company_news(
            symbol=symbol,
            _from=today,
            to=today,
        )
