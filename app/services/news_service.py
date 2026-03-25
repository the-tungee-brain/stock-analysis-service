from app.builders.finnhub_builder import FinnhubBuilder
from app.models.finnhub_news_models import NewsResponse
from datetime import date, timedelta


class NewsService:
    def __init__(self, finnhub_builder: FinnhubBuilder):
        self.finnhub_builder = finnhub_builder

    def get_company_news(self, symbol: str) -> NewsResponse:
        today = date.today()
        yesterday = today - timedelta(days=1)

        news_response = self.finnhub_builder.get_company_news(
            symbol=symbol,
            _from=yesterday,
            to=today,
        )

        news_response.root = news_response.root[:10]

        return news_response
