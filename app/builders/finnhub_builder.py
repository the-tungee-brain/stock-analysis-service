from app.adapters.finnhub.finnhub_adapter import FinnhubAdapter
from app.models.finnhub_news_models import NewsResponse
from datetime import date


class FinnhubBuilder:
    def __init__(self, finnhub_adapter: FinnhubAdapter):
        self.finnhub_adapter = finnhub_adapter

    def get_company_news(self, symbol: str, _from: date, to: date) -> NewsResponse:
        _from = _from.strftime("%Y-%m-%d")
        to = to.strftime("%Y-%m-%d")
        raw_news_response = self.finnhub_adapter.get_company_news(
            symbol=symbol, _from=_from, to=to
        )
        return NewsResponse.model_validate(raw_news_response)
