from app.adapters.finnhub.finnhub_adapter import FinnhubAdapter
from app.models.finnhub_news_models import NewsResponse
from app.models.finnhub_company_profile_models import CompanyProfile
from app.models.finnhub_quote_models import Quote
from datetime import date
from operator import attrgetter


class FinnhubBuilder:
    def __init__(self, finnhub_adapter: FinnhubAdapter):
        self.finnhub_adapter = finnhub_adapter

    def get_company_news(self, symbol: str, _from: date, to: date) -> NewsResponse:
        _from = _from.strftime("%Y-%m-%d")
        to = to.strftime("%Y-%m-%d")
        raw_news_response = self.finnhub_adapter.get_company_news(
            symbol=symbol, _from=_from, to=to
        )
        news_response = NewsResponse.model_validate(raw_news_response)

        news_response.root.sort(key=attrgetter("datetime"), reverse=True)
        return news_response

    def get_company_profile(self, symbol: str) -> CompanyProfile:
        raw_company_profile = self.finnhub_adapter.get_company_profile(symbol=symbol)
        return CompanyProfile.model_validate(raw_company_profile)

    def get_quote(self, symbol: str):
        raw_quote = self.finnhub_adapter.get_quote(symbol=symbol)
        return Quote.model_validate(raw_quote)
