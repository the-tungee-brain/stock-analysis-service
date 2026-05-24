from app.models.company_research_models import (
    NewsHeadline,
    ResearchContext,
)
from app.builders.fundamentals_builder import FundamentalsBuilder
from app.services.company_profile_service import CompanyProfileService
from app.services.market_service import MarketService
from app.services.news_service import NewsService


class CompanyResearchService:
    def __init__(
        self,
        company_profile_service: CompanyProfileService,
        market_service: MarketService,
        news_service: NewsService,
        fundamentals_builder: FundamentalsBuilder,
    ):
        self.company_profile_service = company_profile_service
        self.market_service = market_service
        self.news_service = news_service
        self.fundamentals_builder = fundamentals_builder

    def build_context(self, symbol: str) -> ResearchContext:
        snapshot = None
        performance = None
        news: list[NewsHeadline] = []
        fundamentals = []

        try:
            snapshot = self.company_profile_service.get_snapshot(symbol=symbol)
        except Exception:
            pass

        try:
            performance = self.market_service.get_performance(symbol=symbol)
        except Exception:
            pass

        try:
            news_response = self.news_service.get_company_news(
                symbol=symbol, lookback_days=7
            )
            news = [
                NewsHeadline(
                    headline=item.headline,
                    summary=item.summary,
                    source=item.source,
                    datetime=item.datetime.isoformat(),
                )
                for item in news_response.root[:10]
            ]
        except Exception:
            pass

        try:
            fundamentals = self.fundamentals_builder.build(symbol=symbol)
        except Exception:
            pass

        return ResearchContext(
            symbol=symbol.upper(),
            snapshot=snapshot,
            performance=performance,
            news=news,
            fundamentals=fundamentals,
        )
