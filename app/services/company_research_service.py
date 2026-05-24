from app.models.company_research_models import (
    FundamentalMetric,
    NewsHeadline,
    ResearchContext,
    SecFilingHeadline,
)
from app.builders.fundamentals_builder import FundamentalsBuilder
from app.services.company_profile_service import CompanyProfileService
from app.services.market_service import MarketService
from app.services.news_service import NewsService
from app.services.sec_research_service import SecResearchService


class CompanyResearchService:
    def __init__(
        self,
        company_profile_service: CompanyProfileService,
        market_service: MarketService,
        news_service: NewsService,
        fundamentals_builder: FundamentalsBuilder,
        sec_research_service: SecResearchService,
    ):
        self.company_profile_service = company_profile_service
        self.market_service = market_service
        self.news_service = news_service
        self.fundamentals_builder = fundamentals_builder
        self.sec_research_service = sec_research_service

    @staticmethod
    def merge_fundamentals(
        sec_metrics: list[FundamentalMetric],
        market_metrics: list[FundamentalMetric],
    ) -> list[FundamentalMetric]:
        merged: list[FundamentalMetric] = []
        seen_labels: set[str] = set()

        for metric in sec_metrics:
            merged.append(metric)
            seen_labels.add(metric.label.lower())

        for metric in market_metrics:
            if metric.label.lower() in seen_labels:
                continue
            merged.append(metric)

        return merged

    @staticmethod
    def _build_sec_company_info(lookup) -> str:
        lines: list[str] = [f"- SEC registered name: {lookup.name}"]

        if lookup.sic_description:
            lines.append(f"- Industry (SIC): {lookup.sic_description}")
        if lookup.fiscal_year_end:
            lines.append(f"- Fiscal year end: {lookup.fiscal_year_end}")
        if lookup.exchanges:
            lines.append(f"- Exchange: {', '.join(lookup.exchanges)}")
        if lookup.category:
            lines.append(f"- SEC filer category: {lookup.category}")

        return "\n".join(lines)

    def _load_sec_context(
        self, symbol: str
    ) -> tuple[list[FundamentalMetric], list[SecFilingHeadline], str | None]:
        lookup = self.sec_research_service.lookup(symbol=symbol)
        sec_company_info = self._build_sec_company_info(lookup)

        sec_fundamentals = self.sec_research_service.latest_fundamental_metrics(
            symbol=symbol
        )

        filings_response = self.sec_research_service.filings(symbol=symbol, limit=8)
        sec_recent_filings = [
            SecFilingHeadline(
                form=filing.form,
                filing_date=filing.filing_date,
                report_date=filing.report_date,
            )
            for filing in filings_response.filings[:5]
        ]

        return sec_fundamentals, sec_recent_filings, sec_company_info

    def build_context(self, symbol: str) -> ResearchContext:
        snapshot = None
        performance = None
        news: list[NewsHeadline] = []
        fundamentals: list[FundamentalMetric] = []
        sec_fundamentals: list[FundamentalMetric] = []
        sec_recent_filings: list[SecFilingHeadline] = []
        sec_company_info: str | None = None

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

        try:
            sec_fundamentals, sec_recent_filings, sec_company_info = (
                self._load_sec_context(symbol=symbol)
            )
        except Exception:
            pass

        return ResearchContext(
            symbol=symbol.upper(),
            snapshot=snapshot,
            performance=performance,
            news=news,
            fundamentals=fundamentals,
            sec_fundamentals=sec_fundamentals,
            sec_recent_filings=sec_recent_filings,
            sec_company_info=sec_company_info,
        )
