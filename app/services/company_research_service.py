import os
from concurrent.futures import ThreadPoolExecutor

from app.adapters.cache.research_context_cache import ResearchContextCache
from app.builders.fundamentals_builder import FundamentalsBuilder
from app.builders.yfinance_financials_builder import YFinanceFinancialsBuilder
from app.builders.ticker_symbol_builder import TickerSymbolBuilder
from app.models.company_research_models import (
    EarningsContext,
    FundamentalMetric,
    NewsHeadline,
    ResearchContext,
    SecFilingHeadline,
    SecRatioTrendPoint,
)
from app.services.company_profile_service import CompanyProfileService
from app.services.earnings_service import EarningsService
from app.services.etf_research_service import EtfResearchService
from app.services.enriched_news_service import EnrichedNewsService
from app.services.market_service import MarketService
from app.services.news_service import NewsService, finnhub_press_releases_enabled
from app.services.sec_research_service import SecResearchService


class CompanyResearchService:
    def __init__(
        self,
        company_profile_service: CompanyProfileService,
        market_service: MarketService,
        news_service: NewsService,
        fundamentals_builder: FundamentalsBuilder,
        sec_research_service: SecResearchService,
        earnings_service: EarningsService,
        research_context_cache: ResearchContextCache | None = None,
        enriched_news_service: EnrichedNewsService | None = None,
        ticker_symbol_builder: TickerSymbolBuilder | None = None,
        etf_research_service: EtfResearchService | None = None,
        yfinance_financials_builder: YFinanceFinancialsBuilder | None = None,
    ):
        self.company_profile_service = company_profile_service
        self.market_service = market_service
        self.news_service = news_service
        self.fundamentals_builder = fundamentals_builder
        self.sec_research_service = sec_research_service
        self.earnings_service = earnings_service
        self.research_context_cache = research_context_cache
        self.enriched_news_service = enriched_news_service
        self.ticker_symbol_builder = ticker_symbol_builder
        self.etf_research_service = etf_research_service
        self.yfinance_financials_builder = yfinance_financials_builder

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
    def context_fingerprint(context: ResearchContext) -> str:
        from app.adapters.cache.llm_output_cache import LLMOutputCache

        return LLMOutputCache.fingerprint_from_text(context.model_dump_json())

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
    ) -> tuple[
        list[FundamentalMetric],
        list[SecRatioTrendPoint],
        list[SecFilingHeadline],
        str | None,
    ]:
        lookup = self.sec_research_service.lookup(symbol=symbol)
        sec_company_info = self._build_sec_company_info(lookup)

        sec_fundamentals = self.sec_research_service.latest_fundamental_metrics(
            symbol=symbol
        )
        sec_ratio_trends = self.sec_research_service.annual_ratio_trends(
            symbol=symbol,
            limit=5,
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

        return sec_fundamentals, sec_ratio_trends, sec_recent_filings, sec_company_info

    @staticmethod
    def _run_loader(gap_key: str, loader):
        try:
            return loader(), None
        except Exception:
            return None, gap_key

    def _load_news(
        self, symbol: str, *, lookback_days: int = 7
    ) -> list[NewsHeadline]:
        # Finnhub /company-news is scoped to the requested symbol.
        news_response = self.news_service.get_company_news(
            symbol=symbol, lookback_days=lookback_days
        )
        return [
            NewsHeadline(
                headline=item.headline,
                summary=item.summary,
                source=item.source,
                datetime=item.datetime.isoformat(),
                url=str(item.url),
            )
            for item in news_response.root[:10]
        ]

    def _load_press_releases(self, symbol: str) -> list[NewsHeadline]:
        releases_response = self.news_service.get_press_releases(
            symbol=symbol, lookback_days=30
        )
        return [
            NewsHeadline(
                headline=item.headline,
                summary=item.summary,
                source=item.source or "Press release",
                datetime=item.datetime.isoformat(),
                url=str(item.url),
            )
            for item in releases_response.root[:5]
        ]

    def build_context(
        self, symbol: str, *, news_lookback_days: int = 7
    ) -> ResearchContext:
        symbol_upper = symbol.strip().upper()

        if self.research_context_cache is not None:
            try:
                cached = self.research_context_cache.get(
                    symbol=symbol_upper,
                    lookback_days=news_lookback_days,
                )
                if cached is not None:
                    return self._attach_enriched_news(cached)
            except Exception:
                pass

        context = self._build_context(
            symbol=symbol_upper,
            news_lookback_days=news_lookback_days,
        )
        context = self._attach_enriched_news(context)

        if self.research_context_cache is not None:
            try:
                self.research_context_cache.put(
                    symbol=symbol_upper,
                    context=context,
                    lookback_days=news_lookback_days,
                )
            except Exception:
                pass

        return context

    def build_lightweight_context(self, symbol: str) -> ResearchContext:
        """Snapshot + cached enrichment only — no Finnhub news/earnings fetches."""
        symbol_upper = symbol.strip().upper()

        snapshot = None
        try:
            snapshot = self.company_profile_service.get_snapshot(symbol_upper)
        except Exception:
            pass

        earnings = None
        if self.research_context_cache is not None:
            try:
                cached = self.research_context_cache.get(symbol=symbol_upper)
                if cached is not None:
                    earnings = cached.earnings
            except Exception:
                pass

        enriched_news = None
        if self.enriched_news_service is not None:
            enriched_news = self.enriched_news_service.get_cached_summary(
                symbol_upper
            )

        data_gaps: list[str] = []
        if snapshot is None:
            data_gaps.append("snapshot")

        return ResearchContext(
            symbol=symbol_upper,
            snapshot=snapshot,
            enriched_news=enriched_news,
            earnings=earnings,
            data_gaps=data_gaps,
        )

    def _attach_enriched_news(self, context: ResearchContext) -> ResearchContext:
        if self.enriched_news_service is None or context.enriched_news is not None:
            return context
        summary = self.enriched_news_service.get_cached_summary(symbol=context.symbol)
        if summary is None:
            return context
        return context.model_copy(update={"enriched_news": summary})

    def _resolve_asset_type(self, symbol: str) -> str | None:
        if self.ticker_symbol_builder is None:
            return None
        try:
            item = self.ticker_symbol_builder.get_by_symbol(symbol=symbol)
        except Exception:
            return None
        if item is None:
            if (
                self.etf_research_service is not None
                and self.etf_research_service.is_etf_symbol(symbol=symbol)
            ):
                return "ETF"
            return None
        if item.asset_type:
            return item.asset_type
        if (
            self.etf_research_service is not None
            and self.etf_research_service.is_etf_symbol(symbol=symbol)
        ):
            return "ETF"
        return item.asset_type

    @staticmethod
    def _is_etf(asset_type: str | None) -> bool:
        return asset_type == "ETF"

    def _build_etf_fundamentals(self, symbol: str) -> list[FundamentalMetric]:
        metrics = self.fundamentals_builder.build_etf_metrics(symbol=symbol)
        fundamentals: list[FundamentalMetric] = []
        if metrics.get("dividend_yield"):
            fundamentals.append(
                FundamentalMetric(
                    label="Dividend yield",
                    value=metrics["dividend_yield"],
                    note="Annual distributions as a share of the ETF price.",
                )
            )
        if metrics.get("expense_ratio"):
            fundamentals.append(
                FundamentalMetric(
                    label="Expense ratio",
                    value=metrics["expense_ratio"],
                    note="Annual fund operating costs as a share of assets.",
                )
            )
        return fundamentals

    def _build_context(
        self, symbol: str, *, news_lookback_days: int = 7
    ) -> ResearchContext:
        asset_type = self._resolve_asset_type(symbol)
        is_etf = self._is_etf(asset_type)

        data_gaps: list[str] = []

        with ThreadPoolExecutor(
            max_workers=int(os.getenv("RESEARCH_FETCH_WORKERS", "4"))
        ) as executor:
            future_snapshot = executor.submit(
                self._run_loader,
                "snapshot",
                lambda: self.company_profile_service.get_snapshot(symbol=symbol),
            )
            future_peers = executor.submit(
                self._run_loader,
                "peers",
                lambda: self.company_profile_service.get_peers(symbol=symbol),
            ) if not is_etf else None
            future_performance = executor.submit(
                self._run_loader,
                "performance",
                lambda: self.market_service.get_performance(symbol=symbol),
            )
            future_news = executor.submit(
                self._run_loader,
                "news",
                lambda: self._load_news(
                    symbol=symbol, lookback_days=news_lookback_days
                ),
            )
            if finnhub_press_releases_enabled():
                future_press = executor.submit(
                    self._run_loader,
                    "press_releases",
                    lambda: self._load_press_releases(symbol=symbol),
                )
            else:
                future_press = None
            future_fundamentals = executor.submit(
                self._run_loader,
                "fundamentals",
                lambda: (
                    self._build_etf_fundamentals(symbol=symbol)
                    if is_etf
                    else self.fundamentals_builder.build(symbol=symbol)
                ),
            )
            future_sec = (
                None
                if is_etf
                else executor.submit(
                    self._run_loader,
                    "sec",
                    lambda: self._load_sec_context(symbol=symbol),
                )
            )
            future_earnings = (
                None
                if is_etf
                else executor.submit(
                    self._run_loader,
                    "earnings",
                    lambda: self.earnings_service.build_research_context(symbol=symbol),
                )
            )
            future_yfinance_financials = (
                None
                if is_etf or self.yfinance_financials_builder is None
                else executor.submit(
                    self._run_loader,
                    "yfinance_financials",
                    lambda: self.yfinance_financials_builder.build(symbol=symbol),
                )
            )
            future_etf_holdings = (
                executor.submit(
                    self._run_loader,
                    "etf_holdings",
                    lambda: self.etf_research_service.build_holdings_context(
                        symbol=symbol
                    ),
                )
                if is_etf and self.etf_research_service is not None
                else None
            )

            snapshot, gap = future_snapshot.result()
            if gap:
                data_gaps.append(gap)

            peers, gap = future_peers.result() if future_peers is not None else (None, None)
            if gap:
                data_gaps.append(gap)
            peers = peers or []

            performance, gap = future_performance.result()
            if gap:
                data_gaps.append(gap)

            news, gap = future_news.result()
            if gap:
                data_gaps.append(gap)
            news = news or []

            press_releases: list[NewsHeadline] = []
            if future_press is not None:
                press_releases, gap = future_press.result()
                if gap:
                    data_gaps.append(gap)
                press_releases = press_releases or []

            fundamentals, gap = future_fundamentals.result()
            if gap:
                data_gaps.append(gap)
            fundamentals = fundamentals or []

            sec_result, gap = (
                future_sec.result() if future_sec is not None else (None, None)
            )
            sec_fundamentals: list[FundamentalMetric] = []
            sec_ratio_trends: list[SecRatioTrendPoint] = []
            sec_recent_filings: list[SecFilingHeadline] = []
            sec_company_info: str | None = None
            if gap:
                data_gaps.append(gap)
            elif sec_result is not None:
                (
                    sec_fundamentals,
                    sec_ratio_trends,
                    sec_recent_filings,
                    sec_company_info,
                ) = sec_result

            earnings, gap = (
                future_earnings.result() if future_earnings is not None else (None, None)
            )
            if gap:
                data_gaps.append(gap)

            yfinance_financials = None
            if future_yfinance_financials is not None:
                yfinance_financials, gap = future_yfinance_financials.result()
                if gap:
                    data_gaps.append(gap)

            etf_holdings = None
            if future_etf_holdings is not None:
                etf_holdings, gap = future_etf_holdings.result()
                if gap:
                    data_gaps.append(gap)

        return ResearchContext(
            symbol=symbol.upper(),
            asset_type=asset_type,
            snapshot=snapshot,
            performance=performance,
            news=news,
            press_releases=press_releases,
            fundamentals=fundamentals,
            sec_fundamentals=sec_fundamentals,
            sec_ratio_trends=sec_ratio_trends,
            sec_recent_filings=sec_recent_filings,
            sec_company_info=sec_company_info,
            peers=peers,
            earnings=earnings,
            yfinance_financials=yfinance_financials,
            etf_holdings=etf_holdings,
            data_gaps=data_gaps,
        )
