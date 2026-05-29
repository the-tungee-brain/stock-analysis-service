from datetime import date, timedelta

from typing import TYPE_CHECKING

from app.builders.earnings_builder import EarningsBuilder
from app.builders.finnhub_builder import FinnhubBuilder
from app.models.earnings_models import (
    EarningsDetailResponse,
    EarningsListResponse,
)

if TYPE_CHECKING:
    from app.services.news_service import NewsService


class EarningsService:
    def __init__(
        self,
        earnings_builder: EarningsBuilder,
        finnhub_builder: FinnhubBuilder,
        news_service: "NewsService | None" = None,
    ):
        self.earnings_builder = earnings_builder
        self.finnhub_builder = finnhub_builder
        self.news_service = news_service

    def list_earnings(self, symbol: str, limit: int = 8) -> EarningsListResponse:
        return self.earnings_builder.build_list(symbol=symbol, limit=limit)

    def get_detail(
        self,
        symbol: str,
        report_date: date,
        transcript_id: str | None = None,
        include_transcript: bool = True,
    ) -> EarningsDetailResponse | None:
        event = self.earnings_builder.build_event_for_date(
            symbol=symbol,
            report_date=report_date,
            transcript_id=transcript_id,
        )
        if event is None:
            return None

        related_news = self._news_around_earnings(
            symbol=symbol,
            report_date=report_date,
        )
        official_releases = self._press_releases_around_earnings(
            symbol=symbol,
            report_date=report_date,
        )

        transcript_segments = []
        resolved_transcript_id = transcript_id or event.transcriptId
        if include_transcript:
            if not resolved_transcript_id:
                resolved_transcript_id = self.earnings_builder.lookup_transcript_id(
                    symbol=symbol,
                    report_date=report_date,
                )
            if resolved_transcript_id:
                transcript_segments = self.earnings_builder.fetch_transcript(
                    resolved_transcript_id
                )
                event = event.model_copy(
                    update={"transcriptId": resolved_transcript_id},
                )

        return EarningsDetailResponse(
            symbol=symbol.upper(),
            event=event,
            relatedNews=related_news,
            officialReleases=official_releases,
            transcriptAvailable=bool(transcript_segments),
            transcript=transcript_segments,
            analysis=None,
        )

    def transcript_excerpt(
        self,
        detail: EarningsDetailResponse,
        max_chars: int = 12_000,
    ) -> str | None:
        if not detail.transcript:
            return None
        return self.earnings_builder.transcript_to_text(
            detail.transcript,
            max_chars=max_chars,
        )

    def _news_around_earnings(self, symbol: str, report_date: date):
        start = report_date - timedelta(days=3)
        end = report_date + timedelta(days=3)
        if end > date.today():
            end = date.today()
        try:
            raw_news = self.finnhub_builder.get_company_news(
                symbol=symbol,
                _from=start,
                to=end,
            )
        except Exception:
            return []
        return self.earnings_builder.news_to_headlines(raw_news.root, limit=10)

    def _press_releases_around_earnings(self, symbol: str, report_date: date):
        if self.news_service is None:
            return []
        return self.news_service.get_press_releases_around_date(
            symbol=symbol,
            report_date=report_date,
            days_before=7,
            days_after=3,
            limit=10,
        )

    def build_research_context(self, symbol: str):
        from app.models.company_research_models import EarningsContext

        listing = self.list_earnings(symbol=symbol, limit=4)
        upcoming = listing.upcoming
        last_report = listing.history[0] if listing.history else None

        if upcoming is None and last_report is None:
            return None

        return EarningsContext(
            upcoming_report_date=upcoming.reportDate if upcoming else None,
            upcoming_fiscal_period=upcoming.fiscalPeriod if upcoming else None,
            upcoming_timing=upcoming.timing if upcoming else None,
            last_report_date=last_report.reportDate if last_report else None,
            last_fiscal_period=last_report.fiscalPeriod if last_report else None,
            last_beat_label=last_report.beatLabel if last_report else None,
            last_eps_surprise_pct=(
                self._fmt_surprise(last_report.epsSurprisePct)
                if last_report and last_report.epsSurprisePct is not None
                else None
            ),
            last_revenue_surprise_pct=(
                self._fmt_surprise(last_report.revenueSurprisePct)
                if last_report and last_report.revenueSurprisePct is not None
                else None
            ),
        )

    @staticmethod
    def _fmt_surprise(value: float) -> str:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.1f}%"
