from datetime import date, timedelta

from app.builders.earnings_builder import EarningsBuilder
from app.builders.finnhub_builder import FinnhubBuilder
from app.models.earnings_models import (
    EarningsDetailResponse,
    EarningsListResponse,
)


class EarningsService:
    def __init__(
        self,
        earnings_builder: EarningsBuilder,
        finnhub_builder: FinnhubBuilder,
    ):
        self.earnings_builder = earnings_builder
        self.finnhub_builder = finnhub_builder

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
        raw_news = self.finnhub_builder.get_company_news(
            symbol=symbol,
            _from=start,
            to=end,
        )
        return self.earnings_builder.news_to_headlines(raw_news.root, limit=10)
