from datetime import date, datetime, timezone
from unittest.mock import MagicMock

from app.models.company_research_models import NewsHeadline
from app.models.earnings_models import EarningsEvent
from app.models.finnhub_news_models import NewsItem, NewsResponse
from app.builders.earnings_builder import EarningsBuilder
from app.services.earnings_service import EarningsService
from app.services.news_service import NewsService


def _news_item(*, headline: str, published: datetime) -> NewsItem:
    return NewsItem(
        category="press release",
        datetime=published,
        headline=headline,
        id=1,
        related="AAPL",
        source="Business Wire",
        summary="Summary",
        url="https://example.com/pr",
    )


def test_get_press_releases_around_date_filters_window():
    report = date(2026, 5, 15)
    in_window = datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)
    out_window = datetime(2026, 4, 1, 12, 0, tzinfo=timezone.utc)

    yfinance = MagicMock()
    finnhub = MagicMock()
    news_service = NewsService(finnhub_builder=finnhub, yfinance_adapter=yfinance)
    news_service.get_press_releases = MagicMock(
        return_value=NewsResponse(
            root=[
                _news_item(headline="In window", published=in_window),
                _news_item(headline="Too old", published=out_window),
            ]
        )
    )

    items = news_service.get_press_releases_around_date(
        "AAPL",
        report,
        days_before=7,
        days_after=3,
    )

    assert len(items) == 1
    assert items[0].headline == "In window"


def test_earnings_detail_includes_official_releases():
    event = EarningsEvent(
        symbol="AAPL",
        reportDate="2026-05-15",
        fiscalPeriod="Q2 FY2026",
    )
    builder = MagicMock()
    builder.build_event_for_date.return_value = event
    finnhub = MagicMock()
    finnhub.get_company_news.return_value = MagicMock(root=[])

    news_service = MagicMock()
    news_service.get_press_releases_around_date.return_value = [
        NewsHeadline(
            headline="Earnings PR",
            source="IR",
            datetime="2026-05-14T00:00:00+00:00",
        )
    ]

    service = EarningsService(
        earnings_builder=builder,
        finnhub_builder=finnhub,
        news_service=news_service,
    )

    detail = service.get_detail(
        symbol="AAPL",
        report_date=date(2026, 5, 15),
        include_transcript=False,
    )

    assert detail is not None
    assert len(detail.officialReleases) == 1
    assert detail.officialReleases[0].headline == "Earnings PR"


def test_earnings_detail_soft_returns_empty_transcript_when_provider_disabled():
    yfinance = MagicMock()
    yfinance.get_earnings_bundle.return_value = {
        "surprises": [
            {
                "period": "2026-05-15",
                "quarter": 2,
                "year": 2026,
                "fiscalPeriod": "Q2 FY2026",
                "actual": 1.2,
                "estimate": 1.1,
            }
        ],
        "upcoming": None,
        "revenue_by_period": {},
    }
    transcript_provider = MagicMock()
    transcript_provider.get_transcripts_list.return_value = {"transcripts": []}
    transcript_provider.get_transcript.return_value = {"transcript": []}
    builder = EarningsBuilder(
        yfinance_adapter=yfinance,
        finnhub_adapter=transcript_provider,
    )
    finnhub = MagicMock()
    finnhub.get_company_news.return_value = MagicMock(root=[])
    service = EarningsService(
        earnings_builder=builder,
        finnhub_builder=finnhub,
        news_service=None,
    )

    detail = service.get_detail(
        symbol="AAPL",
        report_date=date(2026, 5, 15),
        include_transcript=True,
    )

    assert detail is not None
    assert detail.transcriptAvailable is False
    assert detail.transcript == []
    assert detail.event.transcriptId is None
    transcript_provider.get_transcripts_list.assert_called_once_with(symbol="AAPL")
    transcript_provider.get_transcript.assert_not_called()
