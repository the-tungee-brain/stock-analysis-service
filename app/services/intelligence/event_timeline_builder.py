from __future__ import annotations

from datetime import datetime, timezone

from app.broker.order_utils import (
    order_average_fill_price,
    order_fill_time,
    order_primary_leg,
)
from app.models.company_research_models import NewsHeadline, ResearchContext
from app.models.intelligence_models import EventTimelineEntry
from app.models.schwab_order_models import SchwabOrder


class EventTimelineBuilder:
    MAX_ENTRIES = 12

    @staticmethod
    def build(
        *,
        research: ResearchContext,
        orders: list[SchwabOrder] | None = None,
        since: datetime | None = None,
    ) -> list[EventTimelineEntry]:
        entries: list[tuple[datetime, EventTimelineEntry]] = []

        if orders:
            for order in orders:
                fill_time = order_fill_time(order)
                if fill_time is None:
                    continue
                if since is not None and fill_time < since:
                    continue
                leg = order_primary_leg(order)
                instruction = leg.instruction if leg else "TRADE"
                qty = leg.quantity if leg else order.filledQuantity
                price = order_average_fill_price(order)
                detail_parts = []
                if qty is not None:
                    detail_parts.append(f"Qty {qty:g}")
                if price is not None:
                    detail_parts.append(f"@ ${price:.2f}")
                entries.append(
                    (
                        fill_time,
                        EventTimelineEntry(
                            date=fill_time.strftime("%Y-%m-%d"),
                            kind="trade",
                            title=f"{instruction} {research.symbol}",
                            detail=" ".join(detail_parts) or None,
                        ),
                    )
                )

        for filing in research.sec_recent_filings[:3]:
            try:
                filed = datetime.strptime(filing.filing_date[:10], "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                continue
            entries.append(
                (
                    filed,
                    EventTimelineEntry(
                        date=filing.filing_date[:10],
                        kind="filing",
                        title=f"{filing.form} filed",
                        detail=f"Period end {filing.report_date}",
                    ),
                )
            )

        earnings = research.earnings
        if earnings and earnings.last_report_date:
            try:
                reported = datetime.strptime(
                    earnings.last_report_date[:10], "%Y-%m-%d"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                reported = None
            if reported is not None:
                beat = earnings.last_beat_label or "unknown"
                eps = earnings.last_eps_surprise_pct or "N/A"
                entries.append(
                    (
                        reported,
                        EventTimelineEntry(
                            date=earnings.last_report_date[:10],
                            kind="earnings",
                            title=f"Earnings report ({beat})",
                            detail=f"EPS surprise {eps}",
                        ),
                    )
                )

        if earnings and earnings.upcoming_report_date:
            try:
                upcoming = datetime.strptime(
                    earnings.upcoming_report_date[:10], "%Y-%m-%d"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                upcoming = None
            if upcoming is not None:
                period = earnings.upcoming_fiscal_period or "upcoming quarter"
                entries.append(
                    (
                        upcoming,
                        EventTimelineEntry(
                            date=earnings.upcoming_report_date[:10],
                            kind="earnings",
                            title=f"Upcoming earnings ({period})",
                            detail=earnings.upcoming_timing,
                        ),
                    )
                )

        news_items = list(research.news)
        if research.enriched_news and research.enriched_news.dominant_driver:
            pass

        for item in news_items[:5]:
            published = EventTimelineBuilder._parse_datetime(item.datetime)
            if published is None:
                continue
            if since is not None and published < since:
                continue
            entries.append(
                (
                    published,
                    EventTimelineEntry(
                        date=published.strftime("%Y-%m-%d"),
                        kind="news",
                        title=item.headline[:120],
                        detail=item.source or None,
                        url=item.url,
                    ),
                )
            )

        for item in research.press_releases[:3]:
            published = EventTimelineBuilder._parse_datetime(item.datetime)
            if published is None:
                continue
            entries.append(
                (
                    published,
                    EventTimelineEntry(
                        date=published.strftime("%Y-%m-%d"),
                        kind="press_release",
                        title=item.headline[:120],
                        detail=item.source or None,
                        url=item.url,
                    ),
                )
            )

        entries.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in entries[: EventTimelineBuilder.MAX_ENTRIES]]

    @staticmethod
    def _parse_datetime(value: str) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
