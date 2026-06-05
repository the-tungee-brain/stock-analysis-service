from __future__ import annotations

import logging

from app.models.company_research_models import ResearchContext, SecFilingHeadline
from app.models.intelligence_models import EventTimelineEntry
from app.services.earnings_service import EarningsService
from app.services.intelligence.event_timeline_builder import EventTimelineBuilder
from app.services.sec_research_service import SecResearchService

logger = logging.getLogger(__name__)


class ResearchEventsService:
    """Lightweight public event timeline for Research overview."""

    def __init__(
        self,
        *,
        sec_research_service: SecResearchService,
        earnings_service: EarningsService,
    ) -> None:
        self.sec_research_service = sec_research_service
        self.earnings_service = earnings_service

    def get_events(self, symbol: str) -> list[EventTimelineEntry]:
        symbol_upper = symbol.strip().upper()
        try:
            context = ResearchContext(
                symbol=symbol_upper,
                sec_recent_filings=self._load_sec_filings(symbol_upper),
                earnings=self._load_earnings(symbol_upper),
            )
            return EventTimelineBuilder.build(research=context)
        except Exception:
            logger.warning(
                "Research events unavailable for %s",
                symbol_upper,
                exc_info=True,
            )
            return []

    def _load_sec_filings(self, symbol: str) -> list[SecFilingHeadline]:
        try:
            filings_response = self.sec_research_service.filings(symbol=symbol, limit=8)
        except Exception:
            logger.warning("SEC events unavailable for %s", symbol, exc_info=True)
            return []

        return [
            SecFilingHeadline(
                form=filing.form,
                filing_date=filing.filing_date,
                report_date=filing.report_date,
            )
            for filing in filings_response.filings[:3]
        ]

    def _load_earnings(self, symbol: str):
        try:
            return self.earnings_service.build_research_context(symbol=symbol)
        except Exception:
            logger.warning("Earnings events unavailable for %s", symbol, exc_info=True)
            return None
