from __future__ import annotations

from datetime import date
from typing import Any

from app.adapters.finnhub.finnhub_adapter import FinnhubAdapter
from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.builders.yfinance_analysis_builder import YFinanceAnalysisBuilder
from app.broker.fiscal_period import format_fiscal_period
from app.models.company_research_models import NewsHeadline
from app.models.earnings_models import (
    BeatLabel,
    EarningsEvent,
    EarningsListResponse,
    EarningsTiming,
    TranscriptSegment,
)


class EarningsBuilder:
    _TIMING_MAP = {
        "bmo": "bmo",
        "amc": "amc",
        "dmh": "dmh",
        "before market open": "bmo",
        "after market close": "amc",
        "during market hours": "dmh",
    }

    def __init__(
        self,
        yfinance_adapter: YFinanceAdapter,
        finnhub_adapter: FinnhubAdapter | None = None,
        yfinance_analysis_builder: YFinanceAnalysisBuilder | None = None,
    ):
        self.yfinance_adapter = yfinance_adapter
        self.finnhub_adapter = finnhub_adapter
        self.yfinance_analysis_builder = yfinance_analysis_builder

    def build_list(self, symbol: str, limit: int = 8) -> EarningsListResponse:
        symbol = symbol.upper()
        bundle = self.yfinance_adapter.get_earnings_bundle(symbol=symbol, limit=limit)
        surprises = bundle.get("surprises") or []
        street_analysis = None
        if self.yfinance_analysis_builder is not None:
            street_analysis = self.yfinance_analysis_builder.build(symbol)

        if not surprises and not bundle.get("upcoming"):
            return EarningsListResponse(
                symbol=symbol,
                street_analysis=street_analysis,
            )

        revenue_by_period: dict[str, float] = bundle.get("revenue_by_period") or {}

        history: list[EarningsEvent] = []
        for item in surprises:
            report_date = self._period_to_report_date(item.get("period"))
            if not report_date:
                continue
            calendar_row = self._calendar_row_for_period(
                report_date=report_date,
                revenue_by_period=revenue_by_period,
            )
            event = self._build_event(
                symbol=symbol,
                report_date=report_date,
                surprise_row=item,
                calendar_row=calendar_row,
                transcript_id=None,
                is_upcoming=False,
            )
            history.append(event)

        upcoming = self._build_upcoming_event(
            symbol=symbol,
            upcoming_row=bundle.get("upcoming"),
        )

        return EarningsListResponse(
            symbol=symbol,
            upcoming=upcoming,
            history=history,
            street_analysis=street_analysis,
        )

    def build_event_for_date(
        self,
        symbol: str,
        report_date: date,
        transcript_id: str | None = None,
    ) -> EarningsEvent | None:
        symbol = symbol.upper()
        iso_date = report_date.isoformat()
        bundle = self.yfinance_adapter.get_earnings_bundle(symbol=symbol, limit=20)
        surprises = bundle.get("surprises") or []
        revenue_by_period: dict[str, float] = bundle.get("revenue_by_period") or {}

        surprise_row: dict[str, Any] | None = None
        for item in surprises:
            if item.get("period") == iso_date:
                surprise_row = item
                break
        if surprise_row is None:
            surprise_row = self._surprise_for_fiscal_period(
                surprises,
                report_date=report_date,
            )

        upcoming_row = bundle.get("upcoming")
        calendar_row = self._calendar_row_for_period(
            report_date=report_date,
            revenue_by_period=revenue_by_period,
        )
        if upcoming_row and upcoming_row.get("period") == iso_date:
            calendar_row = {
                **calendar_row,
                "epsEstimate": upcoming_row.get("estimate"),
                "revenueEstimate": upcoming_row.get("revenueEstimate"),
                "hour": upcoming_row.get("timing"),
                "quarter": upcoming_row.get("quarter"),
                "year": upcoming_row.get("year"),
            }

        if not surprise_row and not calendar_row:
            return None

        resolved_surprise = surprise_row or {}
        return self._build_event(
            symbol=symbol,
            report_date=report_date,
            surprise_row=resolved_surprise,
            calendar_row=calendar_row,
            transcript_id=transcript_id,
            is_upcoming=self._event_is_upcoming(
                report_date,
                surprise_row=resolved_surprise,
                calendar_row=calendar_row,
            ),
        )

    def lookup_transcript_id(self, symbol: str, report_date: date) -> str | None:
        if self.finnhub_adapter is None:
            return None
        by_date = self._load_transcript_ids_by_date(symbol=symbol)
        return by_date.get(report_date.isoformat())

    def fetch_transcript(self, transcript_id: str) -> list[TranscriptSegment]:
        if self.finnhub_adapter is None:
            return []
        try:
            raw = self.finnhub_adapter.get_transcript(transcript_id)
        except Exception:
            return []
        return self.parse_transcript(raw)

    def parse_transcript(self, raw: Any) -> list[TranscriptSegment]:
        if not isinstance(raw, dict):
            return []

        segments: list[TranscriptSegment] = []
        for entry in raw.get("transcript") or []:
            if not isinstance(entry, dict):
                continue
            speaker = str(entry.get("name") or "Unknown").strip()
            role = entry.get("title") or entry.get("role")
            speech_parts = entry.get("speech") or []
            if isinstance(speech_parts, list):
                text = "\n".join(str(part).strip() for part in speech_parts if part)
            else:
                text = str(speech_parts).strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    speaker=speaker,
                    role=str(role).strip() if role else None,
                    text=text,
                )
            )
        return segments

    def transcript_to_text(
        self, segments: list[TranscriptSegment], max_chars: int = 12_000
    ) -> str:
        chunks: list[str] = []
        total = 0
        for segment in segments:
            block = f"{segment.speaker}: {segment.text}"
            if total + len(block) > max_chars:
                remaining = max_chars - total
                if remaining > 200:
                    chunks.append(block[:remaining] + "…")
                break
            chunks.append(block)
            total += len(block) + 2
        return "\n\n".join(chunks)

    def news_to_headlines(self, raw_news: Any, limit: int = 8) -> list[NewsHeadline]:
        headlines: list[NewsHeadline] = []
        for item in self._safe_list(raw_news)[:limit]:
            if not isinstance(item, dict):
                continue
            dt = item.get("datetime")
            if isinstance(dt, (int, float)):
                from datetime import datetime

                dt_str = datetime.fromtimestamp(dt).isoformat()
            else:
                dt_str = str(dt or "")
            headlines.append(
                NewsHeadline(
                    headline=str(item.get("headline") or ""),
                    summary=item.get("summary") or None,
                    source=str(item.get("source") or ""),
                    datetime=dt_str,
                )
            )
        return headlines

    def _build_upcoming_event(
        self,
        *,
        symbol: str,
        upcoming_row: dict[str, Any] | None,
    ) -> EarningsEvent | None:
        if not upcoming_row:
            return None
        report_date = self._period_to_report_date(upcoming_row.get("period"))
        if not report_date:
            return None
        calendar_row = {
            "epsEstimate": upcoming_row.get("estimate"),
            "revenueEstimate": upcoming_row.get("revenueEstimate"),
            "hour": upcoming_row.get("timing"),
            "quarter": upcoming_row.get("quarter"),
            "year": upcoming_row.get("year"),
        }
        return self._build_event(
            symbol=symbol,
            report_date=report_date,
            surprise_row={},
            calendar_row=calendar_row,
            transcript_id=None,
            is_upcoming=True,
        )

    def _load_transcript_ids_by_date(self, symbol: str) -> dict[str, str]:
        if self.finnhub_adapter is None:
            return {}
        try:
            raw = self.finnhub_adapter.get_transcripts_list(symbol=symbol)
        except Exception:
            return {}
        entries = raw.get("transcripts") if isinstance(raw, dict) else []
        by_date: dict[str, str] = {}
        for entry in self._safe_list(entries):
            if not isinstance(entry, dict):
                continue
            transcript_id = entry.get("id")
            entry_time = entry.get("time") or entry.get("title")
            if not transcript_id or not entry_time:
                continue
            parsed = self._parse_transcript_time(str(entry_time))
            if parsed:
                by_date[parsed.isoformat()] = str(transcript_id)
        return by_date

    def _build_event(
        self,
        symbol: str,
        report_date: date,
        surprise_row: dict[str, Any],
        calendar_row: dict[str, Any],
        transcript_id: str | None,
        is_upcoming: bool,
    ) -> EarningsEvent:
        quarter = surprise_row.get("quarter") or calendar_row.get("quarter")
        year = surprise_row.get("year") or calendar_row.get("year")
        fiscal_period = surprise_row.get("fiscalPeriod") or format_fiscal_period(
            int(quarter) if quarter is not None else None,
            int(year) if year is not None else None,
        )

        eps_actual = None if is_upcoming else self._first_float(
            surprise_row.get("actual"),
            calendar_row.get("epsActual"),
        )
        eps_estimate = self._first_float(
            surprise_row.get("estimate"),
            calendar_row.get("epsEstimate"),
        )
        eps_surprise_pct = None if is_upcoming else self._first_float(
            surprise_row.get("surprisePercent")
        )
        if (
            not is_upcoming
            and eps_surprise_pct is None
            and eps_actual is not None
            and eps_estimate
        ):
            eps_surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100

        revenue_actual = (
            None
            if is_upcoming
            else self._first_float(calendar_row.get("revenueActual"))
        )
        revenue_estimate = self._first_float(calendar_row.get("revenueEstimate"))
        revenue_surprise_pct = None
        if (
            not is_upcoming
            and revenue_actual is not None
            and revenue_estimate
        ):
            revenue_surprise_pct = (
                (revenue_actual - revenue_estimate) / abs(revenue_estimate)
            ) * 100

        timing = self._normalize_timing(
            calendar_row.get("hour") or calendar_row.get("timing")
        )
        beat_label = self._beat_label(
            eps_actual=eps_actual,
            eps_estimate=eps_estimate,
            is_upcoming=is_upcoming,
        )

        return EarningsEvent(
            symbol=symbol,
            reportDate=report_date.isoformat(),
            fiscalPeriod=fiscal_period,
            quarter=int(quarter) if quarter is not None else None,
            year=int(year) if year is not None else None,
            timing=timing,
            epsActual=eps_actual,
            epsEstimate=eps_estimate,
            epsSurprisePct=eps_surprise_pct,
            revenueActual=revenue_actual,
            revenueEstimate=revenue_estimate,
            revenueSurprisePct=revenue_surprise_pct,
            beatLabel=beat_label,
            transcriptId=transcript_id,
            isUpcoming=is_upcoming,
        )

    @staticmethod
    def _calendar_row_for_period(
        *,
        report_date: date,
        revenue_by_period: dict[str, float],
    ) -> dict[str, Any]:
        row: dict[str, Any] = {}
        revenue = revenue_by_period.get(report_date.isoformat())
        if revenue is not None:
            row["revenueActual"] = revenue
        return row

    @staticmethod
    def _surprise_for_fiscal_period(
        surprises: list[dict[str, Any]],
        *,
        report_date: date,
    ) -> dict[str, Any] | None:
        """Match by fiscal quarter when the UI date differs slightly from yfinance period."""
        for item in surprises:
            if item.get("period") == report_date.isoformat():
                return item
        return None

    @staticmethod
    def _safe_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return []

    @staticmethod
    def _period_to_report_date(period: Any) -> date | None:
        if not period:
            return None
        try:
            return date.fromisoformat(str(period)[:10])
        except ValueError:
            return None

    @staticmethod
    def _parse_transcript_time(value: str) -> date | None:
        from datetime import datetime

        value = value.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(
                    value[:19] if fmt.endswith("%S") else value[:10],
                    fmt,
                ).date()
            except ValueError:
                continue
        if len(value) >= 10:
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
        return None

    @classmethod
    def _normalize_timing(cls, value: Any) -> EarningsTiming | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return cls._TIMING_MAP.get(normalized)  # type: ignore[return-value]

    @staticmethod
    def _first_float(*values: Any) -> float | None:
        for value in values:
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _event_is_upcoming(
        report_date: date,
        *,
        surprise_row: dict[str, Any],
        calendar_row: dict[str, Any],
    ) -> bool:
        if surprise_row.get("actual") is not None:
            return False
        if calendar_row.get("epsActual") is not None:
            return False
        return report_date > date.today()

    @staticmethod
    def _beat_label(
        eps_actual: float | None,
        eps_estimate: float | None,
        is_upcoming: bool,
    ) -> BeatLabel | None:
        if is_upcoming:
            return "pending"
        if eps_actual is None or eps_estimate is None:
            return None
        diff = eps_actual - eps_estimate
        tolerance = max(abs(eps_estimate) * 0.01, 0.01)
        if diff > tolerance:
            return "beat"
        if diff < -tolerance:
            return "miss"
        return "inline"
