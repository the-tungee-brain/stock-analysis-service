from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from app.adapters.finnhub.finnhub_adapter import FinnhubAdapter
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

    def __init__(self, finnhub_adapter: FinnhubAdapter):
        self.finnhub_adapter = finnhub_adapter

    def build_list(self, symbol: str, limit: int = 8) -> EarningsListResponse:
        symbol = symbol.upper()
        raw_surprises = self._safe_list(
            self.finnhub_adapter.get_company_earnings(symbol=symbol, limit=limit)
        )
        if not raw_surprises:
            return EarningsListResponse(symbol=symbol)

        report_dates = [
            self._period_to_report_date(item.get("period"))
            for item in raw_surprises
            if item.get("period")
        ]
        report_dates = [d for d in report_dates if d]

        calendar_by_date = self._load_calendar_by_date(
            symbol=symbol,
            start=min(report_dates) if report_dates else date.today() - timedelta(days=365),
            end=max(report_dates) if report_dates else date.today(),
        )
        transcript_by_date = self._load_transcript_ids_by_date(symbol=symbol)

        history: list[EarningsEvent] = []
        for item in raw_surprises:
            report_date = self._period_to_report_date(item.get("period"))
            if not report_date:
                continue
            calendar_row = calendar_by_date.get(report_date.isoformat(), {})
            event = self._build_event(
                symbol=symbol,
                report_date=report_date,
                surprise_row=item,
                calendar_row=calendar_row,
                transcript_id=transcript_by_date.get(report_date.isoformat()),
                is_upcoming=False,
            )
            history.append(event)

        upcoming = self._load_upcoming(
            symbol=symbol,
            calendar_by_date=calendar_by_date,
            transcript_by_date=transcript_by_date,
        )

        return EarningsListResponse(
            symbol=symbol,
            upcoming=upcoming,
            history=history,
        )

    def build_event_for_date(
        self,
        symbol: str,
        report_date: date,
        transcript_id: str | None = None,
    ) -> EarningsEvent | None:
        symbol = symbol.upper()
        iso_date = report_date.isoformat()

        calendar_by_date = self._load_calendar_by_date(
            symbol=symbol,
            start=report_date - timedelta(days=7),
            end=report_date + timedelta(days=7),
        )
        calendar_row = calendar_by_date.get(iso_date, {})

        surprise_row: dict[str, Any] | None = None
        raw_surprises = self._safe_list(
            self.finnhub_adapter.get_company_earnings(symbol=symbol, limit=20)
        )
        for item in raw_surprises:
            item_date = self._period_to_report_date(item.get("period"))
            if item_date and item_date.isoformat() == iso_date:
                surprise_row = item
                break

        if not surprise_row and not calendar_row:
            return None

        if transcript_id is None:
            transcript_by_date = self._load_transcript_ids_by_date(symbol=symbol)
            transcript_id = transcript_by_date.get(iso_date)

        return self._build_event(
            symbol=symbol,
            report_date=report_date,
            surprise_row=surprise_row or {},
            calendar_row=calendar_row,
            transcript_id=transcript_id,
            is_upcoming=report_date > date.today(),
        )

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

    def _load_upcoming(
        self,
        symbol: str,
        calendar_by_date: dict[str, dict[str, Any]],
        transcript_by_date: dict[str, str],
    ) -> EarningsEvent | None:
        today = date.today()
        future_calendar = self._load_calendar_by_date(
            symbol=symbol,
            start=today,
            end=today + timedelta(days=120),
        )
        calendar_by_date.update(future_calendar)

        upcoming_dates = sorted(
            d for d in calendar_by_date if date.fromisoformat(d) >= today
        )
        if not upcoming_dates:
            return None

        report_date = date.fromisoformat(upcoming_dates[0])
        calendar_row = calendar_by_date[upcoming_dates[0]]
        return self._build_event(
            symbol=symbol,
            report_date=report_date,
            surprise_row={},
            calendar_row=calendar_row,
            transcript_id=transcript_by_date.get(upcoming_dates[0]),
            is_upcoming=True,
        )

    def _load_calendar_by_date(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> dict[str, dict[str, Any]]:
        raw = self.finnhub_adapter.get_earnings_calendar(
            _from=start.isoformat(),
            to=end.isoformat(),
            symbol=symbol,
            international=False,
        )
        rows = raw.get("earningsCalendar") if isinstance(raw, dict) else []
        by_date: dict[str, dict[str, Any]] = {}
        for row in self._safe_list(rows):
            if not isinstance(row, dict):
                continue
            row_symbol = str(row.get("symbol") or "").upper()
            if row_symbol and row_symbol != symbol.upper():
                continue
            row_date = row.get("date")
            if not row_date:
                continue
            by_date[str(row_date)] = row
        return by_date

    def _load_transcript_ids_by_date(self, symbol: str) -> dict[str, str]:
        raw = self.finnhub_adapter.get_transcripts_list(symbol=symbol)
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

        eps_actual = self._first_float(
            surprise_row.get("actual"),
            calendar_row.get("epsActual"),
        )
        eps_estimate = self._first_float(
            surprise_row.get("estimate"),
            calendar_row.get("epsEstimate"),
        )
        eps_surprise_pct = self._first_float(surprise_row.get("surprisePercent"))
        if eps_surprise_pct is None and eps_actual is not None and eps_estimate:
            eps_surprise_pct = ((eps_actual - eps_estimate) / abs(eps_estimate)) * 100

        revenue_actual = self._first_float(calendar_row.get("revenueActual"))
        revenue_estimate = self._first_float(calendar_row.get("revenueEstimate"))
        revenue_surprise_pct = None
        if revenue_actual is not None and revenue_estimate:
            revenue_surprise_pct = (
                (revenue_actual - revenue_estimate) / abs(revenue_estimate)
            ) * 100

        timing = self._normalize_timing(calendar_row.get("hour"))
        beat_label = self._beat_label(
            eps_actual=eps_actual,
            eps_estimate=eps_estimate,
            is_upcoming=is_upcoming,
        )

        return EarningsEvent(
            symbol=symbol,
            reportDate=report_date.isoformat(),
            fiscalPeriod=self._fiscal_period(quarter, year),
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
        value = value.strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(value[:19] if fmt.endswith("%S") else value[:10], fmt).date()
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
    def _fiscal_period(quarter: Any, year: Any) -> str:
        if quarter and year:
            return f"Q{int(quarter)} {int(year)}"
        if year:
            return str(int(year))
        return "Unknown period"

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
