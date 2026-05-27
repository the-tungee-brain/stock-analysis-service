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
        try:
            raw_surprises = self._safe_list(
                self.finnhub_adapter.get_company_earnings(symbol=symbol, limit=limit)
            )
        except Exception:
            raw_surprises = []
        if not raw_surprises:
            return EarningsListResponse(symbol=symbol)

        report_dates = [
            self._period_to_report_date(item.get("period"))
            for item in raw_surprises
            if item.get("period")
        ]
        report_dates = [d for d in report_dates if d]

        today = date.today()
        history_start = (
            min(report_dates) if report_dates else today - timedelta(days=365)
        )
        history_end = max(report_dates) if report_dates else today
        calendar_start = min(history_start, today)
        calendar_end = max(history_end, today + timedelta(days=120))

        calendar_by_date = self._load_calendar_by_date(
            symbol=symbol,
            start=calendar_start,
            end=calendar_end,
        )

        reported_periods = self._reported_fiscal_periods(raw_surprises)

        history: list[EarningsEvent] = []
        for item in raw_surprises:
            report_date = self._period_to_report_date(item.get("period"))
            if not report_date:
                continue
            calendar_row = self._calendar_row_for_surprise(
                calendar_by_date,
                surprise_row=item,
                report_date=report_date,
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

        upcoming = self._pick_upcoming(
            symbol=symbol,
            calendar_by_date=calendar_by_date,
            reported_periods=reported_periods,
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
        try:
            raw_surprises = self._safe_list(
                self.finnhub_adapter.get_company_earnings(symbol=symbol, limit=20)
            )
        except Exception:
            raw_surprises = []
        for item in raw_surprises:
            item_date = self._period_to_report_date(item.get("period"))
            if item_date and item_date.isoformat() == iso_date:
                surprise_row = item
                break
        if surprise_row is None and calendar_row:
            surprise_row = self._surprise_for_calendar_row(
                raw_surprises,
                calendar_row=calendar_row,
            )

        if not surprise_row and not calendar_row:
            return None

        return self._build_event(
            symbol=symbol,
            report_date=report_date,
            surprise_row=surprise_row or {},
            calendar_row=calendar_row,
            transcript_id=transcript_id,
            is_upcoming=report_date > date.today(),
        )

    def lookup_transcript_id(self, symbol: str, report_date: date) -> str | None:
        by_date = self._load_transcript_ids_by_date(symbol=symbol)
        return by_date.get(report_date.isoformat())

    def fetch_transcript(self, transcript_id: str) -> list[TranscriptSegment]:
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

    def _pick_upcoming(
        self,
        symbol: str,
        calendar_by_date: dict[str, dict[str, Any]],
        *,
        reported_periods: set[tuple[int, int]] | None = None,
    ) -> EarningsEvent | None:
        today = date.today()
        reported = reported_periods or set()
        upcoming_dates = sorted(
            d
            for d in calendar_by_date
            if date.fromisoformat(d) >= today
            and not self._is_calendar_period_already_reported(
                calendar_by_date[d],
                reported_periods=reported,
            )
        )
        if not upcoming_dates:
            future_calendar = self._load_calendar_by_date(
                symbol=symbol,
                start=today,
                end=today + timedelta(days=120),
            )
            upcoming_dates = sorted(
                d
                for d in future_calendar
                if date.fromisoformat(d) >= today
                and not self._is_calendar_period_already_reported(
                    future_calendar[d],
                    reported_periods=reported,
                )
            )
            calendar_by_date = {**calendar_by_date, **future_calendar}
        if not upcoming_dates:
            return None

        report_date = date.fromisoformat(upcoming_dates[0])
        calendar_row = calendar_by_date[upcoming_dates[0]]
        return self._build_event(
            symbol=symbol,
            report_date=report_date,
            surprise_row={},
            calendar_row=calendar_row,
            transcript_id=None,
            is_upcoming=True,
        )

    def _load_upcoming(
        self,
        symbol: str,
        calendar_by_date: dict[str, dict[str, Any]],
    ) -> EarningsEvent | None:
        future_calendar = self._load_calendar_by_date(
            symbol=symbol,
            start=date.today(),
            end=date.today() + timedelta(days=120),
        )
        calendar_by_date.update(future_calendar)
        return self._pick_upcoming(symbol=symbol, calendar_by_date=calendar_by_date)

    def _load_calendar_by_date(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> dict[str, dict[str, Any]]:
        try:
            raw = self.finnhub_adapter.get_earnings_calendar(
                _from=start.isoformat(),
                to=end.isoformat(),
                symbol=symbol,
                international=False,
            )
        except Exception:
            return {}
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
    def _reported_fiscal_periods(raw_surprises: list[Any]) -> set[tuple[int, int]]:
        reported: set[tuple[int, int]] = set()
        for item in raw_surprises:
            if not isinstance(item, dict):
                continue
            if item.get("actual") is None:
                continue
            quarter = item.get("quarter")
            year = item.get("year")
            if quarter is None or year is None:
                continue
            reported.add((int(quarter), int(year)))
        return reported

    @staticmethod
    def _is_calendar_period_already_reported(
        calendar_row: dict[str, Any],
        *,
        reported_periods: set[tuple[int, int]],
    ) -> bool:
        quarter = calendar_row.get("quarter")
        year = calendar_row.get("year")
        if quarter is not None and year is not None:
            if (int(quarter), int(year)) in reported_periods:
                return True
        eps_actual = calendar_row.get("epsActual")
        if eps_actual is not None and calendar_row.get("date"):
            try:
                row_date = date.fromisoformat(str(calendar_row["date"])[:10])
            except ValueError:
                row_date = None
            if row_date is not None and row_date <= date.today():
                return True
        return False

    @staticmethod
    def _calendar_row_for_surprise(
        calendar_by_date: dict[str, dict[str, Any]],
        *,
        surprise_row: dict[str, Any],
        report_date: date,
    ) -> dict[str, Any]:
        exact = calendar_by_date.get(report_date.isoformat())
        if exact:
            return exact

        quarter = surprise_row.get("quarter")
        year = surprise_row.get("year")
        if quarter is None or year is None:
            return {}

        target = (int(quarter), int(year))
        best_row: dict[str, Any] | None = None
        best_delta: int | None = None
        for date_str, row in calendar_by_date.items():
            row_quarter = row.get("quarter")
            row_year = row.get("year")
            if row_quarter is None or row_year is None:
                continue
            if (int(row_quarter), int(row_year)) != target:
                continue
            try:
                cal_date = date.fromisoformat(date_str)
            except ValueError:
                continue
            delta = abs((cal_date - report_date).days)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_row = row
        if best_row is not None and best_delta is not None and best_delta <= 120:
            return best_row
        return {}

    @staticmethod
    def _surprise_for_calendar_row(
        raw_surprises: list[Any],
        *,
        calendar_row: dict[str, Any],
    ) -> dict[str, Any] | None:
        quarter = calendar_row.get("quarter")
        year = calendar_row.get("year")
        if quarter is None or year is None:
            return None
        target = (int(quarter), int(year))
        for item in raw_surprises:
            if not isinstance(item, dict):
                continue
            item_quarter = item.get("quarter")
            item_year = item.get("year")
            if item_quarter is None or item_year is None:
                continue
            if (int(item_quarter), int(item_year)) == target:
                return item
        return None

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
