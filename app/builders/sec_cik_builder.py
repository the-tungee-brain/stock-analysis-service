from dataclasses import dataclass

from fastapi import HTTPException

from app.adapters.sec.sec_edgar_adapter import SecEdgarAdapter
from app.models.sec_research_models import (
    SecFilingsResponse,
    SecFilingSummary,
    SecLookupResponse,
)


@dataclass(frozen=True)
class ResolvedSecCompany:
    symbol: str
    cik_int: int
    cik: str
    ticker_title: str | None = None


class SecCikBuilder:
    def __init__(self, sec_edgar_adapter: SecEdgarAdapter) -> None:
        self.sec_edgar_adapter = sec_edgar_adapter
        self._ticker_to_entry: dict[str, dict] | None = None

    def _load_ticker_map(self) -> dict[str, dict]:
        if self._ticker_to_entry is not None:
            return self._ticker_to_entry

        raw = self.sec_edgar_adapter.get_company_tickers()
        mapping: dict[str, dict] = {}
        for entry in raw.values():
            ticker = str(entry.get("ticker", "")).upper().strip()
            if ticker:
                mapping[ticker] = entry
        self._ticker_to_entry = mapping
        return mapping

    def resolve_symbol(self, symbol: str) -> ResolvedSecCompany:
        key = symbol.upper().strip()
        if not key:
            raise HTTPException(status_code=400, detail="Symbol is required")

        ticker_map = self._load_ticker_map()
        entry = ticker_map.get(key)
        if not entry:
            raise HTTPException(
                status_code=404,
                detail=f"No SEC CIK found for symbol '{key}'",
            )

        cik_int = int(entry["cik_str"])
        return ResolvedSecCompany(
            symbol=key,
            cik_int=cik_int,
            cik=SecEdgarAdapter.format_cik(cik_int),
            ticker_title=entry.get("title"),
        )

    def build_lookup(self, symbol: str) -> SecLookupResponse:
        resolved = self.resolve_symbol(symbol=symbol)
        submissions = self.sec_edgar_adapter.get_submissions(cik=resolved.cik_int)

        return SecLookupResponse(
            symbol=resolved.symbol,
            cik=resolved.cik,
            cik_int=resolved.cik_int,
            name=submissions.get("name") or resolved.ticker_title or resolved.symbol,
            tickers=submissions.get("tickers") or [resolved.symbol],
            exchanges=submissions.get("exchanges") or [],
            sic=submissions.get("sic"),
            sic_description=submissions.get("sicDescription"),
            fiscal_year_end=submissions.get("fiscalYearEnd"),
            state_of_incorporation=submissions.get("stateOfIncorporation"),
            category=submissions.get("category"),
            entity_type=submissions.get("entityType"),
        )

    def build_filings(self, symbol: str, limit: int = 20) -> SecFilingsResponse:
        resolved = self.resolve_symbol(symbol=symbol)
        submissions = self.sec_edgar_adapter.get_submissions(cik=resolved.cik_int)
        recent = submissions.get("filings", {}).get("recent", {})

        accession_numbers = recent.get("accessionNumber") or []
        filing_dates = recent.get("filingDate") or []
        report_dates = recent.get("reportDate") or []
        forms = recent.get("form") or []
        primary_docs = recent.get("primaryDocument") or []

        filings: list[SecFilingSummary] = []
        for idx in range(min(limit, len(accession_numbers))):
            filings.append(
                SecFilingSummary(
                    accession_number=accession_numbers[idx],
                    filing_date=filing_dates[idx] if idx < len(filing_dates) else "",
                    report_date=report_dates[idx] if idx < len(report_dates) else "",
                    form=forms[idx] if idx < len(forms) else "",
                    primary_document=(
                        primary_docs[idx] if idx < len(primary_docs) else None
                    ),
                )
            )

        return SecFilingsResponse(
            symbol=resolved.symbol,
            cik=resolved.cik,
            filings=filings,
        )
