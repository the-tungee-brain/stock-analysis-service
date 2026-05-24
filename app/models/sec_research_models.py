from pydantic import BaseModel, Field
from typing import Literal


class SecLookupResponse(BaseModel):
    symbol: str
    cik: str
    cik_int: int
    name: str
    tickers: list[str] = Field(default_factory=list)
    exchanges: list[str] = Field(default_factory=list)
    sic: str | None = None
    sic_description: str | None = None
    fiscal_year_end: str | None = None
    state_of_incorporation: str | None = None
    category: str | None = None
    entity_type: str | None = None


class SecFilingSummary(BaseModel):
    accession_number: str
    filing_date: str
    report_date: str
    form: str
    primary_document: str | None = None


class SecFilingsResponse(BaseModel):
    symbol: str
    cik: str
    filings: list[SecFilingSummary]


class FinancialObservation(BaseModel):
    end: str
    start: str | None = None
    value: float
    fiscal_year: int | None = None
    fiscal_period: str
    form: str
    filed: str


class FinancialLineItem(BaseModel):
    tag: str
    label: str
    unit: str
    observations: list[FinancialObservation]


class SecFinancialsResponse(BaseModel):
    symbol: str
    cik: str
    entity_name: str
    period: Literal["annual", "quarterly"]
    currency: str
    income_statement: list[FinancialLineItem]
    balance_sheet: list[FinancialLineItem]
    cash_flow: list[FinancialLineItem]


class RatioSnapshot(BaseModel):
    end: str
    fiscal_period: str
    fiscal_year: int | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    roe: float | None = None
    roa: float | None = None
    debt_to_equity: float | None = None
    free_cash_flow: float | None = None
    fcf_margin: float | None = None
    revenue_growth_yoy: float | None = None
    net_income_growth_yoy: float | None = None


class SecRatiosResponse(BaseModel):
    symbol: str
    cik: str
    entity_name: str
    period: Literal["annual", "quarterly"]
    snapshots: list[RatioSnapshot]
