from pydantic import BaseModel, Field


class DividendPaymentItem(BaseModel):
    date: str
    amount_per_share: float = Field(serialization_alias="amountPerShare")


class AnnualDividendIncome(BaseModel):
    year: int
    total_per_share: float = Field(serialization_alias="totalPerShare")
    income_on_shares: float = Field(serialization_alias="incomeOnShares")
    is_partial_year: bool = Field(default=False, serialization_alias="isPartialYear")


class DividendSnowballScenario(BaseModel):
    shares: float
    start_year: int = Field(serialization_alias="startYear")
    total_collected: float = Field(serialization_alias="totalCollected")
    annual_income_latest: float = Field(serialization_alias="annualIncomeLatest")
    annual_income_start: float = Field(serialization_alias="annualIncomeStart")
    latest_year: int = Field(serialization_alias="latestYear")


class DividendHistoryContext(BaseModel):
    ticker: str
    total_dividends: int = Field(serialization_alias="totalDividends")
    total_splits: int = Field(default=0, serialization_alias="totalSplits")
    consecutive_annual_increases: int = Field(
        default=0, serialization_alias="consecutiveAnnualIncreases"
    )
    cagr_5y_pct: float | None = Field(default=None, serialization_alias="cagr5yPct")
    cagr_10y_pct: float | None = Field(default=None, serialization_alias="cagr10yPct")
    annual_income: list[AnnualDividendIncome] = Field(
        default_factory=list, serialization_alias="annualIncome"
    )
    recent_payments: list[DividendPaymentItem] = Field(
        default_factory=list, serialization_alias="recentPayments"
    )
    payments: list[DividendPaymentItem] = Field(default_factory=list)
    scenario: DividendSnowballScenario
    data_as_of: str | None = Field(default=None, serialization_alias="dataAsOf")
    confidence_score: float | None = Field(
        default=None, serialization_alias="confidenceScore"
    )
