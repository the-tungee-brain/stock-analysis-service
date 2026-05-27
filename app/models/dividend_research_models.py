from pydantic import BaseModel, Field


class DividendPaymentItem(BaseModel):
    date: str
    amount_per_share: float = Field(serialization_alias="amountPerShare")


class AnnualDividendIncome(BaseModel):
    year: int
    total_per_share: float = Field(serialization_alias="totalPerShare")
    income_on_shares: float = Field(serialization_alias="incomeOnShares")
    is_partial_year: bool = Field(default=False, serialization_alias="isPartialYear")


class DividendAdvancedSnowballScenario(BaseModel):
    enabled: bool = True
    initial_shares: float = Field(serialization_alias="initialShares")
    final_shares: float = Field(serialization_alias="finalShares")
    share_price_at_start: float = Field(serialization_alias="sharePriceAtStart")
    share_price_latest: float = Field(serialization_alias="sharePriceLatest")
    price_cagr_pct: float = Field(serialization_alias="priceCagrPct")
    annual_income_latest_drip: float = Field(serialization_alias="annualIncomeLatestDrip")
    portfolio_value_latest: float = Field(serialization_alias="portfolioValueLatest")
    total_dividends_reinvested: float = Field(
        serialization_alias="totalDividendsReinvested"
    )


class DividendSnowballScenario(BaseModel):
    shares: float
    start_year: int = Field(serialization_alias="startYear")
    total_collected: float = Field(serialization_alias="totalCollected")
    annual_income_latest: float = Field(serialization_alias="annualIncomeLatest")
    annual_income_start: float = Field(serialization_alias="annualIncomeStart")
    latest_year: int = Field(serialization_alias="latestYear")
    investment_usd: float | None = Field(default=None, serialization_alias="investmentUsd")
    share_price: float | None = Field(default=None, serialization_alias="sharePrice")
    advanced: DividendAdvancedSnowballScenario | None = None


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
