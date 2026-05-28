from pydantic import BaseModel, ConfigDict, Field


class FundWeighting(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    label: str
    weight_pct: float = Field(serialization_alias="weightPct")


class FundTopHolding(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    symbol: str | None = None
    name: str
    weight_pct: float = Field(serialization_alias="weightPct")


class EtfFundsSnapshot(BaseModel):
    """Yahoo Finance ETF/mutual fund profile (yfinance get_funds_data)."""

    model_config = ConfigDict(populate_by_name=True)

    category: str | None = None
    family: str | None = None
    legal_type: str | None = Field(default=None, serialization_alias="legalType")
    description: str | None = None
    expense_ratio_pct: float | None = Field(
        default=None, serialization_alias="expenseRatioPct"
    )
    category_expense_ratio_pct: float | None = Field(
        default=None, serialization_alias="categoryExpenseRatioPct"
    )
    holdings_turnover_pct: float | None = Field(
        default=None, serialization_alias="holdingsTurnoverPct"
    )
    total_net_assets: float | None = Field(
        default=None, serialization_alias="totalNetAssets"
    )
    asset_classes: list[FundWeighting] = Field(
        default_factory=list, serialization_alias="assetClasses"
    )
    sector_weightings: list[FundWeighting] = Field(
        default_factory=list, serialization_alias="sectorWeightings"
    )
    bond_ratings: list[FundWeighting] = Field(
        default_factory=list, serialization_alias="bondRatings"
    )
    top_holdings: list[FundTopHolding] = Field(
        default_factory=list, serialization_alias="topHoldings"
    )
