from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime, timedelta, timezone


class Instrument(BaseModel):
    assetType: Literal["EQUITY", "OPTION"]
    cusip: str
    symbol: str
    description: Optional[str] = None
    netChange: Optional[float] = None
    type: Optional[str] = None
    putCall: Optional[Literal["CALL", "PUT"]] = None
    underlyingSymbol: Optional[str] = None


class Position(BaseModel):
    shortQuantity: float
    averagePrice: float
    currentDayProfitLoss: float
    currentDayProfitLossPercentage: float
    longQuantity: float
    settledLongQuantity: float
    settledShortQuantity: float
    instrument: Instrument
    marketValue: float
    maintenanceRequirement: float
    averageLongPrice: Optional[float] = None
    taxLotAverageLongPrice: Optional[float] = None
    longOpenProfitLoss: Optional[float] = None
    previousSessionLongQuantity: Optional[float] = None
    averageShortPrice: Optional[float] = None
    taxLotAverageShortPrice: Optional[float] = None
    shortOpenProfitLoss: Optional[float] = None
    previousSessionShortQuantity: Optional[float] = None
    currentDayCost: float


class InitialBalances(BaseModel):
    accruedInterest: float
    availableFundsNonMarginableTrade: float
    bondValue: float
    buyingPower: float
    cashBalance: float
    cashAvailableForTrading: float
    cashReceipts: float
    dayTradingBuyingPower: float
    dayTradingBuyingPowerCall: float
    dayTradingEquityCall: float
    equity: float
    equityPercentage: float
    liquidationValue: float
    longMarginValue: float
    longOptionMarketValue: float
    longStockValue: float
    maintenanceCall: float
    maintenanceRequirement: float
    margin: float
    marginEquity: float
    moneyMarketFund: float
    mutualFundValue: float
    regTCall: float
    shortMarginValue: float
    shortOptionMarketValue: float
    shortStockValue: float
    totalCash: float
    isInCall: bool
    pendingDeposits: float
    marginBalance: float
    shortBalance: float
    accountValue: float


class CurrentBalances(BaseModel):
    accruedInterest: float
    cashBalance: float
    cashReceipts: float
    longOptionMarketValue: float
    liquidationValue: float
    longMarketValue: float
    moneyMarketFund: float
    savings: float
    shortMarketValue: float
    pendingDeposits: float
    mutualFundValue: float
    bondValue: float
    shortOptionMarketValue: float
    availableFunds: float
    availableFundsNonMarginableTrade: float
    buyingPower: float
    buyingPowerNonMarginableTrade: float
    dayTradingBuyingPower: float
    equity: float
    equityPercentage: float
    longMarginValue: float
    maintenanceCall: float
    maintenanceRequirement: float
    marginBalance: float
    regTCall: float
    shortBalance: float
    shortMarginValue: float
    sma: float


class ProjectedBalances(BaseModel):
    availableFunds: float
    availableFundsNonMarginableTrade: float
    buyingPower: float
    dayTradingBuyingPower: float
    dayTradingBuyingPowerCall: float
    maintenanceCall: float
    regTCall: float
    isInCall: bool
    stockBuyingPower: float


class SecuritiesAccount(BaseModel):
    type: Literal["MARGIN"]
    accountNumber: str
    roundTrips: int
    isDayTrader: bool
    isClosingOnlyRestricted: bool
    pfcbFlag: bool
    positions: List[Position]
    initialBalances: InitialBalances
    currentBalances: CurrentBalances
    projectedBalances: ProjectedBalances


class AggregatedBalance(BaseModel):
    currentLiquidationValue: float
    liquidationValue: float


class SchwabAccounts(BaseModel):
    securitiesAccount: SecuritiesAccount
    aggregatedBalance: AggregatedBalance


class SchwabAccessTokenResponse(BaseModel):
    expires_in: int
    token_type: str
    scope: str
    refresh_token: str
    access_token: str
    id_token: Optional[str] = None
    access_expires_at: Optional[datetime] = None
    refresh_expires_at: Optional[datetime] = None

    def set_expiration(self) -> None:
        now = datetime.now(timezone.utc)
        self.access_expires_at = now + timedelta(seconds=self.expires_in)
        self.refresh_expires_at = now + timedelta(days=7)

    def is_access_token_expired(self) -> bool:
        if not self.access_expires_at:
            return True

        buffer = timedelta(seconds=60)

        return datetime.now(timezone.utc) >= (self.access_expires_at - buffer)

    def is_refresh_token_expired(self) -> bool:
        if not self.refresh_expires_at:
            return True
        buffer = timedelta(seconds=60)
        return datetime.now(timezone.utc) >= (self.refresh_expires_at - buffer)


class SchwabAuthTokenItem(BaseModel):
    id: Optional[int] = Field(default=None, description="Auto-generated ID (IDENTITY)")
    user_id: str = Field(..., max_length=100, description="Unique user identifier")
    access_token: str = Field(..., description="Schwab access token")
    refresh_token: Optional[str] = Field(
        default=None, description="Schwab refresh token"
    )
    access_expires_at: datetime = Field(
        ..., description="Access token expiry (TIMESTAMP WITH TIME ZONE)"
    )
    refresh_expires_at: Optional[datetime] = Field(
        default=None, description="Refresh token expiry (TIMESTAMP WITH TIME ZONE)"
    )
    created_at: Optional[datetime] = Field(
        default=None, description="Record created timestamp"
    )
    updated_at: Optional[datetime] = Field(
        default=None, description="Record updated timestamp"
    )

    class Config:
        from_attributes = True
        populate_by_name = True
