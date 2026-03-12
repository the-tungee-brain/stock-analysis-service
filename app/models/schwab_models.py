from pydantic import BaseModel
from typing import Optional, List, Literal
from datetime import datetime, timedelta, timezone


class Instrument(BaseModel):
    assetType: Literal["EQUITY", "OPTION"]
    cusip: str
    symbol: str
    description: Optional[str] = None
    netChange: float
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
    expires_at: Optional[datetime] = None

    def set_expiration(self):
        self.expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=self.expires_in
        )

    def is_expired(self) -> bool:
        if not self.expires_at:
            return True

        buffer = timedelta(seconds=60)

        return datetime.now(timezone.utc) >= (self.expires_at - buffer)
