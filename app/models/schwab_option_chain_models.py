from typing import Dict, List, Optional
from pydantic import BaseModel


class OptionDeliverable(BaseModel):
    symbol: Optional[str] = None
    assetType: Optional[str] = None
    deliverableUnits: Optional[str] = None
    currencyType: Optional[str] = None


class OptionContract(BaseModel):
    putCall: str
    symbol: str
    description: Optional[str] = None
    exchangeName: Optional[str] = None

    bidPrice: Optional[float] = None
    askPrice: Optional[float] = None
    lastPrice: Optional[float] = None
    markPrice: Optional[float] = None

    bidSize: Optional[int] = None
    askSize: Optional[int] = None
    lastSize: Optional[int] = None
    highPrice: Optional[float] = None
    lowPrice: Optional[float] = None
    openPrice: Optional[float] = None
    closePrice: Optional[float] = None
    totalVolume: Optional[int] = None

    tradeDate: Optional[int] = None
    quoteTimeInLong: Optional[int] = None
    tradeTimeInLong: Optional[int] = None
    netChange: Optional[float] = None
    volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    timeValue: Optional[float] = None
    openInterest: Optional[int] = None
    isInTheMoney: Optional[bool] = None

    theoreticalOptionValue: Optional[float] = None
    theoreticalVolatility: Optional[float] = None
    isMini: Optional[bool] = None
    isNonStandard: Optional[bool] = None
    optionDeliverablesList: Optional[List[OptionDeliverable]] = None

    strikePrice: float
    expirationDate: str
    daysToExpiration: int
    expirationType: Optional[str] = None
    lastTradingDay: Optional[int] = None
    multiplier: Optional[float] = None
    settlementType: Optional[str] = None
    deliverableNote: Optional[str] = None
    isIndexOption: Optional[bool] = None

    percentChange: Optional[float] = None
    markChange: Optional[float] = None
    markPercentChange: Optional[float] = None
    isPennyPilot: Optional[bool] = None
    intrinsicValue: Optional[float] = None
    optionRoot: Optional[str] = None


class UnderlyingQuote(BaseModel):
    ask: Optional[float] = None
    askSize: Optional[int] = None
    bid: Optional[float] = None
    bidSize: Optional[int] = None
    change: Optional[float] = None
    close: Optional[float] = None
    delayed: Optional[bool] = None
    description: Optional[str] = None
    exchangeName: Optional[str] = None
    fiftyTwoWeekHigh: Optional[float] = None
    fiftyTwoWeekLow: Optional[float] = None
    highPrice: Optional[float] = None
    last: Optional[float] = None
    lowPrice: Optional[float] = None
    mark: Optional[float] = None
    markChange: Optional[float] = None
    markPercentChange: Optional[float] = None
    openPrice: Optional[float] = None
    percentChange: Optional[float] = None
    quoteTime: Optional[int] = None
    symbol: Optional[str] = None
    totalVolume: Optional[int] = None
    tradeTime: Optional[int] = None


class OptionChain(BaseModel):
    symbol: str
    status: Optional[str] = None
    underlying: Optional[UnderlyingQuote] = None
    strategy: Optional[str] = None
    interval: Optional[float] = None
    isDelayed: Optional[bool] = None
    isIndex: Optional[bool] = None
    daysToExpiration: Optional[int] = None
    interestRate: Optional[float] = None
    underlyingPrice: Optional[float] = None
    volatility: Optional[float] = None

    callExpDateMap: Dict[str, Dict[str, List[OptionContract]]] = {}
    putExpDateMap: Dict[str, Dict[str, List[OptionContract]]] = {}
