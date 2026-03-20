from typing import Dict, Optional
from pydantic import BaseModel, RootModel, Field


class Reference(BaseModel):
    cusip: Optional[str] = None
    description: str
    exchange: str
    exchangeName: str
    contractType: Optional[str] = None
    daysToExpiration: Optional[int] = None
    expirationDay: Optional[int] = None
    expirationMonth: Optional[int] = None
    expirationYear: Optional[int] = None
    isPennyPilot: Optional[bool] = None
    lastTradingDay: Optional[int] = None
    multiplier: Optional[int] = None
    settlementType: Optional[str] = None
    strikePrice: Optional[float] = None
    underlying: Optional[str] = None
    uvExpirationType: Optional[str] = None
    otcMarketTier: Optional[str] = None
    futureActiveSymbol: Optional[str] = None
    futureExpirationDate: Optional[int] = None
    futureIsActive: Optional[bool] = None
    futureIsTradable: Optional[bool] = None
    futureMultiplier: Optional[float] = None
    futurePriceFormat: Optional[str] = None
    futureSettlementPrice: Optional[float] = None
    futureTradingHours: Optional[str] = None
    product: Optional[str] = None
    isTradable: Optional[bool] = None
    marketMaker: Optional[str] = None
    tradingHours: Optional[str] = None


class Quote(BaseModel):
    week_high_52: Optional[float] = Field(None, alias="52WeekHigh")
    week_low_52: Optional[float] = Field(None, alias="52WeekLow")
    askMICId: Optional[str] = None
    askPrice: Optional[float] = None
    askSize: Optional[int] = None
    askTime: Optional[int] = None
    bidMICId: Optional[str] = None
    bidPrice: Optional[float] = None
    bidSize: Optional[int] = None
    bidTime: Optional[int] = None
    closePrice: Optional[float] = None
    highPrice: Optional[float] = None
    lastMICId: Optional[str] = None
    lastPrice: Optional[float] = None
    lastSize: Optional[int] = None
    lowPrice: Optional[float] = None
    mark: Optional[float] = None
    markChange: Optional[float] = None
    markPercentChange: Optional[float] = None
    netChange: Optional[float] = None
    netPercentChange: Optional[float] = None
    openPrice: Optional[float] = None
    quoteTime: Optional[int] = None
    securityStatus: Optional[str] = None
    totalVolume: Optional[int] = None
    tradeTime: Optional[int] = None
    volatility: Optional[float] = None

    nAV: Optional[float] = None

    futurePercentChange: Optional[float] = None
    settleTime: Optional[int] = None
    tick: Optional[float] = None
    tickAmount: Optional[float] = None
    openInterest: Optional[int] = None

    delta: Optional[float] = None
    gamma: Optional[float] = None
    impliedYield: Optional[float] = None
    indAskPrice: Optional[float] = None
    indBidPrice: Optional[float] = None
    indQuoteTime: Optional[int] = None
    rho: Optional[float] = None
    theoreticalOptionValue: Optional[float] = None
    theta: Optional[float] = None
    timeValue: Optional[float] = None
    underlyingPrice: Optional[float] = None
    vega: Optional[float] = None

    tickAmount: Optional[float] = None


class Regular(BaseModel):
    regularMarketLastPrice: Optional[float] = None
    regularMarketLastSize: Optional[int] = None
    regularMarketNetChange: Optional[float] = None
    regularMarketPercentChange: Optional[float] = None
    regularMarketTradeTime: Optional[int] = None


class Fundamental(BaseModel):
    avg10DaysVolume: Optional[int] = None
    avg1YearVolume: Optional[int] = None
    declarationDate: Optional[str] = None
    divAmount: Optional[float] = None
    divExDate: Optional[str] = None
    divFreq: Optional[int] = None
    divPayAmount: Optional[float] = None
    divPayDate: Optional[str] = None
    divYield: Optional[float] = None
    eps: Optional[float] = None
    fundLeverageFactor: Optional[float] = None
    fundStrategy: Optional[str] = None
    nextDivExDate: Optional[str] = None
    nextDivPayDate: Optional[str] = None
    peRatio: Optional[float] = None


class InstrumentQuote(BaseModel):
    assetMainType: str
    symbol: str
    realtime: bool
    ssid: int
    quoteType: Optional[str] = None
    assetSubType: Optional[str] = None

    reference: Reference
    quote: Quote
    regular: Optional[Regular] = None
    fundamental: Optional[Fundamental] = None


class QuotesResponse(RootModel[Dict[str, InstrumentQuote]]):
    pass


class PromptQuoteSnapshot(BaseModel):
    symbol: str
    asset_main_type: str
    asset_sub_type: Optional[str]
    description: str
    last: Optional[float]
    net_change: Optional[float]
    net_change_pct: Optional[float]
    high_52w: Optional[float]
    low_52w: Optional[float]
    volume: Optional[int]
    avg_10d_volume: Optional[int]
    avg_1y_volume: Optional[int]
    implied_vol: Optional[float]
