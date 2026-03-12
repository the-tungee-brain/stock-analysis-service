from pydantic import BaseModel
from datetime import date


class OptionPosition(BaseModel):
    type: str
    strike: float
    expiration: date
    contracts: int
    entry_price: float
    current_price: float
    implied_volatility: float


class OptionAnalysisRequest(BaseModel):
    underlying_price: float
    option_position: OptionPosition
    days_to_expiration: int
    risk_profile: str
    market_trend: str
