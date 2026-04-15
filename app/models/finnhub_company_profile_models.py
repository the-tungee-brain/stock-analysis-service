from pydantic import BaseModel, HttpUrl
from typing import Optional


class CompanyProfile(BaseModel):
    country: str
    currency: str
    exchange: str
    ipo: str
    marketCapitalization: float
    name: str
    phone: Optional[str] = None
    shareOutstanding: float
    ticker: str
    weburl: HttpUrl
    logo: HttpUrl
    finnhubIndustry: str
