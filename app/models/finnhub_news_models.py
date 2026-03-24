from datetime import datetime
from typing import List
from pydantic import RootModel, BaseModel, HttpUrl, field_validator


class NewsItem(BaseModel):
    category: str
    datetime: datetime
    headline: str
    id: int
    image: HttpUrl
    related: str
    source: str
    summary: str
    url: HttpUrl

    @field_validator("datetime", mode="before")
    @classmethod
    def parse_unix_ts(cls, v):
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v)
        return v


class NewsResponse(RootModel[List[NewsItem]]):
    pass
