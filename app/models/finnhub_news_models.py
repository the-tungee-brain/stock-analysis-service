from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, HttpUrl, RootModel, field_validator


class NewsItem(BaseModel):
    category: str
    datetime: datetime
    headline: str
    id: int
    image: Optional[HttpUrl] = None
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

    @field_validator("image", mode="before")
    @classmethod
    def empty_image_to_none(cls, v):
        if v == "" or v is None:
            return None
        return v


class NewsResponse(RootModel[List[NewsItem]]):
    pass
