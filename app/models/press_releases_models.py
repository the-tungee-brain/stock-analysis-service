from pydantic import BaseModel, ConfigDict, Field

from app.models.company_research_models import NewsHeadline

_CONFIG = ConfigDict(populate_by_name=True)


class PressReleasesResponse(BaseModel):
    model_config = _CONFIG

    symbol: str
    items: list[NewsHeadline] = Field(default_factory=list)
