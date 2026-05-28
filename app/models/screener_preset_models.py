from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

_SCREENER_PRESET_CONFIG = ConfigDict(extra="forbid", populate_by_name=True)


EquityClauseOp = Literal["eq", "gt", "gte", "lt", "lte", "is-in", "btwn"]


class EquityQueryClause(BaseModel):
    model_config = _SCREENER_PRESET_CONFIG

    op: EquityClauseOp
    field: str
    value: float | str | None = None
    values: list[str | float] | None = None


class EquityQuerySpec(BaseModel):
    model_config = _SCREENER_PRESET_CONFIG

    operator: Literal["and", "or"] = "and"
    clauses: list[EquityQueryClause]


class FundUniverseSpec(BaseModel):
    model_config = _SCREENER_PRESET_CONFIG

    region: str = "us"
    asset_class: str = "equity"


class ScreenerPreset(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str
    label: str
    description: str
    equity_query: EquityQuerySpec | None = None
    fund_universe: FundUniverseSpec | None = None
    post_filters: dict[str, Any] = Field(default_factory=dict)


class ScreenerPresetSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    label: str
    description: str
    post_filters: dict[str, Any] = Field(default_factory=dict, alias="postFilters")
    post_filter_status: str = Field(
        default="metadata_only",
        alias="postFilterStatus",
        description=(
            "metadata_only: post-filters returned for display/future enrichment; "
            "not applied in this screener pass"
        ),
    )
