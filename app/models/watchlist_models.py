from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class WatchlistSymbolRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    ticker: str
    sort_order: int = Field(default=0, alias="sortOrder")
    created_at: datetime | None = Field(default=None, alias="createdAt")


class WatchlistFolderRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    icon_name: str = Field(default="folder.fill", alias="iconName")
    swatch_id: str = Field(default="slate", alias="swatchID")
    accent_hex: int | None = Field(default=None, alias="accentHex")
    is_pinned: bool = Field(default=False, alias="isPinned")
    is_collapsed: bool = Field(default=False, alias="isCollapsed")
    sort_order: int = Field(default=0, alias="sortOrder")
    created_at: datetime | None = Field(default=None, alias="createdAt")
    symbols: list[WatchlistSymbolRecord] = Field(default_factory=list)


class WatchlistWorkspaceSyncRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    folders: list[WatchlistFolderRecord] = Field(default_factory=list)
    base_version: int | None = Field(default=None, alias="baseVersion")


class WatchlistSymbolResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    ticker: str
    sort_order: int = Field(default=0, alias="sortOrder")
    company_name: str = Field(alias="companyName")
    price: float | None = None
    day_change: float | None = Field(default=None, alias="dayChange")
    day_change_percent: float | None = Field(default=None, alias="dayChangePercent")
    created_at: datetime | None = Field(default=None, alias="createdAt")


class WatchlistFolderResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    icon_name: str = Field(alias="iconName")
    swatch_id: str = Field(alias="swatchID")
    accent_hex: int | None = Field(default=None, alias="accentHex")
    is_pinned: bool = Field(alias="isPinned")
    is_collapsed: bool = Field(alias="isCollapsed")
    sort_order: int = Field(alias="sortOrder")
    created_at: datetime = Field(alias="createdAt")
    symbols: list[WatchlistSymbolResponse] = Field(default_factory=list)


class WatchlistWorkspaceResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    folders: list[WatchlistFolderResponse] = Field(default_factory=list)
    as_of: datetime = Field(alias="asOf")
    workspace_version: int = Field(default=0, alias="workspaceVersion")
