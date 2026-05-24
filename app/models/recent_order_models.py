from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.core.prompts import AnalysisAction


class RecentOrderLegEntry(BaseModel):
    leg_id: Optional[int] = Field(default=None, serialization_alias="legId")
    instruction: str
    quantity: Optional[float] = None
    asset_type: Optional[str] = Field(default=None, serialization_alias="assetType")
    option_symbol: Optional[str] = Field(default=None, serialization_alias="optionSymbol")
    underlying_symbol: Optional[str] = Field(
        default=None, serialization_alias="underlyingSymbol"
    )
    strike: Optional[float] = None
    expiration: Optional[date] = None
    put_call: Optional[str] = Field(default=None, serialization_alias="putCall")
    contract_label: Optional[str] = Field(
        default=None, serialization_alias="contractLabel"
    )
    average_fill_price: Optional[float] = Field(
        default=None, serialization_alias="averageFillPrice"
    )
    premium_per_contract: Optional[float] = Field(
        default=None, serialization_alias="premiumPerContract"
    )
    total_cash: Optional[float] = Field(default=None, serialization_alias="totalCash")
    position_effect: Optional[str] = Field(
        default=None, serialization_alias="positionEffect"
    )

    model_config = {"populate_by_name": True}


class RecentOrderEntry(BaseModel):
    order_id: Optional[int] = Field(default=None, serialization_alias="orderId")
    symbol: str
    fill_time: Optional[datetime] = Field(default=None, serialization_alias="fillTime")
    side: str
    quantity: Optional[float] = None
    average_fill_price: Optional[float] = Field(
        default=None, serialization_alias="averageFillPrice"
    )
    order_type: Optional[str] = Field(default=None, serialization_alias="orderType")
    position_effect: Optional[str] = Field(
        default=None, serialization_alias="positionEffect"
    )
    tax_lot_method: Optional[str] = Field(
        default=None, serialization_alias="taxLotMethod"
    )
    asset_type: Optional[str] = Field(default=None, serialization_alias="assetType")
    description: Optional[str] = None
    premium_per_contract: Optional[float] = Field(
        default=None, serialization_alias="premiumPerContract"
    )
    total_premium: Optional[float] = Field(default=None, serialization_alias="totalPremium")
    total_cash: Optional[float] = Field(default=None, serialization_alias="totalCash")
    leg_count: int = Field(default=1, serialization_alias="legCount")
    strategy_label: Optional[str] = Field(
        default=None, serialization_alias="strategyLabel"
    )
    contract_label: Optional[str] = Field(
        default=None, serialization_alias="contractLabel"
    )
    strike: Optional[float] = None
    expiration: Optional[date] = None
    put_call: Optional[str] = Field(default=None, serialization_alias="putCall")
    legs: List[RecentOrderLegEntry] = Field(default_factory=list)
    activity_group_id: Optional[str] = Field(
        default=None, serialization_alias="activityGroupId"
    )
    activity_group_kind: Optional[str] = Field(
        default=None, serialization_alias="activityGroupKind"
    )
    activity_group_label: Optional[str] = Field(
        default=None, serialization_alias="activityGroupLabel"
    )

    model_config = {"populate_by_name": True}


class SuggestedAnalysisAction(BaseModel):
    action: AnalysisAction
    label: str
    reason: str
    priority: int


class RecentOrdersResponse(BaseModel):
    days_back: int = Field(serialization_alias="daysBack")
    symbol: Optional[str] = None
    orders: List[RecentOrderEntry]
    suggested_actions: List[SuggestedAnalysisAction] = Field(
        serialization_alias="suggestedActions"
    )
    activity_by_symbol: dict[str, int] = Field(
        default_factory=dict, serialization_alias="activityBySymbol"
    )

    model_config = {"populate_by_name": True}


class RecentActivitySymbolSummary(BaseModel):
    symbol: str
    order_count: int = Field(serialization_alias="orderCount")
    last_fill_time: Optional[datetime] = Field(
        default=None, serialization_alias="lastFillTime"
    )

    model_config = {"populate_by_name": True}


class RecentActivitySummary(BaseModel):
    days_back: int = Field(serialization_alias="daysBack")
    total_orders: int = Field(serialization_alias="totalOrders")
    recent_order_count: int = Field(serialization_alias="recentOrderCount")
    symbols_traded: List[RecentActivitySymbolSummary] = Field(
        serialization_alias="symbolsTraded"
    )
    latest_orders: List[RecentOrderEntry] = Field(serialization_alias="latestOrders")
    suggested_actions: List[SuggestedAnalysisAction] = Field(
        serialization_alias="suggestedActions"
    )

    model_config = {"populate_by_name": True}
