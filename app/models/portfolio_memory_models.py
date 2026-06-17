from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.prompts import AnalysisAction
from app.models.intelligence_models import (
    IntelligenceSignal,
    PortfolioDigest,
    ProactiveAlert,
)

_MEMORY_MODEL_CONFIG = ConfigDict(populate_by_name=True)

AlertStatus = Literal["active", "resolved", "dismissed"]


class SnapshotPosition(BaseModel):
    model_config = _MEMORY_MODEL_CONFIG

    symbol: str
    asset_type: str = Field(serialization_alias="assetType")
    quantity: float
    market_value: float = Field(serialization_alias="marketValue")
    weight_pct: float = Field(serialization_alias="weightPct")
    day_pnl: float | None = Field(default=None, serialization_alias="dayPnl")
    day_pnl_pct: float | None = Field(default=None, serialization_alias="dayPnlPct")
    pnl: float | None = None
    pnl_pct: float | None = Field(default=None, serialization_alias="pnlPct")
    option_strategy: str | None = Field(
        default=None, serialization_alias="optionStrategy"
    )
    strike: float | None = None
    expiration: str | None = None
    put_call: str | None = Field(default=None, serialization_alias="putCall")


class PortfolioSnapshotSummary(BaseModel):
    model_config = _MEMORY_MODEL_CONFIG

    alert_count: int = Field(default=0, serialization_alias="alertCount")
    signal_count: int = Field(default=0, serialization_alias="signalCount")
    position_count: int = Field(default=0, serialization_alias="positionCount")
    sector_weights: dict[str, float] = Field(
        default_factory=dict, serialization_alias="sectorWeights"
    )
    diversification_score: int | None = Field(
        default=None, serialization_alias="diversificationScore"
    )
    diversification_rating: str | None = Field(
        default=None, serialization_alias="diversificationRating"
    )


class PortfolioSnapshotRecord(BaseModel):
    model_config = _MEMORY_MODEL_CONFIG

    id: str | None = None
    user_id: str = Field(serialization_alias="userId")
    snapshot_date: date = Field(serialization_alias="snapshotDate")
    account_number: str | None = Field(default=None, serialization_alias="accountNumber")
    liquidation_value: float | None = Field(
        default=None, serialization_alias="liquidationValue"
    )
    cash_balance: float | None = Field(default=None, serialization_alias="cashBalance")
    positions: list[SnapshotPosition] = Field(default_factory=list)
    summary: PortfolioSnapshotSummary | None = None
    created_at: datetime | None = Field(default=None, serialization_alias="createdAt")


class PositionWeightChange(BaseModel):
    model_config = _MEMORY_MODEL_CONFIG

    symbol: str
    previous_weight_pct: float = Field(serialization_alias="previousWeightPct")
    current_weight_pct: float = Field(serialization_alias="currentWeightPct")
    change_pct: float = Field(serialization_alias="changePct")


class PortfolioChanges(BaseModel):
    model_config = _MEMORY_MODEL_CONFIG

    from_date: date | None = Field(default=None, serialization_alias="fromDate")
    to_date: date | None = Field(default=None, serialization_alias="toDate")
    liquidation_value_change: float | None = Field(
        default=None, serialization_alias="liquidationValueChange"
    )
    liquidation_value_change_pct: float | None = Field(
        default=None, serialization_alias="liquidationValueChangePct"
    )
    new_symbols: list[str] = Field(default_factory=list, serialization_alias="newSymbols")
    removed_symbols: list[str] = Field(
        default_factory=list, serialization_alias="removedSymbols"
    )
    weight_changes: list[PositionWeightChange] = Field(
        default_factory=list, serialization_alias="weightChanges"
    )
    summary: str | None = None


class AlertHistoryItem(BaseModel):
    model_config = _MEMORY_MODEL_CONFIG

    id: str
    fingerprint: str
    action: AnalysisAction
    label: str
    symbol: str | None = None
    reason: str
    priority: int
    status: AlertStatus
    first_seen_at: datetime = Field(serialization_alias="firstSeenAt")
    last_seen_at: datetime = Field(serialization_alias="lastSeenAt")
    days_active: int = Field(serialization_alias="daysActive")


class AttentionItem(BaseModel):
    model_config = _MEMORY_MODEL_CONFIG

    action: AnalysisAction
    label: str
    symbol: str | None = None
    reason: str
    priority: int
    source: Literal["current", "historical"] = "current"
    first_seen_at: datetime | None = Field(
        default=None, serialization_alias="firstSeenAt"
    )
    days_active: int | None = Field(default=None, serialization_alias="daysActive")
    alert_id: str | None = Field(default=None, serialization_alias="alertId")


class MorningBriefMover(BaseModel):
    model_config = _MEMORY_MODEL_CONFIG

    symbol: str
    day_pnl: float | None = Field(default=None, serialization_alias="dayPnl")
    day_pnl_pct: float | None = Field(default=None, serialization_alias="dayPnlPct")


class MorningBriefSnapshot(BaseModel):
    model_config = _MEMORY_MODEL_CONFIG

    portfolio_value: float | None = Field(
        default=None, serialization_alias="portfolioValue"
    )
    day_pnl: float | None = Field(default=None, serialization_alias="dayPnl")
    day_pnl_pct: float | None = Field(default=None, serialization_alias="dayPnlPct")
    cash_available: float | None = Field(
        default=None, serialization_alias="cashAvailable"
    )
    diversification_score: int | None = Field(
        default=None, serialization_alias="diversificationScore"
    )
    diversification_rating: str | None = Field(
        default=None, serialization_alias="diversificationRating"
    )
    biggest_winner: MorningBriefMover | None = Field(
        default=None, serialization_alias="biggestWinner"
    )
    biggest_loser: MorningBriefMover | None = Field(
        default=None, serialization_alias="biggestLoser"
    )


class MorningBrief(BaseModel):
    model_config = _MEMORY_MODEL_CONFIG

    generated_at: datetime = Field(serialization_alias="generatedAt")
    snapshot: MorningBriefSnapshot | None = None
    macro_regime: str | None = Field(default=None, serialization_alias="macroRegime")
    digest: PortfolioDigest | None = None
    changes: PortfolioChanges | None = None
    signals: list[IntelligenceSignal] = Field(default_factory=list)
    top_alerts: list[ProactiveAlert] = Field(
        default_factory=list, serialization_alias="topAlerts"
    )
    attention_queue: list[AttentionItem] = Field(
        default_factory=list, serialization_alias="attentionQueue"
    )
    delivery_ready: bool = Field(default=True, serialization_alias="deliveryReady")
