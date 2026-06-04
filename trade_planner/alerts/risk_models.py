"""Risk gating models for educational trade-plan alerts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from trade_planner.models import TradePlan
from trade_planner.research.models import MarketRegime


class AlertGateAction(str, Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    WARN = "WARN"
    SIZE_DOWN = "SIZE_DOWN"


class AlertPriority(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True, slots=True)
class AlertRiskSettings:
    max_open_positions: int = 5
    max_risk_per_trade_pct: float = 0.01
    max_total_open_risk_pct: float = 0.06
    consecutive_loss_limit: int = 4
    rolling_window_trades: int = 20
    rolling_drawdown_limit_pct: float = -0.10
    mega_cap_correlation_threshold: int = 3
    volume_climax_ratio: float = 3.0


@dataclass(frozen=True, slots=True)
class OpenTradeSnapshot:
    symbol: str
    setup_name: str
    entry_price: float
    stop_price: float
    direction: str = "LONG"
    """Optional precomputed risk as fraction of account (e.g. 0.01 = 1%)."""
    position_risk_pct: float | None = None


@dataclass(frozen=True, slots=True)
class ClosedTradeSnapshot:
    setup_name: str
    return_pct: float
    symbol: str = ""


@dataclass(frozen=True, slots=True)
class AlertRiskContext:
    candidate_plan: TradePlan
    open_trades: tuple[OpenTradeSnapshot, ...] = ()
    recent_closed: tuple[ClosedTradeSnapshot, ...] = ()
    current_symbol: str = ""
    sector_or_group: str | None = None
    market_regime: MarketRegime | None = None
    volume_ratio: float | None = None
    account_equity_usd: float | None = None
    settings: AlertRiskSettings = field(default_factory=AlertRiskSettings)


@dataclass(frozen=True, slots=True)
class AlertDecision:
    allowed: bool
    action: AlertGateAction
    reasons: tuple[str, ...]
    recommended_position_risk_pct: float
    max_shares_or_dollars: float | None
    alert_priority: AlertPriority

    @property
    def is_educational_only(self) -> bool:
        return True
