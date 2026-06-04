"""Paper-trading performance records for live Momentum Breakout alerts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from trade_planner.alerts.lifecycle_models import AlertLifecycleStatus


PAPER_TRADE_TRACKED_STATUSES: frozenset[AlertLifecycleStatus] = frozenset(
    {
        AlertLifecycleStatus.ENTRY_TRIGGERED,
        AlertLifecycleStatus.OPEN,
        AlertLifecycleStatus.TARGET_HIT,
        AlertLifecycleStatus.STOP_HIT,
        AlertLifecycleStatus.EXPIRED,
    }
)

LIVE_PAPER_TRADING_LABEL = "Live paper-trading performance"
LIVE_PAPER_TRADING_DISCLAIMER = (
    "Simulated outcomes from monitored trade-plan levels only. "
    "Not brokerage execution. Past paper results do not guarantee future performance. "
    "Educational information only — not investment advice."
)


@dataclass(frozen=True, slots=True)
class PaperTradePerformanceRecord:
    alert_id: str
    user_id: str
    symbol: str
    setup_name: str
    signal_date: date
    entry_triggered_at: datetime | None
    entry_price: float
    stop_price: float
    target_price: float
    exit_at: datetime | None
    exit_price: float | None
    status: str
    outcome_return_pct: float | None
    holding_days: int | None
    risk_gate_action: str
    market_regime: str | None
    volume_ratio: float | None
    rs_percentile: float | None
    created_at: datetime
