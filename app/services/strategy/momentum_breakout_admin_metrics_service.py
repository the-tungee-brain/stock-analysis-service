"""Admin operational metrics for Momentum Breakout rollout."""

from __future__ import annotations

from trade_planner.alerts.lifecycle_store import MomentumBreakoutAlertStore
from trade_planner.alerts.paper_trade_store import PaperTradePerformanceStore
from app.services.strategy.momentum_breakout_ops_metrics import (
    MomentumBreakoutAdminMetricsSnapshot,
    MomentumBreakoutOpsMetricsRegistry,
    build_admin_metrics_snapshot,
    get_momentum_breakout_ops_metrics,
)


class MomentumBreakoutAdminMetricsService:
    def __init__(
        self,
        *,
        alert_store: MomentumBreakoutAlertStore,
        paper_store: PaperTradePerformanceStore,
        registry: MomentumBreakoutOpsMetricsRegistry | None = None,
    ) -> None:
        self._alert_store = alert_store
        self._paper_store = paper_store
        self._registry = registry or get_momentum_breakout_ops_metrics()

    @property
    def snapshot(self) -> MomentumBreakoutAdminMetricsSnapshot:
        return build_admin_metrics_snapshot(
            registry=self._registry,
            alert_store=self._alert_store,
            paper_store=self._paper_store,
        )
