"""Backfill paper-trading rows for existing Momentum Breakout alerts."""

from __future__ import annotations

from dataclasses import dataclass

from trade_planner.alerts.lifecycle_store import MomentumBreakoutAlertStore
from app.services.strategy.paper_trade_performance_service import (
    PaperTradePerformanceService,
)


@dataclass(frozen=True, slots=True)
class PaperTradeBackfillResult:
    alerts_scanned: int
    rows_created: int
    rows_skipped: int
    rows_failed: int
    failures: tuple[str, ...]


class MomentumBreakoutPaperTradeBackfillService:
    def __init__(
        self,
        *,
        alert_store: MomentumBreakoutAlertStore,
        paper_trade_service: PaperTradePerformanceService,
    ) -> None:
        self._alerts = alert_store
        self._paper = paper_trade_service

    def run(
        self,
        *,
        limit: int = 10_000,
        dry_run: bool = False,
    ) -> PaperTradeBackfillResult:
        alerts = self._alerts.list_all_alerts(limit=limit)
        created = 0
        skipped = 0
        failed = 0
        failures: list[str] = []

        for alert in alerts:
            try:
                if self._paper.store.get(alert.user_id, alert.alert_id) is not None:
                    skipped += 1
                    continue
                if dry_run:
                    created += 1
                    continue
                self._paper.backfill_from_alert(alert)
                created += 1
            except Exception as exc:  # noqa: BLE001 — operational script aggregates failures
                failed += 1
                failures.append(f"{alert.alert_id}: {exc}")

        return PaperTradeBackfillResult(
            alerts_scanned=len(alerts),
            rows_created=created,
            rows_skipped=skipped,
            rows_failed=failed,
            failures=tuple(failures),
        )
