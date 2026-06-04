"""Operational launch readiness for Momentum Breakout alerts."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import oracledb

from trade_planner.alerts.lifecycle_models import AlertLifecycleStatus
from trade_planner.alerts.lifecycle_service import AlertLifecycleService
from trade_planner.alerts.lifecycle_store import MomentumBreakoutAlertStore
from trade_planner.alerts.paper_trade_store import PaperTradePerformanceStore
from app.adapters.strategy.momentum_breakout_store_config import (
    is_production_environment,
    resolve_alert_store_mode,
    resolve_paper_trade_store_mode,
)
from app.core.market_hours import is_us_regular_market_hours
from app.jobs.momentum_breakout_alert_scheduler import (
    _MIN_INTERVAL_SEC,
    is_scheduler_enabled,
    resolve_refresh_interval_sec,
)
from app.services.strategy.momentum_breakout_alert_price_provider import (
    MomentumBreakoutPriceProvider,
)
from app.services.strategy.momentum_breakout_ddl_status import inspect_ddl_status

_PROBE_USER = "__mb_readiness_probe__"


@dataclass(frozen=True, slots=True)
class HealthCheckResult:
    name: str
    ok: bool
    detail: str


@dataclass
class LaunchReadinessReport:
    alert_store_type: str
    paper_trade_store_type: str
    scheduler_enabled: bool
    refresh_interval_sec: int
    raw_refresh_interval_sec: int
    oracle_alert_ddl_required: bool
    oracle_alert_ddl_applied: bool | None
    oracle_paper_ddl_required: bool
    oracle_paper_ddl_applied: bool | None
    ddl_detail_alert: str
    ddl_detail_paper: str
    active_alerts_count: int
    paper_trade_rows_count: int
    total_alerts_count: int
    alerts_missing_paper_rows: int
    latest_lifecycle_update_at: datetime | None
    latest_paper_trade_update_at: datetime | None
    quote_provider_available: bool
    health_checks: tuple[HealthCheckResult, ...] = ()
    warnings: list[str] = field(default_factory=list)
    ready: bool = False


def _raw_refresh_interval_sec() -> int:
    return int(os.getenv("MB_ALERT_REFRESH_INTERVAL_SEC", "180"))


class MomentumBreakoutLaunchReadinessService:
    def __init__(
        self,
        *,
        alert_store: MomentumBreakoutAlertStore,
        paper_store: PaperTradePerformanceStore,
        price_provider: MomentumBreakoutPriceProvider | None = None,
        db_pool: oracledb.ConnectionPool | None = None,
    ) -> None:
        self._alert_store = alert_store
        self._paper_store = paper_store
        self._price_provider = price_provider
        self._db_pool = db_pool

    def evaluate(self) -> LaunchReadinessReport:
        alert_ddl, paper_ddl = inspect_ddl_status(db_pool=self._db_pool)
        alerts = self._alert_store.list_all_alerts()
        active = self._alert_store.list_all_active()
        paper_rows = self._count_paper_rows()
        missing_paper = self._count_alerts_missing_paper(alerts)

        health_checks = (
            self._check_alert_store_writable(),
            self._check_paper_store_writable(),
            self._check_quote_provider(),
            self._check_scheduler_config(),
            self._check_market_hours_module(),
        )

        warnings: list[str] = []
        if is_production_environment():
            if resolve_alert_store_mode() == "memory":
                warnings.append("Alert store is in-memory in a production environment.")
            if resolve_paper_trade_store_mode() == "memory":
                warnings.append("Paper trade store is in-memory in a production environment.")

        if not is_scheduler_enabled():
            warnings.append("Momentum Breakout alert scheduler is disabled.")

        if self._price_provider is None:
            warnings.append("No quote provider configured for alert price refresh.")
        elif not health_checks[2].ok:
            warnings.append("Quote provider could not fetch a price for SPY.")

        if len(alerts) > 0 and paper_rows == 0:
            warnings.append(
                "Paper trade table is empty while alerts exist — run backfill script."
            )

        if missing_paper > 0:
            warnings.append(
                f"{missing_paper} alert(s) have no paper performance row — run backfill script."
            )

        raw_interval = _raw_refresh_interval_sec()
        if raw_interval < _MIN_INTERVAL_SEC:
            warnings.append(
                f"MB_ALERT_REFRESH_INTERVAL_SEC={raw_interval} is below minimum "
                f"{_MIN_INTERVAL_SEC}s; clamped at runtime."
            )

        if alert_ddl.required and alert_ddl.applied is False:
            warnings.append(f"Alert DDL not applied: {alert_ddl.detail}")
        if paper_ddl.required and paper_ddl.applied is False:
            warnings.append(f"Paper trade DDL not applied: {paper_ddl.detail}")

        blocking = [
            w
            for w in warnings
            if "in-memory in a production" in w
            or "DDL not applied" in w
            or "No quote provider" in w
        ]
        checks_ok = all(check.ok for check in health_checks)
        ready = checks_ok and not blocking

        return LaunchReadinessReport(
            alert_store_type=resolve_alert_store_mode(),
            paper_trade_store_type=resolve_paper_trade_store_mode(),
            scheduler_enabled=is_scheduler_enabled(),
            refresh_interval_sec=resolve_refresh_interval_sec(),
            raw_refresh_interval_sec=raw_interval,
            oracle_alert_ddl_required=alert_ddl.required,
            oracle_alert_ddl_applied=alert_ddl.applied,
            oracle_paper_ddl_required=paper_ddl.required,
            oracle_paper_ddl_applied=paper_ddl.applied,
            ddl_detail_alert=alert_ddl.detail,
            ddl_detail_paper=paper_ddl.detail,
            active_alerts_count=len(active),
            paper_trade_rows_count=paper_rows,
            total_alerts_count=len(alerts),
            alerts_missing_paper_rows=missing_paper,
            latest_lifecycle_update_at=self._latest_alert_timestamp(alerts),
            latest_paper_trade_update_at=self._latest_paper_timestamp(),
            quote_provider_available=self._price_provider is not None
            and health_checks[2].ok,
            health_checks=health_checks,
            warnings=warnings,
            ready=ready,
        )

    def _count_paper_rows(self) -> int:
        if hasattr(self._paper_store, "count_all"):
            return int(self._paper_store.count_all())  # type: ignore[attr-defined]
        return len(self._paper_store.list_all())

    def _count_alerts_missing_paper(
        self, alerts: tuple[Any, ...]
    ) -> int:
        missing = 0
        for alert in alerts:
            if alert.status in {AlertLifecycleStatus.CANCELLED}:
                continue
            if self._paper_store.get(alert.user_id, alert.alert_id) is None:
                missing += 1
        return missing

    def _latest_alert_timestamp(self, alerts: tuple[Any, ...]) -> datetime | None:
        if hasattr(self._alert_store, "latest_updated_at"):
            value = self._alert_store.latest_updated_at()  # type: ignore[attr-defined]
            return self._parse_ts(value)
        stamps: list[datetime] = []
        for alert in alerts:
            for candidate in (alert.exit_at, alert.triggered_at, alert.created_at):
                if candidate is not None:
                    stamps.append(self._aware(candidate))
        return max(stamps) if stamps else None

    def _latest_paper_timestamp(self) -> datetime | None:
        if hasattr(self._paper_store, "latest_updated_at"):
            value = self._paper_store.latest_updated_at()  # type: ignore[attr-defined]
            return self._parse_ts(value)
        rows = self._paper_store.list_all(limit=1)
        if not rows:
            return None
        return self._aware(rows[0].created_at)

    def _check_alert_store_writable(self) -> HealthCheckResult:
        try:
            record = AlertLifecycleService.build_record(
                user_id=_PROBE_USER,
                symbol="SPY",
                signal_date=datetime.now(timezone.utc).date(),
                entry_price=1.0,
                stop_price=0.5,
                target_price=2.0,
                entry_is_stop=True,
            )
            record = record.with_status(AlertLifecycleStatus.CANCELLED)
            self._alert_store.save(_PROBE_USER, record)
            loaded = self._alert_store.get(_PROBE_USER, record.alert_id)
            ok = loaded is not None and loaded.symbol == "SPY"
            return HealthCheckResult(
                name="alert_store_writable",
                ok=ok,
                detail="Alert store save/read probe succeeded"
                if ok
                else "Alert store probe read failed",
            )
        except Exception as exc:  # noqa: BLE001
            return HealthCheckResult(
                name="alert_store_writable",
                ok=False,
                detail=f"Alert store probe failed: {exc}",
            )

    def _check_paper_store_writable(self) -> HealthCheckResult:
        try:
            from app.services.strategy.paper_trade_performance_service import (
                PaperTradePerformanceService,
            )

            svc = PaperTradePerformanceService(self._paper_store)
            record = AlertLifecycleService.build_record(
                user_id=_PROBE_USER,
                symbol="SPY",
                signal_date=datetime.now(timezone.utc).date(),
                entry_price=1.0,
                stop_price=0.5,
                target_price=2.0,
                entry_is_stop=True,
            )
            record = record.with_status(AlertLifecycleStatus.PENDING_ENTRY)
            svc.backfill_from_alert(record)
            loaded = self._paper_store.get(_PROBE_USER, record.alert_id)
            ok = loaded is not None
            return HealthCheckResult(
                name="paper_trade_store_writable",
                ok=ok,
                detail="Paper store save/read probe succeeded"
                if ok
                else "Paper store probe read failed",
            )
        except Exception as exc:  # noqa: BLE001
            return HealthCheckResult(
                name="paper_trade_store_writable",
                ok=False,
                detail=f"Paper store probe failed: {exc}",
            )

    def _check_quote_provider(self) -> HealthCheckResult:
        if self._price_provider is None:
            return HealthCheckResult(
                name="quote_fetch_spy",
                ok=False,
                detail="Quote provider not configured",
            )
        try:
            price = self._price_provider.get_latest_price("SPY")
            ok = price is not None and price > 0
            return HealthCheckResult(
                name="quote_fetch_spy",
                ok=ok,
                detail=f"SPY quote={price}" if ok else "SPY quote unavailable",
            )
        except Exception as exc:  # noqa: BLE001
            return HealthCheckResult(
                name="quote_fetch_spy",
                ok=False,
                detail=f"Quote fetch failed: {exc}",
            )

    @staticmethod
    def _check_scheduler_config() -> HealthCheckResult:
        interval = resolve_refresh_interval_sec()
        enabled = is_scheduler_enabled()
        ok = enabled and interval >= _MIN_INTERVAL_SEC
        return HealthCheckResult(
            name="scheduler_configured",
            ok=ok,
            detail=(
                f"enabled={enabled}, effective_interval_sec={interval}"
            ),
        )

    @staticmethod
    def _check_market_hours_module() -> HealthCheckResult:
        try:
            open_session = is_us_regular_market_hours(
                datetime(2024, 6, 3, 15, 0, tzinfo=timezone.utc)
            )
            closed = is_us_regular_market_hours(
                datetime(2024, 6, 3, 21, 0, tzinfo=timezone.utc)
            )
            ok = open_session and not closed
            return HealthCheckResult(
                name="market_hours_module",
                ok=ok,
                detail="Market hours checks behaved as expected"
                if ok
                else "Market hours module returned unexpected values",
            )
        except Exception as exc:  # noqa: BLE001
            return HealthCheckResult(
                name="market_hours_module",
                ok=False,
                detail=f"Market hours check failed: {exc}",
            )

    @staticmethod
    def _aware(ts: datetime) -> datetime:
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts

    @staticmethod
    def _parse_ts(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return MomentumBreakoutLaunchReadinessService._aware(value)
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
