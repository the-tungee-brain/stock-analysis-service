"""Refresh active Momentum Breakout alerts from latest market prices."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from trade_planner.alerts.lifecycle_models import AlertLifecycleStatus
from trade_planner.alerts.lifecycle_service import AlertLifecycleService

from app.core.market_hours import is_us_regular_market_hours
from app.core.momentum_breakout_monitoring import log_mb_event
from app.services.strategy.momentum_breakout_ops_metrics import (
    get_momentum_breakout_ops_metrics,
)
from app.services.strategy.momentum_breakout_alert_price_provider import (
    MomentumBreakoutPriceProvider,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AlertStatusChange:
    alert_id: str
    symbol: str
    prior_status: str
    new_status: str


@dataclass(frozen=True, slots=True)
class MomentumBreakoutRefreshResult:
    processed: int
    updated: int
    skipped_market_hours: bool
    warnings: tuple[str, ...]
    changes: tuple[AlertStatusChange, ...]


class MomentumBreakoutAlertRefreshService:
    def __init__(
        self,
        *,
        lifecycle_service: AlertLifecycleService,
        price_provider: MomentumBreakoutPriceProvider,
    ) -> None:
        self._lifecycle = lifecycle_service
        self._price_provider = price_provider

    def refresh_all_active_alerts(
        self,
        *,
        force: bool = False,
    ) -> MomentumBreakoutRefreshResult:
        if not force and not is_us_regular_market_hours():
            return MomentumBreakoutRefreshResult(
                processed=0,
                updated=0,
                skipped_market_hours=True,
                warnings=(),
                changes=(),
            )
        result = self._refresh_records(
            self._lifecycle.store.list_all_active(),
            force=force,
        )
        log_mb_event(
            "scheduler_refresh_completed",
            processed=result.processed,
            updated=result.updated,
            changes=len(result.changes),
            skipped_market_hours=result.skipped_market_hours,
        )
        get_momentum_breakout_ops_metrics().record_scheduler_success()
        return result

    def refresh_user_active_alerts(
        self,
        user_id: str,
        *,
        force: bool = True,
    ) -> MomentumBreakoutRefreshResult:
        if not force and not is_us_regular_market_hours():
            return MomentumBreakoutRefreshResult(
                processed=0,
                updated=0,
                skipped_market_hours=True,
                warnings=(),
                changes=(),
            )
        return self._refresh_records(
            self._lifecycle.list_active_alerts(user_id),
            force=force,
        )

    def _refresh_records(
        self,
        records: tuple,
        *,
        force: bool,
    ) -> MomentumBreakoutRefreshResult:
        _ = force
        warnings: list[str] = []
        changes: list[AlertStatusChange] = []
        updated = 0
        now = datetime.now(timezone.utc)

        for record in records:
            prior_status = record.status
            price = self._price_provider.get_latest_price(record.symbol)
            if price is None:
                message = (
                    f"Price fetch failed for {record.symbol} "
                    f"(alert {record.alert_id}); status unchanged."
                )
                warnings.append(message)
                logger.warning(message)
                continue

            try:
                refreshed = self._lifecycle.update_with_latest_price(
                    record.user_id,
                    record.alert_id,
                    symbol=record.symbol,
                    price=price,
                    timestamp=now,
                )
            except Exception as exc:
                message = (
                    f"Lifecycle update failed for alert {record.alert_id}: {exc}"
                )
                warnings.append(message)
                logger.warning(message, exc_info=True)
                continue

            if refreshed.status != prior_status:
                updated += 1
                changes.append(
                    AlertStatusChange(
                        alert_id=refreshed.alert_id,
                        symbol=refreshed.symbol,
                        prior_status=prior_status.value,
                        new_status=refreshed.status.value,
                    )
                )

        return MomentumBreakoutRefreshResult(
            processed=len(records),
            updated=updated,
            skipped_market_hours=False,
            warnings=tuple(warnings),
            changes=tuple(changes),
        )
