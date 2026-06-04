"""In-memory persistence for Momentum Breakout alert lifecycle."""

from __future__ import annotations

from typing import Protocol

from trade_planner.alerts.lifecycle_models import (
    ACTIVE_STATUSES,
    AlertLifecycleEvent,
    MomentumBreakoutAlertRecord,
)
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup


class DuplicateActiveMomentumAlertError(ValueError):
    """Raised when an active alert already exists for the symbol."""


class MomentumBreakoutAlertStore(Protocol):
    def get(self, user_id: str, alert_id: str) -> MomentumBreakoutAlertRecord | None: ...

    def save(self, user_id: str, record: MomentumBreakoutAlertRecord) -> None: ...

    def has_active_for_symbol(
        self, user_id: str, symbol: str, setup_name: str
    ) -> bool: ...

    def list_active(self, user_id: str) -> tuple[MomentumBreakoutAlertRecord, ...]: ...

    def list_history(
        self, user_id: str, *, limit: int = 100
    ) -> tuple[MomentumBreakoutAlertRecord, ...]: ...

    def append_event(self, user_id: str, event: AlertLifecycleEvent) -> None: ...

    def list_events(
        self, user_id: str, alert_id: str
    ) -> tuple[AlertLifecycleEvent, ...]: ...

    def list_all_active(self) -> tuple[MomentumBreakoutAlertRecord, ...]: ...


class InMemoryMomentumBreakoutAlertStore:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], MomentumBreakoutAlertRecord] = {}
        self._events: dict[tuple[str, str], list[AlertLifecycleEvent]] = {}
        self._history_ids: dict[str, list[str]] = {}

    def get(self, user_id: str, alert_id: str) -> MomentumBreakoutAlertRecord | None:
        return self._records.get((user_id, alert_id))

    def save(self, user_id: str, record: MomentumBreakoutAlertRecord) -> None:
        key = (user_id, record.alert_id)
        self._records[key] = record
        ids = self._history_ids.setdefault(user_id, [])
        if record.alert_id not in ids:
            ids.append(record.alert_id)

    def has_active_for_symbol(
        self, user_id: str, symbol: str, setup_name: str
    ) -> bool:
        sym = symbol.upper()
        for record in self.list_active(user_id):
            if record.symbol == sym and record.setup_name == setup_name:
                return True
        return False

    def list_active(self, user_id: str) -> tuple[MomentumBreakoutAlertRecord, ...]:
        active = [
            self._records[(user_id, aid)]
            for aid in self._history_ids.get(user_id, [])
            if (user_id, aid) in self._records
            and self._records[(user_id, aid)].status in ACTIVE_STATUSES
        ]
        return tuple(sorted(active, key=lambda r: r.created_at, reverse=True))

    def list_history(
        self, user_id: str, *, limit: int = 100
    ) -> tuple[MomentumBreakoutAlertRecord, ...]:
        ids = self._history_ids.get(user_id, [])
        records = [
            self._records[(user_id, aid)]
            for aid in reversed(ids)
            if (user_id, aid) in self._records
        ]
        return tuple(records[:limit])

    def append_event(self, user_id: str, event: AlertLifecycleEvent) -> None:
        key = (user_id, event.alert_id)
        self._events.setdefault(key, []).append(event)

    def list_events(
        self, user_id: str, alert_id: str
    ) -> tuple[AlertLifecycleEvent, ...]:
        return tuple(self._events.get((user_id, alert_id), ()))

    def list_all_active(self) -> tuple[MomentumBreakoutAlertRecord, ...]:
        active = [
            record
            for record in self._records.values()
            if record.status in ACTIVE_STATUSES
        ]
        return tuple(sorted(active, key=lambda r: r.created_at, reverse=True))

    def clear_user(self, user_id: str) -> None:
        for aid in list(self._history_ids.get(user_id, [])):
            self._records.pop((user_id, aid), None)
            self._events.pop((user_id, aid), None)
        self._history_ids.pop(user_id, None)
