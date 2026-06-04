"""Persistence protocol for paper-trading performance records."""

from __future__ import annotations

from typing import Protocol

from trade_planner.alerts.paper_trade_models import PaperTradePerformanceRecord


class PaperTradePerformanceStore(Protocol):
    def get(self, user_id: str, alert_id: str) -> PaperTradePerformanceRecord | None: ...

    def save(self, user_id: str, record: PaperTradePerformanceRecord) -> None: ...

    def list_for_user(
        self, user_id: str, *, limit: int = 500
    ) -> tuple[PaperTradePerformanceRecord, ...]: ...


class InMemoryPaperTradePerformanceStore:
    def __init__(self) -> None:
        self._records: dict[tuple[str, str], PaperTradePerformanceRecord] = {}

    def get(self, user_id: str, alert_id: str) -> PaperTradePerformanceRecord | None:
        return self._records.get((user_id, alert_id))

    def save(self, user_id: str, record: PaperTradePerformanceRecord) -> None:
        self._records[(user_id, record.alert_id)] = record

    def list_for_user(
        self, user_id: str, *, limit: int = 500
    ) -> tuple[PaperTradePerformanceRecord, ...]:
        rows = [
            record
            for (uid, _), record in self._records.items()
            if uid == user_id
        ]
        rows.sort(key=lambda row: row.created_at, reverse=True)
        return tuple(rows[:limit])
