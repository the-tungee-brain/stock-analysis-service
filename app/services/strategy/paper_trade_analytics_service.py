"""Aggregate analytics from live paper-trading performance records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from trade_planner.alerts.lifecycle_models import AlertLifecycleStatus
from trade_planner.alerts.paper_trade_models import PaperTradePerformanceRecord
from trade_planner.alerts.paper_trade_store import PaperTradePerformanceStore
from app.services.strategy.paper_trade_performance_service import (
    PaperTradePerformanceService,
    compute_holding_days,
)


CLOSED_WIN = AlertLifecycleStatus.TARGET_HIT.value
CLOSED_LOSS = AlertLifecycleStatus.STOP_HIT.value
OPEN_STATUSES = frozenset(
    {
        AlertLifecycleStatus.ENTRY_TRIGGERED.value,
        AlertLifecycleStatus.OPEN.value,
    }
)
TERMINAL_CLOSED = frozenset({CLOSED_WIN, CLOSED_LOSS})


@dataclass(frozen=True, slots=True)
class PaperTradeSummaryMetrics:
    total_alerts: int
    triggered_alerts: int
    expired_alerts: int
    win_rate: float | None
    average_win: float | None
    average_loss: float | None
    expectancy: float | None
    profit_factor: float | None
    average_holding_days: float | None
    max_drawdown: float | None
    current_open_trades: int


@dataclass(frozen=True, slots=True)
class PaperTradeBucketMetrics:
    key: str
    trade_count: int
    win_rate: float | None
    expectancy: float | None
    profit_factor: float | None
    average_return_pct: float | None


def _is_triggered(record: PaperTradePerformanceRecord) -> bool:
    if record.entry_triggered_at is not None:
        return True
    return record.status in {
        AlertLifecycleStatus.OPEN.value,
        CLOSED_WIN,
        CLOSED_LOSS,
    }


def _closed_returns(records: tuple[PaperTradePerformanceRecord, ...]) -> list[float]:
    returns: list[float] = []
    for record in records:
        if record.status not in TERMINAL_CLOSED:
            continue
        if record.outcome_return_pct is None:
            continue
        returns.append(record.outcome_return_pct)
    return returns


def _win_loss_stats(
    records: tuple[PaperTradePerformanceRecord, ...],
) -> tuple[list[float], list[float]]:
    wins: list[float] = []
    losses: list[float] = []
    for record in records:
        if record.outcome_return_pct is None:
            continue
        if record.status == CLOSED_WIN:
            wins.append(record.outcome_return_pct)
        elif record.status == CLOSED_LOSS:
            losses.append(record.outcome_return_pct)
    return wins, losses


def _profit_factor(wins: list[float], losses: list[float]) -> float | None:
    win_sum = sum(wins)
    loss_sum = abs(sum(losses))
    if loss_sum <= 0:
        return None if win_sum <= 0 else float("inf")
    return win_sum / loss_sum


def _expectancy(win_rate: float, avg_win: float, avg_loss: float) -> float:
    return win_rate * avg_win + (1.0 - win_rate) * avg_loss


def _max_drawdown(records: tuple[PaperTradePerformanceRecord, ...]) -> float | None:
    closed = [
        record
        for record in records
        if record.status in TERMINAL_CLOSED and record.outcome_return_pct is not None
    ]
    if not closed:
        return None
    closed.sort(
        key=lambda row: row.exit_at or row.created_at,
    )
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for record in closed:
        equity *= 1.0 + record.outcome_return_pct  # type: ignore[operator]
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
    return round(max_dd, 6)


def _average_holding_days(records: tuple[PaperTradePerformanceRecord, ...]) -> float | None:
    days: list[int] = []
    now = datetime.now(timezone.utc)
    for record in records:
        if record.holding_days is not None:
            days.append(record.holding_days)
            continue
        computed = compute_holding_days(
            record.entry_triggered_at,
            record.exit_at,
            as_of=now,
        )
        if computed is not None:
            days.append(computed)
    if not days:
        return None
    return sum(days) / len(days)


def build_summary(records: tuple[PaperTradePerformanceRecord, ...]) -> PaperTradeSummaryMetrics:
    total = len(records)
    triggered = sum(1 for record in records if _is_triggered(record))
    expired = sum(
        1
        for record in records
        if record.status == AlertLifecycleStatus.EXPIRED.value
    )
    open_count = sum(1 for record in records if record.status in OPEN_STATUSES)
    wins, losses = _win_loss_stats(records)
    closed_count = len(wins) + len(losses)
    win_rate = len(wins) / closed_count if closed_count else None
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    expectancy = None
    if win_rate is not None and avg_win is not None:
        loss_component = avg_loss if avg_loss is not None else 0.0
        expectancy = _expectancy(win_rate, avg_win, loss_component)
    pf = _profit_factor(wins, losses)
    return PaperTradeSummaryMetrics(
        total_alerts=total,
        triggered_alerts=triggered,
        expired_alerts=expired,
        win_rate=round(win_rate, 6) if win_rate is not None else None,
        average_win=round(avg_win, 6) if avg_win is not None else None,
        average_loss=round(avg_loss, 6) if avg_loss is not None else None,
        expectancy=round(expectancy, 6) if expectancy is not None else None,
        profit_factor=round(pf, 6) if pf is not None and pf != float("inf") else pf,
        average_holding_days=round(_average_holding_days(records), 4)
        if _average_holding_days(records) is not None
        else None,
        max_drawdown=_max_drawdown(records),
        current_open_trades=open_count,
    )


def build_bucket_metrics(
    records: tuple[PaperTradePerformanceRecord, ...],
    *,
    key_fn,
) -> tuple[PaperTradeBucketMetrics, ...]:
    groups: dict[str, list[PaperTradePerformanceRecord]] = {}
    for record in records:
        key = key_fn(record) or "UNKNOWN"
        groups.setdefault(key, []).append(record)
    buckets: list[PaperTradeBucketMetrics] = []
    for key in sorted(groups):
        group = tuple(groups[key])
        wins, losses = _win_loss_stats(group)
        closed = len(wins) + len(losses)
        win_rate = len(wins) / closed if closed else None
        avg_win = sum(wins) / len(wins) if wins else None
        avg_loss = sum(losses) / len(losses) if losses else None
        expectancy = None
        if win_rate is not None and avg_win is not None:
            loss_component = avg_loss if avg_loss is not None else 0.0
            expectancy = _expectancy(win_rate, avg_win, loss_component)
        pf = _profit_factor(wins, losses)
        returns = _closed_returns(group)
        avg_ret = sum(returns) / len(returns) if returns else None
        buckets.append(
            PaperTradeBucketMetrics(
                key=key,
                trade_count=len(group),
                win_rate=round(win_rate, 6) if win_rate is not None else None,
                expectancy=round(expectancy, 6) if expectancy is not None else None,
                profit_factor=round(pf, 6)
                if pf is not None and pf != float("inf")
                else pf,
                average_return_pct=round(avg_ret, 6) if avg_ret is not None else None,
            )
        )
    return tuple(buckets)


class PaperTradeAnalyticsService:
    def __init__(
        self,
        performance_service: PaperTradePerformanceService | None = None,
        store: PaperTradePerformanceStore | None = None,
    ) -> None:
        self._performance = performance_service or PaperTradePerformanceService(
            store=store
        )

    def summary(self, user_id: str) -> PaperTradeSummaryMetrics:
        records = self._performance.list_records(user_id)
        return build_summary(records)

    def list_trades(
        self, user_id: str, *, limit: int = 100
    ) -> tuple[PaperTradePerformanceRecord, ...]:
        return self._performance.list_records(user_id, limit=limit)

    def by_symbol(self, user_id: str) -> tuple[PaperTradeBucketMetrics, ...]:
        records = self._performance.list_records(user_id)
        return build_bucket_metrics(records, key_fn=lambda row: row.symbol)

    def by_regime(self, user_id: str) -> tuple[PaperTradeBucketMetrics, ...]:
        records = self._performance.list_records(user_id)
        return build_bucket_metrics(records, key_fn=lambda row: row.market_regime)

    def by_risk_gate_action(self, user_id: str) -> tuple[PaperTradeBucketMetrics, ...]:
        records = self._performance.list_records(user_id)
        return build_bucket_metrics(
            records,
            key_fn=lambda row: row.risk_gate_action or "NONE",
        )
