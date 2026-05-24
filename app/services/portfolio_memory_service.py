from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timezone

from app.adapters.portfolio.alert_history_adapter import AlertHistoryAdapter
from app.adapters.portfolio.portfolio_snapshot_adapter import PortfolioSnapshotAdapter
from app.core.prompts import AnalysisAction, _position_pnl, _position_pnl_pct
from app.models.intelligence_models import PortfolioIntelligence, ProactiveAlert
from app.models.portfolio_memory_models import (
    AttentionItem,
    MorningBrief,
    PortfolioChanges,
    PortfolioSnapshotRecord,
    PortfolioSnapshotSummary,
    PositionWeightChange,
    SnapshotPosition,
)
from app.models.schwab_models import Position, SchwabAccounts

logger = logging.getLogger(__name__)

WEIGHT_CHANGE_THRESHOLD_PCT = 2.0


class PortfolioMemoryService:
    def __init__(
        self,
        portfolio_snapshot_adapter: PortfolioSnapshotAdapter,
        alert_history_adapter: AlertHistoryAdapter,
    ):
        self.portfolio_snapshot_adapter = portfolio_snapshot_adapter
        self.alert_history_adapter = alert_history_adapter

    def capture_snapshot(
        self,
        *,
        user_id: str,
        account: SchwabAccounts,
        positions: list[Position],
        portfolio_brief: PortfolioIntelligence | None = None,
    ) -> PortfolioSnapshotRecord | None:
        try:
            snapshot_date = date.today()
            balances = account.securitiesAccount.currentBalances
            account_number = account.securitiesAccount.accountNumber
            compact_positions = self._compact_positions(
                positions=positions,
                liquidation_value=balances.liquidationValue,
            )
            summary = self._build_summary(
                portfolio_brief=portfolio_brief,
                position_count=len(compact_positions),
            )

            record = PortfolioSnapshotRecord(
                user_id=user_id,
                snapshot_date=snapshot_date,
                account_number=account_number,
                liquidation_value=balances.liquidationValue,
                cash_balance=balances.cashBalance,
                positions=compact_positions,
                summary=summary,
            )
            return self.portfolio_snapshot_adapter.upsert(record)
        except Exception:
            logger.exception("Failed to capture portfolio snapshot for user %s", user_id)
            return None

    def record_alerts(
        self,
        *,
        user_id: str,
        alerts: list[ProactiveAlert],
    ) -> None:
        try:
            active_fingerprints: set[str] = set()
            for alert in alerts:
                fingerprint = self._alert_fingerprint(alert)
                active_fingerprints.add(fingerprint)
                self.alert_history_adapter.upsert_active(
                    user_id=user_id,
                    fingerprint=fingerprint,
                    action=alert.action.value,
                    symbol=alert.symbol,
                    reason=alert.reason,
                    priority=alert.priority,
                )
            self.alert_history_adapter.resolve_missing(user_id, active_fingerprints)
        except Exception:
            logger.exception("Failed to record alert history for user %s", user_id)

    def get_portfolio_changes(
        self,
        *,
        user_id: str,
        compare_days: int = 1,
    ) -> PortfolioChanges:
        snapshots = self.portfolio_snapshot_adapter.list_recent(
            user_id, limit=max(compare_days + 1, 2)
        )
        if len(snapshots) < 2:
            return PortfolioChanges(
                summary="Not enough history yet — check back after your next daily snapshot.",
            )

        current = snapshots[0]
        previous = snapshots[min(compare_days, len(snapshots) - 1)]
        return self._diff_snapshots(previous=previous, current=current)

    def build_attention_queue(
        self,
        *,
        user_id: str,
        current_alerts: list[ProactiveAlert],
    ) -> list[AttentionItem]:
        items: list[AttentionItem] = []
        seen: set[tuple[str, str | None]] = set()

        for alert in sorted(current_alerts, key=lambda item: item.priority):
            key = (alert.action.value, alert.symbol)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                AttentionItem(
                    action=alert.action,
                    label=alert.label,
                    symbol=alert.symbol,
                    reason=alert.reason,
                    priority=alert.priority,
                    source="current",
                )
            )

        try:
            historical = self.alert_history_adapter.list_active(user_id)
        except Exception:
            historical = []

        for item in historical:
            key = (item.action.value, item.symbol)
            if key in seen:
                continue
            seen.add(key)
            items.append(
                AttentionItem(
                    action=item.action,
                    label=item.label,
                    symbol=item.symbol,
                    reason=item.reason,
                    priority=item.priority + 10,
                    source="historical",
                    first_seen_at=item.first_seen_at,
                    days_active=item.days_active,
                    alert_id=item.id,
                )
            )

        items.sort(key=lambda entry: entry.priority)
        return items

    def build_morning_brief(
        self,
        *,
        user_id: str,
        portfolio_brief: PortfolioIntelligence,
        current_alerts: list[ProactiveAlert],
    ) -> MorningBrief:
        changes = self.get_portfolio_changes(user_id=user_id, compare_days=1)
        attention_queue = self.build_attention_queue(
            user_id=user_id,
            current_alerts=current_alerts,
        )
        top_alerts = sorted(current_alerts, key=lambda alert: alert.priority)[:5]

        return MorningBrief(
            generated_at=datetime.now(timezone.utc),
            macro_regime=(
                portfolio_brief.digest.macro_regime if portfolio_brief.digest else None
            ),
            digest=portfolio_brief.digest,
            changes=changes,
            signals=portfolio_brief.signals,
            top_alerts=top_alerts,
            attention_queue=attention_queue[:10],
        )

    def list_alert_history(
        self,
        *,
        user_id: str,
        days: int = 30,
    ) -> list:
        try:
            return self.alert_history_adapter.list_recent(user_id, days=days)
        except Exception:
            logger.exception("Failed to load alert history for user %s", user_id)
            return []

    def dismiss_alert(self, *, user_id: str, alert_id: str) -> bool:
        try:
            return self.alert_history_adapter.dismiss(user_id, alert_id)
        except Exception:
            logger.exception("Failed to dismiss alert %s for user %s", alert_id, user_id)
            return False

    @staticmethod
    def _alert_fingerprint(alert: ProactiveAlert) -> str:
        raw = f"{alert.action.value}:{alert.symbol or 'portfolio'}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @staticmethod
    def _position_key(position: SnapshotPosition) -> str:
        parts = [position.symbol.upper(), position.asset_type]
        if position.strike is not None:
            parts.append(f"{position.strike:g}")
        if position.put_call:
            parts.append(position.put_call)
        if position.expiration:
            parts.append(position.expiration[:10])
        return "|".join(parts)

    @staticmethod
    def _compact_positions(
        *,
        positions: list[Position],
        liquidation_value: float,
    ) -> list[SnapshotPosition]:
        if liquidation_value <= 0:
            return []

        compact: list[SnapshotPosition] = []
        for position in positions:
            instrument = position.instrument
            if instrument.assetType == "OPTION":
                symbol = instrument.underlyingSymbol or instrument.symbol
            else:
                symbol = instrument.symbol

            quantity = position.longQuantity - position.shortQuantity
            pnl = _position_pnl(position)
            pnl_pct = _position_pnl_pct(position)
            weight_pct = (abs(position.marketValue) / liquidation_value) * 100.0

            compact.append(
                SnapshotPosition(
                    symbol=symbol.upper(),
                    asset_type=instrument.assetType,
                    quantity=quantity,
                    market_value=position.marketValue,
                    weight_pct=round(weight_pct, 2),
                    pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 2) if pnl_pct is not None else None,
                    option_strategy=position.optionStrategy,
                    strike=instrument.strikePrice,
                    expiration=instrument.expirationDate,
                    put_call=instrument.putCall,
                )
            )

        compact.sort(key=lambda item: abs(item.market_value), reverse=True)
        return compact

    @staticmethod
    def _build_summary(
        *,
        portfolio_brief: PortfolioIntelligence | None,
        position_count: int,
    ) -> PortfolioSnapshotSummary:
        if portfolio_brief is None:
            return PortfolioSnapshotSummary(position_count=position_count)

        sector_weights: dict[str, float] = {}
        if portfolio_brief.digest and portfolio_brief.digest.sector_weights:
            sector_weights = {
                item.sector: round(item.weight_pct, 2)
                for item in portfolio_brief.digest.sector_weights
            }

        return PortfolioSnapshotSummary(
            alert_count=len(portfolio_brief.alerts),
            signal_count=len(portfolio_brief.signals),
            position_count=position_count,
            sector_weights=sector_weights,
        )

    def _diff_snapshots(
        self,
        *,
        previous: PortfolioSnapshotRecord,
        current: PortfolioSnapshotRecord,
    ) -> PortfolioChanges:
        previous_by_symbol = self._aggregate_weights(previous.positions)
        current_by_symbol = self._aggregate_weights(current.positions)

        previous_symbols = set(previous_by_symbol)
        current_symbols = set(current_by_symbol)

        new_symbols = sorted(current_symbols - previous_symbols)
        removed_symbols = sorted(previous_symbols - current_symbols)

        weight_changes: list[PositionWeightChange] = []
        for symbol in sorted(previous_symbols & current_symbols):
            prev_weight = previous_by_symbol[symbol]
            curr_weight = current_by_symbol[symbol]
            change = curr_weight - prev_weight
            if abs(change) >= WEIGHT_CHANGE_THRESHOLD_PCT:
                weight_changes.append(
                    PositionWeightChange(
                        symbol=symbol,
                        previous_weight_pct=round(prev_weight, 2),
                        current_weight_pct=round(curr_weight, 2),
                        change_pct=round(change, 2),
                    )
                )

        weight_changes.sort(key=lambda item: abs(item.change_pct), reverse=True)

        liquidation_change = None
        liquidation_change_pct = None
        if (
            previous.liquidation_value is not None
            and current.liquidation_value is not None
            and previous.liquidation_value > 0
        ):
            liquidation_change = round(
                current.liquidation_value - previous.liquidation_value, 2
            )
            liquidation_change_pct = round(
                (liquidation_change / previous.liquidation_value) * 100.0, 2
            )

        summary = self._build_change_summary(
            from_date=previous.snapshot_date,
            to_date=current.snapshot_date,
            liquidation_change_pct=liquidation_change_pct,
            new_symbols=new_symbols,
            removed_symbols=removed_symbols,
            weight_changes=weight_changes,
        )

        return PortfolioChanges(
            from_date=previous.snapshot_date,
            to_date=current.snapshot_date,
            liquidation_value_change=liquidation_change,
            liquidation_value_change_pct=liquidation_change_pct,
            new_symbols=new_symbols,
            removed_symbols=removed_symbols,
            weight_changes=weight_changes,
            summary=summary,
        )

    @staticmethod
    def _aggregate_weights(positions: list[SnapshotPosition]) -> dict[str, float]:
        by_symbol: dict[str, float] = {}
        for position in positions:
            by_symbol[position.symbol] = by_symbol.get(position.symbol, 0.0) + abs(
                position.weight_pct
            )
        return by_symbol

    @staticmethod
    def _build_change_summary(
        *,
        from_date: date,
        to_date: date,
        liquidation_change_pct: float | None,
        new_symbols: list[str],
        removed_symbols: list[str],
        weight_changes: list[PositionWeightChange],
    ) -> str:
        parts: list[str] = [f"Changes from {from_date.isoformat()} to {to_date.isoformat()}:"]

        if liquidation_change_pct is not None:
            parts.append(f"portfolio value {liquidation_change_pct:+.2f}%")

        if new_symbols:
            parts.append(f"added {', '.join(new_symbols)}")

        if removed_symbols:
            parts.append(f"removed {', '.join(removed_symbols)}")

        if weight_changes:
            top = weight_changes[0]
            parts.append(
                f"largest weight shift: {top.symbol} "
                f"{top.previous_weight_pct:.1f}% → {top.current_weight_pct:.1f}%"
            )

        if len(parts) == 1:
            parts.append("no material position changes detected")

        return "; ".join(parts)
