"""Production risk gating for Momentum Breakout educational alerts."""

from __future__ import annotations

from trade_planner.alerts.risk_models import (
    AlertDecision,
    AlertGateAction,
    AlertPriority,
    AlertRiskContext,
    AlertRiskSettings,
    ClosedTradeSnapshot,
    OpenTradeSnapshot,
)
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup

MEGA_CAP_TECH_SYMBOLS: frozenset[str] = frozenset(
    {
        "AAPL",
        "MSFT",
        "NVDA",
        "META",
        "AMZN",
        "GOOG",
        "GOOGL",
        "TSLA",
    }
)

VOLUME_CLIMAX_MESSAGE = (
    "High-volume climax day historically showed weaker expectancy."
)

EDUCATIONAL_DISCLAIMER = (
    "Educational trade plan alert only — not investment advice. No orders are placed."
)


class AlertRiskGate:
    """Apply portfolio risk controls before user-facing Momentum Breakout alerts."""

    SETUP_NAME = MomentumBreakoutSetup.name

    def evaluate(self, context: AlertRiskContext) -> AlertDecision:
        plan = context.candidate_plan
        settings = context.settings
        symbol = (context.current_symbol or plan.symbol).upper()

        reasons: list[str] = [EDUCATIONAL_DISCLAIMER]
        block = False
        warn = False
        size_down = False
        priority = AlertPriority.HIGH

        momentum_open = self._momentum_open(context.open_trades)
        momentum_closed = self._momentum_closed(context.recent_closed)

        # 4. Consecutive-loss circuit breaker
        if self._consecutive_losses_triggered(momentum_closed, settings):
            block = True
            reasons.append(
                f"Circuit breaker: last {settings.consecutive_loss_limit} closed "
                f"{self.SETUP_NAME} trades were losses."
            )

        # 5. Rolling drawdown circuit breaker
        if self._rolling_drawdown_triggered(momentum_closed, settings):
            block = True
            reasons.append(
                f"Circuit breaker: last {settings.rolling_window_trades} closed "
                f"{self.SETUP_NAME} trades cumulatively <= "
                f"{settings.rolling_drawdown_limit_pct * 100:.0f}%."
            )

        # 1. Max open positions
        if len(momentum_open) >= settings.max_open_positions:
            block = True
            reasons.append(
                f"Max open positions reached ({len(momentum_open)}/"
                f"{settings.max_open_positions} active {self.SETUP_NAME} trades)."
            )

        # 2. One active trade per symbol
        if any(t.symbol.upper() == symbol for t in momentum_open):
            block = True
            reasons.append(
                f"Active {self.SETUP_NAME} trade already open for {symbol}."
            )

        trade_risk_pct = self._trade_risk_pct(plan)
        total_open_risk = self._total_open_risk(momentum_open, settings)

        # 3. Portfolio risk cap
        projected_total = total_open_risk + trade_risk_pct
        if projected_total > settings.max_total_open_risk_pct:
            block = True
            reasons.append(
                f"Portfolio risk cap: projected open risk {projected_total * 100:.2f}% "
                f"exceeds max {settings.max_total_open_risk_pct * 100:.1f}%."
            )
        elif projected_total > settings.max_total_open_risk_pct * 0.75:
            size_down = True
            reasons.append(
                f"Portfolio risk elevated ({projected_total * 100:.2f}% of "
                f"{settings.max_total_open_risk_pct * 100:.1f}% cap)."
            )

        per_trade_cap = min(settings.max_risk_per_trade_pct, trade_risk_pct)
        if trade_risk_pct > settings.max_risk_per_trade_pct:
            size_down = True
            per_trade_cap = settings.max_risk_per_trade_pct
            reasons.append(
                f"Stop distance implies {trade_risk_pct * 100:.2f}% risk; "
                f"capped at {settings.max_risk_per_trade_pct * 100:.1f}% per trade."
            )

        remaining_budget = max(
            0.0, settings.max_total_open_risk_pct - total_open_risk
        )
        recommended_risk = min(per_trade_cap, remaining_budget)
        if recommended_risk < per_trade_cap and not block:
            size_down = True
            reasons.append(
                f"Recommended risk reduced to {recommended_risk * 100:.2f}% "
                f"to stay within open risk budget."
            )

        # 6. Mega-cap tech correlation throttle
        mega_active = self._mega_cap_active_count(momentum_open)
        in_mega = symbol in MEGA_CAP_TECH_SYMBOLS or (
            context.sector_or_group or ""
        ).upper() in ("MEGA_CAP_TECH", "TECH_MEGA", "MEGA-CAP-TECH")
        if (
            in_mega
            and mega_active >= settings.mega_cap_correlation_threshold
        ):
            block = True
            priority = AlertPriority.LOW
            reasons.append(
                f"Mega-cap tech correlation throttle: {mega_active} active "
                f"highly correlated positions (limit "
                f"{settings.mega_cap_correlation_threshold})."
            )

        # 7. Volume climax warning (no hard block)
        if (
            context.volume_ratio is not None
            and context.volume_ratio >= settings.volume_climax_ratio
        ):
            warn = True
            priority = AlertPriority.MEDIUM if priority == AlertPriority.HIGH else priority
            reasons.append(VOLUME_CLIMAX_MESSAGE)

        if context.market_regime is not None:
            reasons.append(f"Market regime at signal: {context.market_regime.value}.")

        action = self._resolve_action(block=block, warn=warn, size_down=size_down)
        allowed = action != AlertGateAction.BLOCK

        if action == AlertGateAction.SIZE_DOWN:
            priority = AlertPriority.MEDIUM if priority == AlertPriority.HIGH else priority
        if action == AlertGateAction.WARN and priority == AlertPriority.HIGH:
            priority = AlertPriority.MEDIUM
        if not allowed:
            priority = AlertPriority.LOW
            recommended_risk = 0.0

        max_notional = self._max_shares_or_dollars(
            plan,
            account_equity_usd=context.account_equity_usd,
            risk_pct=recommended_risk if allowed else 0.0,
        )

        return AlertDecision(
            allowed=allowed,
            action=action,
            reasons=tuple(reasons),
            recommended_position_risk_pct=round(recommended_risk, 6),
            max_shares_or_dollars=max_notional,
            alert_priority=priority,
        )

    @staticmethod
    def _resolve_action(
        *,
        block: bool,
        warn: bool,
        size_down: bool,
    ) -> AlertGateAction:
        if block:
            return AlertGateAction.BLOCK
        if size_down:
            return AlertGateAction.SIZE_DOWN
        if warn:
            return AlertGateAction.WARN
        return AlertGateAction.ALLOW

    def _momentum_open(
        self, trades: tuple[OpenTradeSnapshot, ...]
    ) -> tuple[OpenTradeSnapshot, ...]:
        return tuple(
            t for t in trades if t.setup_name == self.SETUP_NAME
        )

    def _momentum_closed(
        self, trades: tuple[ClosedTradeSnapshot, ...]
    ) -> tuple[ClosedTradeSnapshot, ...]:
        return tuple(
            t for t in trades if t.setup_name == self.SETUP_NAME
        )

    def _consecutive_losses_triggered(
        self,
        closed: tuple[ClosedTradeSnapshot, ...],
        settings: AlertRiskSettings,
    ) -> bool:
        recent = closed[: settings.consecutive_loss_limit]
        if len(recent) < settings.consecutive_loss_limit:
            return False
        return all(t.return_pct <= 0 for t in recent)

    def _rolling_drawdown_triggered(
        self,
        closed: tuple[ClosedTradeSnapshot, ...],
        settings: AlertRiskSettings,
    ) -> bool:
        window = closed[: settings.rolling_window_trades]
        if len(window) < settings.rolling_window_trades:
            return False
        cumulative = sum(t.return_pct for t in window)
        return cumulative <= settings.rolling_drawdown_limit_pct

    @staticmethod
    def _trade_risk_pct(plan) -> float:
        if plan.entry_price <= 0:
            return 0.0
        if plan.direction == "LONG":
            risk = plan.entry_price - plan.stop_price
        else:
            risk = plan.stop_price - plan.entry_price
        if risk <= 0:
            return 0.0
        return risk / plan.entry_price

    def _total_open_risk(
        self,
        open_trades: tuple[OpenTradeSnapshot, ...],
        settings: AlertRiskSettings,
    ) -> float:
        total = 0.0
        for trade in open_trades:
            if trade.position_risk_pct is not None:
                total += trade.position_risk_pct
            else:
                total += self._snapshot_risk_pct(trade)
        return total

    @staticmethod
    def _snapshot_risk_pct(trade: OpenTradeSnapshot) -> float:
        if trade.entry_price <= 0:
            return 0.0
        if trade.direction == "LONG":
            risk = trade.entry_price - trade.stop_price
        else:
            risk = trade.stop_price - trade.entry_price
        if risk <= 0:
            return 0.0
        return risk / trade.entry_price

    @staticmethod
    def _mega_cap_active_count(open_trades: tuple[OpenTradeSnapshot, ...]) -> int:
        return sum(
            1 for t in open_trades if t.symbol.upper() in MEGA_CAP_TECH_SYMBOLS
        )

    @staticmethod
    def _max_shares_or_dollars(
        plan,
        *,
        account_equity_usd: float | None,
        risk_pct: float,
    ) -> float | None:
        if account_equity_usd is None or account_equity_usd <= 0 or risk_pct <= 0:
            return None
        if plan.direction == "LONG":
            risk_per_share = plan.entry_price - plan.stop_price
        else:
            risk_per_share = plan.stop_price - plan.entry_price
        if risk_per_share <= 0:
            return None
        risk_dollars = account_equity_usd * risk_pct
        shares = risk_dollars / risk_per_share
        return round(shares * plan.entry_price, 2)
