"""Tests for Momentum Breakout AlertRiskGate."""

from __future__ import annotations

import pytest

from trade_planner.alerts.risk_gate import AlertRiskGate, VOLUME_CLIMAX_MESSAGE
from trade_planner.alerts.risk_models import (
    AlertGateAction,
    AlertPriority,
    AlertRiskContext,
    AlertRiskSettings,
    ClosedTradeSnapshot,
    OpenTradeSnapshot,
)
from trade_planner.models import TradePlan, utc_now
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup

SETUP = MomentumBreakoutSetup.name


def _plan(
    *,
    symbol: str = "NVDA",
    entry: float = 100.0,
    stop: float = 99.0,
    target: float = 102.0,
) -> TradePlan:
    return TradePlan(
        symbol=symbol,
        setup_name=SETUP,
        direction="LONG",
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        risk_reward=2.0,
        confidence_score=70.0,
        generated_at=utc_now(),
        entry_is_stop=True,
    )


def _open(
    symbol: str,
    *,
    risk_pct: float | None = 0.01,
    entry: float = 100.0,
    stop: float = 99.0,
) -> OpenTradeSnapshot:
    return OpenTradeSnapshot(
        symbol=symbol,
        setup_name=SETUP,
        entry_price=entry,
        stop_price=stop,
        position_risk_pct=risk_pct,
    )


def _closed(return_pct: float) -> ClosedTradeSnapshot:
    return ClosedTradeSnapshot(setup_name=SETUP, return_pct=return_pct, symbol="X")


def _evaluate(**kwargs) -> AlertRiskContext:
    plan = kwargs.pop("plan", _plan())
    return AlertRiskContext(candidate_plan=plan, **kwargs)


@pytest.fixture
def gate() -> AlertRiskGate:
    return AlertRiskGate()


class TestMaxOpenPositions:
    def test_blocks_at_five_open(self, gate: AlertRiskGate) -> None:
        open_trades = tuple(_open(f"S{i}") for i in range(5))
        decision = gate.evaluate(_evaluate(open_trades=open_trades))
        assert not decision.allowed
        assert decision.action == AlertGateAction.BLOCK
        assert any("Max open positions" in r for r in decision.reasons)

    def test_allows_four_open(self, gate: AlertRiskGate) -> None:
        open_trades = tuple(_open(f"S{i}") for i in range(4))
        decision = gate.evaluate(_evaluate(open_trades=open_trades))
        assert decision.allowed


class TestOneTradePerSymbol:
    def test_blocks_duplicate_symbol(self, gate: AlertRiskGate) -> None:
        decision = gate.evaluate(
            _evaluate(
                plan=_plan(symbol="NVDA"),
                open_trades=(_open("NVDA"),),
                current_symbol="NVDA",
            )
        )
        assert not decision.allowed
        assert any("already open for NVDA" in r for r in decision.reasons)

    def test_allows_different_symbol(self, gate: AlertRiskGate) -> None:
        decision = gate.evaluate(
            _evaluate(open_trades=(_open("AAPL"),), current_symbol="NVDA")
        )
        assert decision.allowed


class TestPortfolioRiskCap:
    def test_blocks_when_total_exceeds_cap(self, gate: AlertRiskGate) -> None:
        open_trades = (
            _open("AAPL", risk_pct=0.02),
            _open("MSFT", risk_pct=0.02),
            _open("META", risk_pct=0.02),
        )
        decision = gate.evaluate(
            _evaluate(
                plan=_plan(entry=100, stop=94, target=112),
                open_trades=open_trades,
            )
        )
        assert not decision.allowed
        assert any("Portfolio risk cap" in r for r in decision.reasons)

    def test_size_down_when_elevated(self, gate: AlertRiskGate) -> None:
        open_trades = (_open("AAPL", risk_pct=0.045, stop=95.5),)
        decision = gate.evaluate(
            _evaluate(plan=_plan(stop=99.0), open_trades=open_trades)
        )
        assert decision.allowed
        assert decision.action == AlertGateAction.SIZE_DOWN


class TestConsecutiveLossCircuitBreaker:
    def test_blocks_after_four_losses(self, gate: AlertRiskGate) -> None:
        closed = tuple(_closed(-0.02) for _ in range(4))
        decision = gate.evaluate(_evaluate(recent_closed=closed))
        assert not decision.allowed
        assert any("last 4 closed" in r for r in decision.reasons)

    def test_allows_three_losses(self, gate: AlertRiskGate) -> None:
        closed = tuple(_closed(-0.02) for _ in range(3))
        decision = gate.evaluate(_evaluate(recent_closed=closed))
        assert decision.allowed


class TestRollingDrawdownCircuitBreaker:
    def test_blocks_on_twenty_trade_drawdown(self, gate: AlertRiskGate) -> None:
        closed = tuple(_closed(-0.006) for _ in range(20))
        decision = gate.evaluate(_evaluate(recent_closed=closed))
        assert not decision.allowed
        assert any("last 20 closed" in r for r in decision.reasons)

    def test_allows_nineteen_trade_window(self, gate: AlertRiskGate) -> None:
        closed = (_closed(0.01),) + tuple(_closed(-0.005) for _ in range(18))
        decision = gate.evaluate(_evaluate(recent_closed=closed))
        assert decision.allowed


class TestMegaCapCorrelationThrottle:
    def test_blocks_fourth_mega_cap_name(self, gate: AlertRiskGate) -> None:
        open_trades = (
            _open("AAPL", risk_pct=0.005),
            _open("MSFT", risk_pct=0.005),
            _open("AMZN", risk_pct=0.005),
        )
        decision = gate.evaluate(
            _evaluate(
                plan=_plan(symbol="NVDA"),
                open_trades=open_trades,
                current_symbol="NVDA",
            )
        )
        assert not decision.allowed
        assert any("Mega-cap tech correlation" in r for r in decision.reasons)

    def test_allows_non_mega_symbol(self, gate: AlertRiskGate) -> None:
        open_trades = (
            _open("AAPL", risk_pct=0.005),
            _open("MSFT", risk_pct=0.005),
            _open("AMZN", risk_pct=0.005),
        )
        decision = gate.evaluate(
            _evaluate(
                plan=_plan(symbol="JPM"),
                open_trades=open_trades,
                current_symbol="JPM",
            )
        )
        assert decision.allowed


class TestVolumeClimaxWarning:
    def test_warns_without_block(self, gate: AlertRiskGate) -> None:
        decision = gate.evaluate(_evaluate(volume_ratio=3.5))
        assert decision.allowed
        assert decision.action == AlertGateAction.WARN
        assert any(VOLUME_CLIMAX_MESSAGE in r for r in decision.reasons)
        assert decision.alert_priority == AlertPriority.MEDIUM


class TestAllowPath:
    def test_allow_high_priority(self, gate: AlertRiskGate) -> None:
        decision = gate.evaluate(_evaluate())
        assert decision.allowed
        assert decision.action == AlertGateAction.ALLOW
        assert decision.alert_priority == AlertPriority.HIGH
        assert decision.recommended_position_risk_pct == pytest.approx(0.01)

    def test_max_notional_with_equity(self, gate: AlertRiskGate) -> None:
        decision = gate.evaluate(
            _evaluate(account_equity_usd=100_000.0)
        )
        assert decision.max_shares_or_dollars == pytest.approx(100_000.0)


class TestIgnoresOtherSetups:
    def test_other_setup_trades_not_counted(self, gate: AlertRiskGate) -> None:
        foreign = OpenTradeSnapshot(
            symbol="NVDA",
            setup_name="Pullback",
            entry_price=100,
            stop_price=95,
        )
        decision = gate.evaluate(_evaluate(open_trades=(foreign,)))
        assert decision.allowed

    def test_closed_other_setup_ignored_for_breaker(self, gate: AlertRiskGate) -> None:
        closed = (
            ClosedTradeSnapshot(setup_name="Pullback", return_pct=-0.05),
        ) * 4
        decision = gate.evaluate(_evaluate(recent_closed=closed))
        assert decision.allowed


class TestCustomSettings:
    def test_custom_max_positions(self, gate: AlertRiskGate) -> None:
        settings = AlertRiskSettings(max_open_positions=2)
        open_trades = (_open("A"), _open("B"))
        decision = gate.evaluate(
            _evaluate(open_trades=open_trades, settings=settings)
        )
        assert not decision.allowed
