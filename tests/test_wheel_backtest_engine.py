from datetime import date, timedelta

import pytest

from app.broker.option_greeks import (
    black_scholes_option_price,
    estimate_delta_black_scholes,
    find_strike_for_abs_delta,
    snap_strike_to_standard_grid,
    standard_strike_increment,
)
from app.services.strategy.wheel_backtest_engine import (
    PriceBar,
    WheelBacktestConfig,
    _premium_per_share,
    realized_volatility_percent,
    run_wheel_backtest,
)


def _flat_bars(
    *,
    days: int,
    price: float = 100.0,
    start: date | None = None,
) -> list[PriceBar]:
    start_date = start or date(2020, 1, 2)
    return [
        PriceBar(
            trading_date=start_date + timedelta(days=offset),
            close=price,
        )
        for offset in range(days)
    ]


def test_black_scholes_put_price_positive():
    price = black_scholes_option_price(
        underlying=100.0,
        strike=95.0,
        days_to_expiration=7,
        put_call="PUT",
        iv_percent=25.0,
    )
    assert price is not None
    assert price > 0


def test_find_strike_targets_put_delta():
    strike = find_strike_for_abs_delta(
        underlying=100.0,
        days_to_expiration=7,
        put_call="PUT",
        target_abs_delta=0.25,
        iv_percent=25.0,
    )
    assert strike is not None
    assert strike < 100.0
    step = standard_strike_increment(100.0)
    assert abs(strike % step) < 0.01
    delta = estimate_delta_black_scholes(
        underlying=100.0,
        strike=strike,
        days_to_expiration=7,
        put_call="PUT",
        iv_percent=25.0,
    )
    assert delta is not None
    assert abs(abs(delta) - 0.25) < 0.05


def test_standard_strike_grid_spacing():
    assert standard_strike_increment(100.0) == 2.5
    assert standard_strike_increment(199.0) == 2.5
    assert standard_strike_increment(200.0) == 5.0
    assert standard_strike_increment(500.0) == 5.0
    assert snap_strike_to_standard_grid(171.08, 100.0) == 170.0
    assert snap_strike_to_standard_grid(503.2, 500.0) == 505.0


def test_find_strike_on_five_dollar_grid_for_high_price():
    strike = find_strike_for_abs_delta(
        underlying=500.0,
        days_to_expiration=30,
        put_call="PUT",
        target_abs_delta=0.25,
        iv_percent=22.0,
    )
    assert strike is not None
    assert strike < 500.0
    assert strike % 5.0 == 0.0


def test_premium_matches_black_scholes_with_haircut():
    iv = 25.0
    strike = 95.0
    dte = 30
    theoretical = black_scholes_option_price(
        underlying=100.0,
        strike=strike,
        days_to_expiration=dte,
        put_call="PUT",
        iv_percent=iv,
        risk_free_rate=0.04,
    )
    assert theoretical is not None
    modeled = _premium_per_share(
        underlying=100.0,
        strike=strike,
        days=dte,
        put_call="PUT",
        iv_percent=iv,
        risk_free_rate=0.04,
        haircut=0.95,
    )
    assert modeled == pytest.approx(theoretical * 0.95, rel=1e-6)


def test_assignment_and_call_cash_flows():
    """Settlement: put assign debits strike×100; call assign credits strike×100."""
    prices: list[float] = [100.0] * 15 + [85.0] * 8 + [110.0] * 20
    bars = [
        PriceBar(trading_date=date(2020, 1, 2) + timedelta(days=i), close=p)
        for i, p in enumerate(prices)
    ]
    result = run_wheel_backtest(
        bars,
        dividends={},
        splits={},
        config=WheelBacktestConfig(
            symbol="TEST",
            lookback_years=5,
            target_delta=0.25,
            dte_days=5,
            vol_lookback_days=5,
            maintain_one_lot=False,
        ),
    )
    put_assign = next(t for t in result.trades if t["action"] == "put_assigned")
    call_assign = next(t for t in result.trades if t["action"] == "call_assigned")
    assert put_assign["cashFlowUsd"] == pytest.approx(-put_assign["strike"] * 100, rel=0.01)
    assert call_assign["cashFlowUsd"] == pytest.approx(call_assign["strike"] * 100, rel=0.01)
    assert put_assign["strike"] % 2.5 == 0.0
    assert call_assign["strike"] % 2.5 == 0.0
    completed = [c for c in result.wheel_cycles if c.get("completed")][0]
    assert completed["stockRoundTripPlUsd"] == pytest.approx(
        (call_assign["strike"] - completed["effectiveEntryPrice"]) * 100,
        rel=0.01,
    )


def test_put_assigns_when_close_below_strike():
    prices = [100.0] * 12 + [70.0] * 28
    bars = [
        PriceBar(trading_date=date(2020, 1, 2) + timedelta(days=i), close=p)
        for i, p in enumerate(prices)
    ]
    result = run_wheel_backtest(
        bars,
        dividends={},
        splits={},
        config=WheelBacktestConfig(
            symbol="TEST",
            lookback_years=5,
            target_delta=0.25,
            dte_days=5,
            vol_lookback_days=5,
        ),
    )
    assert result.put_assignments >= 1
    assert any(trade["action"] == "put_assigned" for trade in result.trades)


def test_put_expires_otm_when_price_above_strike():
    bars = _flat_bars(days=40, price=150.0)
    result = run_wheel_backtest(
        bars,
        dividends={},
        splits={},
        config=WheelBacktestConfig(
            symbol="TEST",
            lookback_years=5,
            target_delta=0.25,
            dte_days=5,
            vol_lookback_days=5,
        ),
    )
    assert result.puts_expired_otm >= 1
    assert result.put_assignments == 0


def test_full_wheel_cycle_call_assigned():
    prices: list[float] = [100.0] * 15 + [85.0] * 8 + [110.0] * 20
    bars = [
        PriceBar(trading_date=date(2020, 1, 2) + timedelta(days=i), close=p)
        for i, p in enumerate(prices)
    ]
    result = run_wheel_backtest(
        bars,
        dividends={},
        splits={},
        config=WheelBacktestConfig(
            symbol="TEST",
            lookback_years=5,
            target_delta=0.25,
            dte_days=5,
            vol_lookback_days=5,
        ),
    )
    assert result.put_assignments >= 1
    assert result.calls_assigned >= 1
    assert result.completed_wheel_cycles >= 1


def test_dividends_credited_while_long():
    bars = _flat_bars(days=35, price=100.0)
    ex_date = bars[25].trading_date
    result = run_wheel_backtest(
        bars,
        dividends={ex_date: 0.50},
        splits={},
        config=WheelBacktestConfig(
            symbol="TEST",
            lookback_years=5,
            target_delta=0.25,
            dte_days=5,
            vol_lookback_days=5,
        ),
    )
    assert result.total_dividends_usd >= 0


def test_realized_vol_clamped():
    closes = [100.0 + (i % 3) * 0.01 for i in range(30)]
    vol = realized_volatility_percent(
        closes,
        29,
        lookback=21,
        floor_pct=12.0,
        cap_pct=80.0,
    )
    assert 12.0 <= vol <= 80.0


def test_fixed_capital_has_no_top_ups():
    prices = [100.0] * 30 + [150.0] * 30
    bars = [
        PriceBar(trading_date=date(2020, 1, 2) + timedelta(days=i), close=p)
        for i, p in enumerate(prices)
    ]
    result = run_wheel_backtest(
        bars,
        dividends={},
        splits={},
        config=WheelBacktestConfig(
            symbol="TEST",
            lookback_years=5,
            target_delta=0.25,
            dte_days=5,
            vol_lookback_days=5,
            maintain_one_lot=False,
        ),
    )
    assert result.capital_top_ups_usd == 0.0
    assert result.total_pl_usd == round(
        result.ending_equity_usd - result.starting_cash_usd, 2
    )


def test_maintain_one_lot_keeps_trading_when_spot_rises():
    """Flat cash can lag SPY notional; top-ups allow continued CSP selling."""
    prices = [100.0] * 30 + [150.0] * 30
    bars = [
        PriceBar(trading_date=date(2020, 1, 2) + timedelta(days=i), close=p)
        for i, p in enumerate(prices)
    ]
    result = run_wheel_backtest(
        bars,
        dividends={},
        splits={},
        config=WheelBacktestConfig(
            symbol="TEST",
            lookback_years=5,
            target_delta=0.25,
            dte_days=5,
            vol_lookback_days=5,
            maintain_one_lot=True,
        ),
    )
    assert result.capital_top_ups_usd > 0
    assert len(result.trades) > 2


def test_buy_and_hold_uses_starting_cash_not_price_ratio():
    prices = [100.0] * 20 + [200.0] * 20
    bars = [
        PriceBar(trading_date=date(2020, 1, 2) + timedelta(days=i), close=p)
        for i, p in enumerate(prices)
    ]
    result = run_wheel_backtest(
        bars,
        dividends={},
        splits={},
        config=WheelBacktestConfig(
            symbol="TEST",
            lookback_years=5,
            target_delta=0.25,
            dte_days=5,
            vol_lookback_days=5,
        ),
    )
    # 100% price move => ~100% return on cash invested, not 259% style price-only
    assert result.buy_and_hold_return_pct < 150.0
    assert result.buy_and_hold_return_pct > 50.0


def test_trade_log_includes_label_dte_and_cycle():
    prices = [100.0] * 15 + [85.0] * 8 + [110.0] * 20
    bars = [
        PriceBar(trading_date=date(2020, 1, 2) + timedelta(days=i), close=p)
        for i, p in enumerate(prices)
    ]
    result = run_wheel_backtest(
        bars,
        dividends={},
        splits={},
        config=WheelBacktestConfig(
            symbol="TEST",
            lookback_years=5,
            target_delta=0.25,
            dte_days=30,
            vol_lookback_days=5,
        ),
    )
    sell_csp = next(t for t in result.trades if t["action"] == "sell_csp")
    assert sell_csp["label"] == "Sell cash-secured put"
    assert sell_csp["dteDays"] == 30
    assert sell_csp["wheelCycle"] == 1
    assert sell_csp["premiumPerShare"] is not None


def test_wheel_cycles_include_entry_and_exit():
    prices = [100.0] * 15 + [85.0] * 8 + [110.0] * 20
    bars = [
        PriceBar(trading_date=date(2020, 1, 2) + timedelta(days=i), close=p)
        for i, p in enumerate(prices)
    ]
    result = run_wheel_backtest(
        bars,
        dividends={},
        splits={},
        config=WheelBacktestConfig(
            symbol="TEST",
            lookback_years=5,
            target_delta=0.25,
            dte_days=5,
            vol_lookback_days=5,
        ),
    )
    completed = [c for c in result.wheel_cycles if c.get("completed")]
    assert completed
    assert completed[0]["effectiveEntryPrice"] is not None
    assert completed[0]["effectiveExitPrice"] is not None


def test_annual_summary_includes_pl_usd():
    bars = _flat_bars(days=40, price=100.0)
    result = run_wheel_backtest(
        bars,
        dividends={},
        splits={},
        config=WheelBacktestConfig(
            symbol="TEST",
            lookback_years=5,
            target_delta=0.25,
            dte_days=5,
            vol_lookback_days=5,
        ),
    )
    assert result.total_pl_usd == round(
        result.ending_equity_usd - result.starting_cash_usd, 2
    )
    assert result.annual_summary
    row = result.annual_summary[0]
    assert row["plUsd"] == round(row["endEquityUsd"] - row["startEquityUsd"], 2)


def test_insufficient_history_raises():
    bars = _flat_bars(days=10)
    with pytest.raises(ValueError, match="Not enough"):
        run_wheel_backtest(
            bars,
            dividends={},
            splits={},
            config=WheelBacktestConfig(
                symbol="TEST",
                lookback_years=5,
                target_delta=0.25,
                dte_days=7,
            ),
        )
