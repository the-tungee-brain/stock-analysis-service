from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Literal

from app.broker.option_greeks import (
    black_scholes_option_price,
    find_strike_for_abs_delta,
)

SHARES_PER_CONTRACT = 100
ALLOWED_LOOKBACK_YEARS = frozenset({5, 10, 15})
DEFAULT_WHEEL_BACKTEST_DTE_DAYS = 30

TRADE_ACTION_LABELS: dict[str, str] = {
    "sell_csp": "Sell cash-secured put",
    "put_expired": "Put expired (OTM)",
    "put_assigned": "Put assigned — buy stock",
    "sell_cc": "Sell covered call",
    "call_expired": "Call expired (OTM)",
    "call_assigned": "Call assigned — shares called away",
    "capital_top_up": "Added cash (CSP needs more)",
}


class WheelPhase(str, Enum):
    CASH = "cash"
    SHORT_PUT = "short_put"
    LONG_STOCK = "long_stock"
    SHORT_CALL = "short_call"


@dataclass(frozen=True)
class PriceBar:
    trading_date: date
    close: float


@dataclass(frozen=True)
class WheelBacktestConfig:
    symbol: str
    lookback_years: int
    target_delta: float
    dte_days: int
    contracts: int = 1
    risk_free_rate: float = 0.04
    vol_lookback_days: int = 21
    fee_per_contract_usd: float = 0.65
    iv_floor_pct: float = 12.0
    iv_cap_pct: float = 120.0
    premium_haircut: float = 0.95
    """Multiply theoretical premium to approximate bid vs mid (conservative)."""
    maintain_one_lot: bool = True
    """Top up cash when strike×100 exceeds wallet so one CSP lot can keep trading."""


@dataclass
class OpenOptionLeg:
    put_call: Literal["PUT", "CALL"]
    strike: float
    expiration_idx: int
    expiration_date: date
    entry_idx: int
    entry_date: date
    premium_per_share: float
    iv_percent: float


@dataclass
class WheelTradeEvent:
    event_date: date
    action: str
    put_call: str | None
    strike: float | None
    premium_usd: float
    fees_usd: float
    close_at_event: float
    stock_price_usd: float | None = None
    effective_entry_price_usd: float | None = None
    effective_exit_price_usd: float | None = None
    premium_per_share: float | None = None
    dte_days: int | None = None
    expiration_date: date | None = None
    iv_percent: float | None = None
    wheel_cycle: int | None = None
    cycle_month: str | None = None
    collateral_reserved_usd: float | None = None
    cash_flow_usd: float | None = None
    note: str | None = None


@dataclass
class WheelBacktestResult:
    symbol: str
    lookback_years: int
    start_date: date
    """First CSP date; P/L and CAGR use this through end_date."""
    history_start_date: date
    """First bar in the downloaded window (vol warmup before first trade)."""
    first_trade_date: date | None
    last_trade_date: date | None
    csp_rounds: int
    end_date: date
    trading_days: int
    config: dict[str, Any]
    assumptions: list[str]
    starting_cash_usd: float
    ending_equity_usd: float
    total_pl_usd: float
    total_return_pct: float
    cagr_pct: float | None
    buy_and_hold_return_pct: float
    buy_and_hold_cagr_pct: float | None
    total_premium_collected_usd: float
    total_fees_usd: float
    total_dividends_usd: float
    put_assignments: int
    puts_expired_otm: int
    calls_assigned: int
    calls_expired_otm: int
    completed_wheel_cycles: int
    skipped_trades_insufficient_cash: int
    capital_top_ups_usd: float
    buy_and_hold_ending_usd: float
    spot_price_at_start: float
    spot_price_at_end: float
    initial_stock_price_usd: float
    initial_put_strike_usd: float
    initial_collateral_usd: float
    wheel_cycles: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    equity_curve: list[dict[str, Any]]
    annual_summary: list[dict[str, Any]]


def realized_volatility_percent(
    closes: list[float],
    end_idx: int,
    *,
    lookback: int,
    floor_pct: float,
    cap_pct: float,
) -> float:
    if end_idx < 1:
        return floor_pct
    start = max(0, end_idx - lookback)
    window = closes[start : end_idx + 1]
    if len(window) < 2:
        return floor_pct

    log_returns: list[float] = []
    for i in range(1, len(window)):
        prev = window[i - 1]
        curr = window[i]
        if prev <= 0 or curr <= 0:
            continue
        log_returns.append(math.log(curr / prev))

    if len(log_returns) < 2:
        return floor_pct

    mean = sum(log_returns) / len(log_returns)
    variance = sum((value - mean) ** 2 for value in log_returns) / (len(log_returns) - 1)
    std = math.sqrt(variance)
    annualized = std * math.sqrt(252) * 100.0
    return max(floor_pct, min(cap_pct, annualized))


def _premium_per_share(
    *,
    underlying: float,
    strike: float,
    days: int,
    put_call: str,
    iv_percent: float,
    risk_free_rate: float,
    haircut: float,
) -> float | None:
    theoretical = black_scholes_option_price(
        underlying=underlying,
        strike=strike,
        days_to_expiration=days,
        put_call=put_call,
        iv_percent=iv_percent,
        risk_free_rate=risk_free_rate,
    )
    if theoretical is None or theoretical <= 0:
        return None
    return theoretical * haircut


def run_wheel_backtest(
    bars: list[PriceBar],
    *,
    dividends: dict[date, float],
    splits: dict[date, float],
    config: WheelBacktestConfig,
) -> WheelBacktestResult:
    if config.lookback_years not in ALLOWED_LOOKBACK_YEARS:
        raise ValueError(f"lookback_years must be one of {sorted(ALLOWED_LOOKBACK_YEARS)}")
    if len(bars) < config.vol_lookback_days + config.dte_days + 5:
        raise ValueError("Not enough trading history for the requested horizon")

    closes = [bar.close for bar in bars]
    n = len(bars)
    shares_per_lot = SHARES_PER_CONTRACT * config.contracts

    phase = WheelPhase.CASH
    cash = 0.0
    shares = 0
    cost_basis_per_share = 0.0
    open_leg: OpenOptionLeg | None = None
    completed_cycles = 0
    capital_top_ups = 0.0
    wheel_cycle_id = 0

    total_premium = 0.0
    total_fees = 0.0
    total_dividends = 0.0
    put_assignments = 0
    puts_expired_otm = 0
    calls_assigned = 0
    calls_expired_otm = 0
    skipped_cash = 0
    csp_blocked_active = False
    first_trade_date: date | None = None
    last_trade_date: date | None = None
    collateral_reserved = 0.0
    current_cycle_month: str | None = None
    trades: list[WheelTradeEvent] = []
    equity_curve: list[dict[str, Any]] = []

    next_entry_idx = config.vol_lookback_days

    def apply_splits(day: date) -> None:
        nonlocal shares, cost_basis_per_share, open_leg
        ratio = splits.get(day)
        if ratio is None or ratio <= 0 or shares <= 0:
            return
        shares = int(round(shares * ratio))
        if cost_basis_per_share > 0:
            cost_basis_per_share /= ratio
        if open_leg is not None:
            open_leg.strike = round(open_leg.strike / ratio, 2)

    def credit_dividends(day: date, close_price: float) -> None:
        nonlocal cash, total_dividends
        if shares <= 0:
            return
        amount_per_share = dividends.get(day)
        if amount_per_share is None or amount_per_share <= 0:
            return
        payout = amount_per_share * shares
        cash += payout
        total_dividends += payout

    def equity_mark(close_price: float) -> float:
        return cash + shares * close_price

    def record(
        day: date,
        action: str,
        *,
        put_call: str | None = None,
        strike: float | None = None,
        premium_usd: float = 0.0,
        fees_usd: float = 0.0,
        close_at_event: float,
        stock_price_usd: float | None = None,
        effective_entry_price_usd: float | None = None,
        effective_exit_price_usd: float | None = None,
        premium_per_share: float | None = None,
        dte_days: int | None = None,
        expiration_date: date | None = None,
        iv_percent: float | None = None,
        wheel_cycle: int | None = None,
        cycle_month: str | None = None,
        collateral_reserved_usd: float | None = None,
        cash_flow_usd: float | None = None,
        note: str | None = None,
    ) -> None:
        trades.append(
            WheelTradeEvent(
                event_date=day,
                action=action,
                put_call=put_call,
                strike=strike,
                premium_usd=round(premium_usd, 2),
                fees_usd=round(fees_usd, 2),
                close_at_event=round(close_at_event, 4),
                stock_price_usd=stock_price_usd,
                effective_entry_price_usd=effective_entry_price_usd,
                effective_exit_price_usd=effective_exit_price_usd,
                premium_per_share=premium_per_share,
                dte_days=dte_days,
                expiration_date=expiration_date,
                iv_percent=iv_percent,
                wheel_cycle=wheel_cycle,
                cycle_month=cycle_month,
                collateral_reserved_usd=collateral_reserved_usd,
                cash_flow_usd=cash_flow_usd,
                note=note,
            )
        )

    def open_short_put(idx: int) -> None:
        nonlocal phase, cash, open_leg, total_premium, total_fees, skipped_cash, next_entry_idx
        nonlocal capital_top_ups, wheel_cycle_id, collateral_reserved, current_cycle_month
        nonlocal csp_blocked_active, first_trade_date, last_trade_date

        bar = bars[idx]
        exp_idx = min(idx + config.dte_days, n - 1)
        dte = max(exp_idx - idx, 1)
        iv = realized_volatility_percent(
            closes,
            idx,
            lookback=config.vol_lookback_days,
            floor_pct=config.iv_floor_pct,
            cap_pct=config.iv_cap_pct,
        )
        strike = find_strike_for_abs_delta(
            underlying=bar.close,
            days_to_expiration=dte,
            put_call="PUT",
            target_abs_delta=config.target_delta,
            iv_percent=iv,
            risk_free_rate=config.risk_free_rate,
        )
        if strike is None:
            next_entry_idx = idx + 1
            return

        collateral = strike * shares_per_lot
        if cash < collateral:
            if config.maintain_one_lot:
                target_cash = collateral * 1.05
                if cash < target_cash:
                    top_up = target_cash - cash
                    capital_top_ups += top_up
                    cash = target_cash
                    record(
                        bar.trading_date,
                        "capital_top_up",
                        strike=strike,
                        close_at_event=bar.close,
                        stock_price_usd=round(bar.close, 4),
                        cash_flow_usd=round(top_up, 2),
                        note=(
                            f"Deposited cash — CSP needs ${collateral:,.0f} secured "
                            f"(${strike:.2f} × 100)"
                        ),
                    )
            else:
                if not csp_blocked_active:
                    skipped_cash += 1
                    csp_blocked_active = True
                next_entry_idx = idx + 1
                return

        csp_blocked_active = False

        premium_ps = _premium_per_share(
            underlying=bar.close,
            strike=strike,
            days=dte,
            put_call="PUT",
            iv_percent=iv,
            risk_free_rate=config.risk_free_rate,
            haircut=config.premium_haircut,
        )
        if premium_ps is None:
            next_entry_idx = idx + 1
            return

        fees = config.fee_per_contract_usd * config.contracts
        premium_total = premium_ps * shares_per_lot
        cash += premium_total - fees
        total_premium += premium_total
        total_fees += fees
        open_leg = OpenOptionLeg(
            put_call="PUT",
            strike=strike,
            expiration_idx=exp_idx,
            expiration_date=bars[exp_idx].trading_date,
            entry_idx=idx,
            entry_date=bar.trading_date,
            premium_per_share=premium_ps,
            iv_percent=iv,
        )
        wheel_cycle_id += 1
        current_cycle_month = bar.trading_date.strftime("%Y-%m")
        collateral_reserved = collateral
        phase = WheelPhase.SHORT_PUT
        if first_trade_date is None:
            first_trade_date = bar.trading_date
        last_trade_date = bar.trading_date
        record(
            bar.trading_date,
            "sell_csp",
            put_call="PUT",
            strike=strike,
            premium_usd=premium_total,
            fees_usd=fees,
            close_at_event=bar.close,
            stock_price_usd=round(bar.close, 4),
            premium_per_share=round(premium_ps, 4),
            dte_days=dte,
            expiration_date=bars[exp_idx].trading_date,
            iv_percent=round(iv, 2),
            wheel_cycle=wheel_cycle_id,
            cycle_month=current_cycle_month,
            collateral_reserved_usd=round(collateral_reserved, 2),
            cash_flow_usd=round(premium_total - fees, 2),
            note=(
                f"Sell put ${strike:.2f} · stock ${bar.close:.2f} · "
                f"${premium_ps:.2f}/sh premium · {dte} DTE · "
                f"${collateral:,.0f} cash secured (100 sh)"
            ),
        )

    def open_short_call(idx: int) -> None:
        nonlocal phase, cash, open_leg, total_premium, total_fees, next_entry_idx
        nonlocal last_trade_date

        bar = bars[idx]
        exp_idx = min(idx + config.dte_days, n - 1)
        dte = max(exp_idx - idx, 1)
        iv = realized_volatility_percent(
            closes,
            idx,
            lookback=config.vol_lookback_days,
            floor_pct=config.iv_floor_pct,
            cap_pct=config.iv_cap_pct,
        )
        strike = find_strike_for_abs_delta(
            underlying=bar.close,
            days_to_expiration=dte,
            put_call="CALL",
            target_abs_delta=config.target_delta,
            iv_percent=iv,
            risk_free_rate=config.risk_free_rate,
        )
        if strike is None:
            next_entry_idx = idx + 1
            return

        premium_ps = _premium_per_share(
            underlying=bar.close,
            strike=strike,
            days=dte,
            put_call="CALL",
            iv_percent=iv,
            risk_free_rate=config.risk_free_rate,
            haircut=config.premium_haircut,
        )
        if premium_ps is None:
            next_entry_idx = idx + 1
            return

        fees = config.fee_per_contract_usd * config.contracts
        premium_total = premium_ps * shares_per_lot
        cash += premium_total - fees
        total_premium += premium_total
        total_fees += fees
        open_leg = OpenOptionLeg(
            put_call="CALL",
            strike=strike,
            expiration_idx=exp_idx,
            expiration_date=bars[exp_idx].trading_date,
            entry_idx=idx,
            entry_date=bar.trading_date,
            premium_per_share=premium_ps,
            iv_percent=iv,
        )
        phase = WheelPhase.SHORT_CALL
        last_trade_date = bar.trading_date
        record(
            bar.trading_date,
            "sell_cc",
            put_call="CALL",
            strike=strike,
            premium_usd=premium_total,
            fees_usd=fees,
            close_at_event=bar.close,
            stock_price_usd=round(bar.close, 4),
            premium_per_share=round(premium_ps, 4),
            dte_days=dte,
            expiration_date=bars[exp_idx].trading_date,
            iv_percent=round(iv, 2),
            wheel_cycle=wheel_cycle_id,
            cycle_month=current_cycle_month,
            cash_flow_usd=round(premium_total - fees, 2),
            note=(
                f"Sell call ${strike:.2f} · stock ${bar.close:.2f} · "
                f"${premium_ps:.2f}/sh premium · {dte} DTE"
            ),
        )

    def settle_put(idx: int) -> None:
        nonlocal phase, cash, shares, cost_basis_per_share, open_leg
        nonlocal put_assignments, puts_expired_otm, next_entry_idx
        nonlocal collateral_reserved, current_cycle_month, last_trade_date

        leg = open_leg
        if leg is None:
            return
        bar = bars[idx]
        close = bar.close

        assign_cost = leg.strike * shares_per_lot
        if close <= leg.strike:
            cash -= assign_cost
            shares = shares_per_lot
            cost_basis_per_share = leg.strike - leg.premium_per_share
            put_assignments += 1
            effective_entry = leg.strike - leg.premium_per_share
            last_trade_date = bar.trading_date
            record(
                bar.trading_date,
                "put_assigned",
                put_call="PUT",
                strike=leg.strike,
                close_at_event=close,
                stock_price_usd=round(close, 4),
                effective_entry_price_usd=round(effective_entry, 4),
                expiration_date=leg.expiration_date,
                wheel_cycle=wheel_cycle_id,
                cycle_month=current_cycle_month,
                cash_flow_usd=round(-assign_cost, 2),
                note=(
                    f"Bought 100 sh at ${leg.strike:.2f} · close ${close:.2f} · "
                    f"effective ${effective_entry:.2f}/sh · collateral released"
                ),
            )
            collateral_reserved = 0.0
            phase = WheelPhase.LONG_STOCK
        else:
            puts_expired_otm += 1
            last_trade_date = bar.trading_date
            record(
                bar.trading_date,
                "put_expired",
                put_call="PUT",
                strike=leg.strike,
                close_at_event=close,
                stock_price_usd=round(close, 4),
                expiration_date=leg.expiration_date,
                wheel_cycle=wheel_cycle_id,
                cycle_month=current_cycle_month,
                cash_flow_usd=0.0,
                note=f"Keep premium · close ${close:.2f} > put ${leg.strike:.2f}",
            )
            collateral_reserved = 0.0
            current_cycle_month = None
            phase = WheelPhase.CASH

        open_leg = None
        next_entry_idx = idx + 1

    def settle_call(idx: int) -> None:
        nonlocal phase, cash, shares, cost_basis_per_share, open_leg
        nonlocal calls_assigned, calls_expired_otm, completed_cycles, next_entry_idx
        nonlocal current_cycle_month, last_trade_date

        leg = open_leg
        if leg is None:
            return
        bar = bars[idx]
        close = bar.close

        proceeds = leg.strike * shares_per_lot
        if close >= leg.strike:
            cash += proceeds
            shares = 0
            cost_basis_per_share = 0.0
            calls_assigned += 1
            completed_cycles += 1
            last_trade_date = bar.trading_date
            record(
                bar.trading_date,
                "call_assigned",
                put_call="CALL",
                strike=leg.strike,
                close_at_event=close,
                stock_price_usd=round(close, 4),
                effective_exit_price_usd=round(leg.strike, 4),
                expiration_date=leg.expiration_date,
                wheel_cycle=wheel_cycle_id,
                cycle_month=current_cycle_month,
                cash_flow_usd=round(proceeds, 2),
                note=(
                    f"Sold 100 sh at ${leg.strike:.2f} · close ${close:.2f}"
                ),
            )
            current_cycle_month = None
            phase = WheelPhase.CASH
        else:
            calls_expired_otm += 1
            last_trade_date = bar.trading_date
            record(
                bar.trading_date,
                "call_expired",
                put_call="CALL",
                strike=leg.strike,
                close_at_event=close,
                stock_price_usd=round(close, 4),
                expiration_date=leg.expiration_date,
                wheel_cycle=wheel_cycle_id,
                cycle_month=current_cycle_month,
                cash_flow_usd=0.0,
                note=f"Keep premium · close ${close:.2f} < call ${leg.strike:.2f}",
            )
            phase = WheelPhase.LONG_STOCK

        open_leg = None
        next_entry_idx = idx + 1

    # Seed starting cash from first CSP collateral at first trade window.
    seed_idx = config.vol_lookback_days
    seed_iv = realized_volatility_percent(
        closes,
        seed_idx,
        lookback=config.vol_lookback_days,
        floor_pct=config.iv_floor_pct,
        cap_pct=config.iv_cap_pct,
    )
    seed_strike = find_strike_for_abs_delta(
        underlying=bars[seed_idx].close,
        days_to_expiration=config.dte_days,
        put_call="PUT",
        target_abs_delta=config.target_delta,
        iv_percent=seed_iv,
        risk_free_rate=config.risk_free_rate,
    )
    if seed_strike is None:
        raise ValueError("Could not derive initial collateral from price history")
    initial_stock_price = bars[seed_idx].close
    initial_put_strike = seed_strike
    initial_collateral = seed_strike * shares_per_lot
    cash = initial_collateral * 1.05
    starting_cash = cash

    for idx in range(n):
        bar = bars[idx]
        apply_splits(bar.trading_date)
        credit_dividends(bar.trading_date, bar.close)

        if open_leg is not None and idx == open_leg.expiration_idx:
            if open_leg.put_call == "PUT":
                settle_put(idx)
            else:
                settle_call(idx)

        if phase == WheelPhase.CASH and idx >= next_entry_idx and open_leg is None:
            open_short_put(idx)
        elif phase == WheelPhase.LONG_STOCK and idx >= next_entry_idx and open_leg is None:
            open_short_call(idx)

        equity_curve.append(
            {
                "date": bar.trading_date.isoformat(),
                "equityUsd": round(equity_mark(bar.close), 2),
                "cashUsd": round(cash, 2),
                "collateralReservedUsd": round(collateral_reserved, 2),
                "availableCashUsd": round(max(cash - collateral_reserved, 0), 2),
                "shares": shares,
                "phase": phase.value,
            }
        )

    history_start_date = bars[0].trading_date
    end_date = bars[-1].trading_date
    start_date = first_trade_date or history_start_date
    ending_equity = equity_mark(bars[-1].close)
    csp_rounds = sum(1 for event in trades if event.action == "sell_csp")
    capital_deployed = starting_cash + capital_top_ups
    total_pl_usd = ending_equity - capital_deployed
    total_return_pct = (
        ((ending_equity / capital_deployed) - 1.0) * 100.0 if capital_deployed > 0 else 0.0
    )

    elapsed_years = max((end_date - start_date).days, 1) / 365.25
    cagr_pct = None
    if elapsed_years >= 1 and capital_deployed > 0 and ending_equity > 0:
        cagr_pct = round(
            ((ending_equity / capital_deployed) ** (1 / elapsed_years) - 1) * 100.0,
            2,
        )

    first_csp = next((e for e in trades if e.action == "sell_csp"), None)
    spot_start = (
        first_csp.stock_price_usd
        if first_csp and first_csp.stock_price_usd is not None
        else bars[seed_idx].close
    )
    spot_end = bars[-1].close
    buy_hold_shares = starting_cash / spot_start if spot_start > 0 else 0.0
    bah_dividends = 0.0
    first_trade_idx = next(
        (i for i, bar in enumerate(bars) if bar.trading_date == start_date),
        seed_idx,
    )
    for bar in bars[first_trade_idx:]:
        div_per_share = dividends.get(bar.trading_date)
        if div_per_share:
            bah_dividends += div_per_share * buy_hold_shares
    buy_hold_equity = buy_hold_shares * spot_end + bah_dividends
    buy_hold_return = (
        ((buy_hold_equity / starting_cash) - 1.0) * 100.0 if starting_cash > 0 else 0.0
    )
    buy_hold_cagr = None
    if elapsed_years >= 1 and starting_cash > 0 and buy_hold_equity > 0:
        buy_hold_cagr = round(
            ((buy_hold_equity / starting_cash) ** (1 / elapsed_years) - 1) * 100.0,
            2,
        )

    wheel_cycles = _build_wheel_cycles(trades)
    annual_summary = _build_annual_summary(equity_curve, trades)

    assumptions = [
        (
            f"Starting wallet ${starting_cash:,.0f} = ${initial_collateral:,.0f} cash secured "
            f"for 1 CSP (put strike ${initial_put_strike:.2f} × 100 sh, stock "
            f"${initial_stock_price:.2f} on first trade date) plus 5% buffer."
        ),
    ]
    if config.maintain_one_lot:
        assumptions.append(
            "When cash cannot cover the next CSP (strike × 100 shares), the model "
            "adds enough cash to keep selling one CSP."
        )
        if capital_top_ups > 0:
            assumptions.append(
                f"Additional deposits ${capital_top_ups:,.0f} over the run (stock rose, "
                f"so each CSP needed more collateral). Total P/L and CAGR use all cash "
                f"you put in: ${capital_deployed:,.0f} (starting wallet plus additions)."
            )
    else:
        assumptions.append(
            "Fixed wallet only: no deposits after day one. Total P/L = ending equity "
            "minus starting wallet. New CSPs only when cash covers strike × 100."
        )
    if skipped_cash > 0:
        assumptions.append(
            f"{skipped_cash} period(s) with insufficient cash for the next CSP "
            f"(strike × 100 exceeds wallet; no new puts until cash is enough)."
        )
    if csp_rounds > 0 and last_trade_date and last_trade_date < end_date:
        assumptions.append(
            f"Last option trade {last_trade_date.isoformat()}; "
            f"wallet could not fund further CSP collateral before {end_date.isoformat()}."
        )
    assumptions.extend(
        [
        "Premiums estimated with Black-Scholes using rolling realized volatility "
        f"({config.vol_lookback_days} trading-day window), not historical option quotes.",
        "Strikes on standard US grids: $2.50 spacing below $200 spot, $5 at/above $200 "
        "(e.g. 100, 102.5, 105 or 500, 505, 510).",
        f"Short options opened at {config.premium_haircut * 100:.0f}% of theoretical "
        "(conservative vs mid/bid).",
        "European-style expiration: puts assign if close <= strike; calls assign if close >= strike.",
        "No early assignment, rolls, or margin interest modeled.",
        "Stock splits adjust share count and strike; dividends paid on ex-dates while long shares.",
        "Total-return-adjusted daily closes (yfinance auto_adjust=True; "
        "historical prices differ from raw quotes).",
        f"Fixed {config.dte_days} trading-day holding period between entries (approx. calendar DTE).",
        (
            f"Buy & hold benchmark: ${starting_cash:,.0f} buys shares at "
            f"${spot_start:.2f} on first trade date; dividends collected; "
            f"marked at ${spot_end:.2f} on last day."
        ),
        ]
    )

    return WheelBacktestResult(
        symbol=config.symbol.upper(),
        lookback_years=config.lookback_years,
        start_date=start_date,
        history_start_date=history_start_date,
        first_trade_date=first_trade_date,
        last_trade_date=last_trade_date,
        csp_rounds=csp_rounds,
        end_date=end_date,
        trading_days=n,
        config={
            "targetDelta": config.target_delta,
            "dteDays": config.dte_days,
            "contracts": config.contracts,
            "riskFreeRate": config.risk_free_rate,
            "volLookbackDays": config.vol_lookback_days,
            "feePerContractUsd": config.fee_per_contract_usd,
            "premiumHaircut": config.premium_haircut,
            "maintainOneLot": config.maintain_one_lot,
        },
        assumptions=assumptions,
        starting_cash_usd=round(starting_cash, 2),
        ending_equity_usd=round(ending_equity, 2),
        total_pl_usd=round(total_pl_usd, 2),
        total_return_pct=round(total_return_pct, 2),
        cagr_pct=cagr_pct,
        buy_and_hold_return_pct=round(buy_hold_return, 2),
        buy_and_hold_cagr_pct=buy_hold_cagr,
        total_premium_collected_usd=round(total_premium, 2),
        total_fees_usd=round(total_fees, 2),
        total_dividends_usd=round(total_dividends, 2),
        put_assignments=put_assignments,
        puts_expired_otm=puts_expired_otm,
        calls_assigned=calls_assigned,
        calls_expired_otm=calls_expired_otm,
        completed_wheel_cycles=completed_cycles,
        skipped_trades_insufficient_cash=skipped_cash,
        capital_top_ups_usd=round(capital_top_ups, 2),
        buy_and_hold_ending_usd=round(buy_hold_equity, 2),
        spot_price_at_start=round(spot_start, 4),
        spot_price_at_end=round(spot_end, 4),
        initial_stock_price_usd=round(initial_stock_price, 4),
        initial_put_strike_usd=round(initial_put_strike, 4),
        initial_collateral_usd=round(initial_collateral, 2),
        wheel_cycles=wheel_cycles,
        trades=[_trade_to_dict(event) for event in trades],
        equity_curve=equity_curve,
        annual_summary=annual_summary,
    )


def _trade_to_dict(event: WheelTradeEvent) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "date": event.event_date.isoformat(),
        "action": event.action,
        "label": TRADE_ACTION_LABELS.get(event.action, event.action),
        "putCall": event.put_call,
        "strike": event.strike,
        "premiumUsd": event.premium_usd,
        "feesUsd": event.fees_usd,
        "close": event.close_at_event,
        "note": event.note,
    }
    if event.stock_price_usd is not None:
        payload["stockPrice"] = event.stock_price_usd
    if event.effective_entry_price_usd is not None:
        payload["effectiveEntryPrice"] = event.effective_entry_price_usd
    if event.effective_exit_price_usd is not None:
        payload["effectiveExitPrice"] = event.effective_exit_price_usd
    if event.premium_per_share is not None:
        payload["premiumPerShare"] = event.premium_per_share
    if event.dte_days is not None:
        payload["dteDays"] = event.dte_days
    if event.expiration_date is not None:
        payload["expirationDate"] = event.expiration_date.isoformat()
    if event.iv_percent is not None:
        payload["ivPercent"] = event.iv_percent
    if event.wheel_cycle is not None:
        payload["wheelCycle"] = event.wheel_cycle
    if event.cycle_month is not None:
        payload["cycleMonth"] = event.cycle_month
    if event.collateral_reserved_usd is not None:
        payload["collateralReservedUsd"] = event.collateral_reserved_usd
    if event.cash_flow_usd is not None:
        payload["cashFlowUsd"] = event.cash_flow_usd
    return payload


def _build_wheel_cycles(trades: list[WheelTradeEvent]) -> list[dict[str, Any]]:
    cycles: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for event in trades:
        if event.action == "put_assigned":
            if current and not current.get("completed"):
                cycles.append(current)
            current = {
                "cycle": len(cycles) + 1,
                "putStrike": event.strike,
                "stockEntryDate": event.event_date.isoformat(),
                "stockEntryClose": event.stock_price_usd or event.close_at_event,
                "effectiveEntryPrice": event.effective_entry_price_usd,
                "callStrike": None,
                "stockExitDate": None,
                "stockExitClose": None,
                "effectiveExitPrice": None,
                "completed": False,
            }
        elif event.action == "sell_cc" and current is not None:
            current["callStrike"] = event.strike
        elif event.action == "call_assigned" and current is not None:
            current["stockExitDate"] = event.event_date.isoformat()
            current["stockExitClose"] = event.stock_price_usd or event.close_at_event
            current["effectiveExitPrice"] = event.effective_exit_price_usd
            current["completed"] = True
            entry = current.get("effectiveEntryPrice")
            exit_px = current.get("effectiveExitPrice")
            if entry is not None and exit_px is not None:
                current["stockRoundTripPlUsd"] = round(
                    (exit_px - entry) * SHARES_PER_CONTRACT, 2
                )
            cycles.append(current)
            current = None

    if current is not None and not current.get("completed"):
        cycles.append(current)

    return cycles


def _build_annual_summary(
    equity_curve: list[dict[str, Any]],
    trades: list[WheelTradeEvent],
) -> list[dict[str, Any]]:
    by_year: dict[int, dict[str, Any]] = {}

    for point in equity_curve:
        year = int(point["date"][:4])
        entry = by_year.setdefault(
            year,
            {
                "year": year,
                "startEquityUsd": point["equityUsd"],
                "endEquityUsd": point["equityUsd"],
                "premiumUsd": 0.0,
                "feesUsd": 0.0,
            },
        )
        entry["endEquityUsd"] = point["equityUsd"]

    for trade in trades:
        year = trade.event_date.year
        entry = by_year.setdefault(
            year,
            {
                "year": year,
                "startEquityUsd": 0.0,
                "endEquityUsd": 0.0,
                "premiumUsd": 0.0,
                "feesUsd": 0.0,
            },
        )
        if trade.premium_usd > 0:
            entry["premiumUsd"] = round(entry["premiumUsd"] + trade.premium_usd, 2)
        entry["feesUsd"] = round(entry["feesUsd"] + trade.fees_usd, 2)

    summary: list[dict[str, Any]] = []
    for year in sorted(by_year):
        row = by_year[year]
        start = row["startEquityUsd"]
        end = row["endEquityUsd"]
        pl_usd = end - start
        row_return = (pl_usd / start) * 100.0 if start > 0 else 0.0
        summary.append(
            {
                "year": year,
                "startEquityUsd": round(start, 2),
                "endEquityUsd": round(end, 2),
                "plUsd": round(pl_usd, 2),
                "returnPct": round(row_return, 2),
                "premiumUsd": row["premiumUsd"],
                "feesUsd": row["feesUsd"],
            }
        )
    return summary
