from __future__ import annotations

from datetime import date

from app.broker.option_chain_table import (
    fair_option_price,
    lookup_option_contract,
    quoted_ask,
    quoted_bid,
)
from app.broker.option_greeks import resolve_option_greeks, sanitize_delta
from app.broker.option_utils import (
    parse_put_call_from_option_symbol,
    position_expiration_date,
    position_strike_price,
)
from app.broker.position_metrics import (
    portfolio_liquidation_value,
    position_open_profit_loss,
    position_open_profit_loss_pct,
    position_portfolio_weight_pct,
)
from app.models.intelligence_models import OptionRollSuggestion, SymbolIntelligence
from app.models.schwab_models import Position, SchwabAccounts
from app.models.schwab_option_chain_models import OptionChain
from app.models.symbol_analysis_precomputed_models import (
    ClosePathOutcome,
    ComparePathOption,
    HeldOptionDecisionDrivers,
    HeldOptionOutcomes,
    HoldPathOutcome,
    OptionLegOutcome,
    RollCashPicture,
    RollPathOutcome,
    SymbolAnalysisPrecomputed,
)
from app.services.intelligence.options_scoring_service import OptionsScoringService

OPTION_CONTRACT_MULTIPLIER = 100.0
PNL_ACTION_TRIGGER_PCT = -30.0
ASSIGNMENT_DELTA_THRESHOLD = OptionsScoringService.ASSIGNMENT_DELTA_THRESHOLD


class SymbolAnalysisPrecomputedService:
    @staticmethod
    def build(
        *,
        symbol: str,
        account: SchwabAccounts,
        positions: list[Position],
        intelligence: SymbolIntelligence | None,
        option_chain: OptionChain | None,
        underlying_price: float | None = None,
    ) -> SymbolAnalysisPrecomputed | None:
        symbol_upper = symbol.strip().upper()
        held_short_options = SymbolAnalysisPrecomputedService._short_options_for_symbol(
            positions, symbol_upper
        )
        if not held_short_options and intelligence is None:
            return None

        if intelligence is not None and not isinstance(intelligence, SymbolIntelligence):
            intelligence = None
            if not held_short_options:
                return None

        if underlying_price is None and intelligence and intelligence.options_scorecard:
            underlying_price = intelligence.options_scorecard.underlying_price
        if underlying_price is None and option_chain is not None:
            underlying_price = option_chain.underlyingPrice or (
                option_chain.underlying.last
                if option_chain.underlying and option_chain.underlying.last
                else None
            )

        portfolio_value = portfolio_liquidation_value(
            account=account, positions=positions
        )
        roll_suggestions = (
            list(intelligence.roll_suggestions) if intelligence is not None else []
        )
        if (
            not roll_suggestions
            and intelligence is not None
            and option_chain is not None
            and intelligence.options_scorecard is not None
            and held_short_options
        ):
            from app.services.intelligence.option_roll_planner_service import (
                OptionRollPlannerService,
            )

            roll_suggestions = OptionRollPlannerService.build_roll_suggestions(
                positions=held_short_options,
                symbol=symbol_upper,
                option_chain=option_chain,
                scorecard=intelligence.options_scorecard,
            )
        held_outcomes: list[HeldOptionOutcomes] = []

        for position in held_short_options:
            outcome = SymbolAnalysisPrecomputedService._build_held_option_outcomes(
                position=position,
                symbol=symbol_upper,
                option_chain=option_chain,
                underlying_price=underlying_price,
                portfolio_value=portfolio_value,
                roll_suggestions=roll_suggestions,
            )
            if outcome is not None:
                held_outcomes.append(outcome)

        if (
            not held_outcomes
            and intelligence is None
            and not roll_suggestions
        ):
            return None

        return SymbolAnalysisPrecomputed(
            symbol=symbol_upper,
            underlying_price=underlying_price,
            options_scorecard=(
                intelligence.options_scorecard if intelligence is not None else None
            ),
            roll_suggestions=roll_suggestions,
            held_option_outcomes=held_outcomes,
        )

    @staticmethod
    def _short_options_for_symbol(
        positions: list[Position], symbol_upper: str
    ) -> list[Position]:
        matched: list[Position] = []
        for position in positions:
            if position.instrument.assetType != "OPTION":
                continue
            if position.shortQuantity <= 0:
                continue
            underlying = (
                position.instrument.underlyingSymbol
                or position.instrument.symbol
                or ""
            ).upper()
            if underlying.split()[0] == symbol_upper:
                matched.append(position)
        return matched

    @staticmethod
    def _build_held_option_outcomes(
        *,
        position: Position,
        symbol: str,
        option_chain: OptionChain | None,
        underlying_price: float | None,
        portfolio_value: float | None,
        roll_suggestions: list[OptionRollSuggestion],
    ) -> HeldOptionOutcomes | None:
        instrument = position.instrument
        put_call = instrument.putCall or parse_put_call_from_option_symbol(
            instrument.symbol or ""
        )
        expiration_date = position_expiration_date(position)
        strike = position_strike_price(position) or instrument.strikePrice
        if put_call is None or expiration_date is None or strike is None:
            return None

        contracts = position.shortQuantity
        side = "call" if put_call == "CALL" else "put"
        expiration_iso = expiration_date.isoformat()

        contract = None
        if option_chain is not None:
            contract = lookup_option_contract(
                option_chain,
                expiration=expiration_date,
                strike=strike,
                put_call=put_call,
            )

        bid = quoted_bid(contract)
        ask = quoted_ask(contract)
        mark = fair_option_price(contract)
        greeks = resolve_option_greeks(
            contract,
            chain=option_chain,
            underlying_price=underlying_price,
            put_call=put_call,
            strike=strike,
            expiration=expiration_date,
        )
        delta = sanitize_delta(greeks.delta if greeks else None)
        dte = max((expiration_date - date.today()).days, 0)

        entry_per_share = position.averagePrice or position.averageShortPrice
        open_pnl = position_open_profit_loss(position)
        open_pnl_pct = position_open_profit_loss_pct(position)
        weight_pct = position_portfolio_weight_pct(position, portfolio_value)

        current_leg = OptionLegOutcome(
            put_call=put_call,
            side=side,
            strike=strike,
            expiration=expiration_iso,
            contracts=contracts,
            days_to_expiration=dte,
            delta=delta,
            bid=bid,
            ask=ask,
            mark=mark,
            cash_per_contract=round(ask * OPTION_CONTRACT_MULTIPLIER, 2)
            if ask is not None
            else None,
            cash_direction="pay" if ask is not None else None,
        )

        close = ClosePathOutcome(
            cost_per_share=ask,
            cost_per_contract=round(ask * OPTION_CONTRACT_MULTIPLIER, 2)
            if ask is not None
            else None,
            open_pnl=open_pnl,
        )

        itm = None
        assignment_note = None
        if underlying_price is not None:
            if put_call == "PUT":
                itm = underlying_price < strike
                assignment_note = (
                    SymbolAnalysisPrecomputedService._short_put_hold_note(
                        symbol=symbol,
                        underlying_price=underlying_price,
                        strike=strike,
                        itm=itm,
                        entry_per_share=entry_per_share,
                    )
                )
            else:
                itm = underlying_price > strike
                assignment_note = (
                    SymbolAnalysisPrecomputedService._short_call_hold_note(
                        symbol=symbol,
                        underlying_price=underlying_price,
                        strike=strike,
                        itm=itm,
                        entry_per_share=entry_per_share,
                    )
                )

        hold = HoldPathOutcome(
            days_to_expiration=dte,
            delta=delta,
            underlying_price=underlying_price,
            in_the_money=itm,
            assignment_note=assignment_note,
        )

        entry_premium_per_contract = (
            round(entry_per_share * OPTION_CONTRACT_MULTIPLIER, 2)
            if entry_per_share and entry_per_share > 0
            else None
        )

        roll = SymbolAnalysisPrecomputedService._build_roll_path(
            roll_suggestions=roll_suggestions,
            strike=strike,
            expiration_iso=expiration_iso,
            side=side,
            put_call=put_call,
            contracts=contracts,
            option_chain=option_chain,
            close_bid=bid,
            close_ask=ask,
            close_mark=mark,
            close_dte=dte,
            close_delta=delta,
            entry_premium_per_contract=entry_premium_per_contract,
        )

        matched_roll_suggestion = (
            SymbolAnalysisPrecomputedService._match_roll_suggestion(
                roll_suggestions,
                strike=strike,
                expiration_iso=expiration_iso,
                side=side,
            )
            if roll_suggestions
            else None
        )
        roll_cash_picture = SymbolAnalysisPrecomputedService._resolve_roll_cash_picture(
            roll=roll,
            entry_premium_per_contract=entry_premium_per_contract,
            close_cost_per_contract=current_leg.cash_per_contract,
            matched_suggestion=matched_roll_suggestion,
        )

        drivers = HeldOptionDecisionDrivers(
            portfolio_weight_pct=weight_pct,
            open_pnl=open_pnl,
            open_pnl_pct=open_pnl_pct,
            entry_premium_per_share=entry_per_share,
            entry_premium_per_contract=entry_premium_per_contract,
            action_trigger=SymbolAnalysisPrecomputedService._action_trigger(
                open_pnl_pct=open_pnl_pct,
                delta=delta,
                dte=dte,
            ),
        )

        return HeldOptionOutcomes(
            drivers=drivers,
            current_leg=current_leg,
            roll=roll,
            roll_cash_picture=roll_cash_picture,
            close=close,
            hold=hold,
            compare_paths=SymbolAnalysisPrecomputedService._build_compare_paths(
                roll=roll,
                close=close,
                hold=hold,
                side=side,
            ),
        )

    @staticmethod
    def _stock_at_price(symbol: str, price: float) -> str:
        return f"{symbol.strip().upper()} at ${price:.2f}"

    @staticmethod
    def _effective_assignment_basis_note(
        strike: float, entry_per_share: float | None
    ) -> str:
        if entry_per_share is None or entry_per_share <= 0:
            return ""
        effective = strike - entry_per_share
        return (
            f"; effective cost ~${effective:.2f}/share if assigned "
            f"(${strike:g} strike minus ${entry_per_share:.2f} premium collected)"
        )

    @staticmethod
    def _short_put_hold_note(
        *,
        symbol: str,
        underlying_price: float,
        strike: float,
        itm: bool,
        entry_per_share: float | None,
    ) -> str:
        ticker = symbol.strip().upper()
        stock = SymbolAnalysisPrecomputedService._stock_at_price(
            ticker, underlying_price
        )
        basis = SymbolAnalysisPrecomputedService._effective_assignment_basis_note(
            strike, entry_per_share
        )
        premium_kept = ""
        if entry_per_share is not None and entry_per_share > 0:
            premium_kept = (
                f" (~${entry_per_share * OPTION_CONTRACT_MULTIPLIER:,.0f}/contract)"
            )

        if itm:
            return (
                f"{stock} is below your ${strike:g} put strike — "
                f"assignment to buy 100 {ticker} shares at ${strike:g} is likely if "
                f"still below the strike at expiration (cash-secured put wheel){basis}."
            )

        return (
            f"{stock} is above your ${strike:g} put strike — "
            f"if {ticker} is still above ${strike:g} at expiration, keep full premium "
            f"collected{premium_kept}; if {ticker} falls below ${strike:g} by expiry, "
            f"assignment buys 100 shares at ${strike:g}{basis}."
        )

    @staticmethod
    def _short_call_hold_note(
        *,
        symbol: str,
        underlying_price: float,
        strike: float,
        itm: bool,
        entry_per_share: float | None,
    ) -> str:
        ticker = symbol.strip().upper()
        stock = SymbolAnalysisPrecomputedService._stock_at_price(
            ticker, underlying_price
        )
        premium_kept = ""
        if entry_per_share is not None and entry_per_share > 0:
            premium_kept = (
                f" (~${entry_per_share * OPTION_CONTRACT_MULTIPLIER:,.0f}/contract)"
            )

        if itm:
            return (
                f"{stock} is above your ${strike:g} call strike — "
                f"100 {ticker} shares may be called away at ${strike:g} if still above "
                "the strike at expiration."
            )

        return (
            f"{stock} is below your ${strike:g} call strike — "
            f"if {ticker} is still below ${strike:g} at expiration, keep full premium "
            f"collected{premium_kept}; if {ticker} rises above ${strike:g} by expiry, "
            "shares may be called away."
        )

    @staticmethod
    def _build_compare_paths(
        *,
        roll: RollPathOutcome | None,
        close: ClosePathOutcome,
        hold: HoldPathOutcome,
        side: str,
    ) -> list[ComparePathOption]:
        options: list[ComparePathOption] = []

        if roll is not None:
            roll_lines = [
                f"Buy to close ${roll.close_leg.strike:g} {side} exp {roll.close_leg.expiration[:10]}",
                f"Sell ${roll.open_leg.strike:g} {side} exp {roll.open_leg.expiration[:10]}",
            ]
            if roll.close_leg.cash_per_contract is not None:
                roll_lines.append(
                    f"Pay ~${roll.close_leg.cash_per_contract:,.0f} to close"
                )
            if roll.open_leg.cash_per_contract is not None:
                roll_lines.append(
                    f"Collect ~${roll.open_leg.cash_per_contract:,.0f} on new leg"
                )
            if roll.net_credit_per_contract is not None:
                if roll.is_net_credit:
                    roll_lines.append(
                        f"Net credit ~${roll.net_credit_per_contract:,.0f} per contract"
                    )
                else:
                    roll_lines.append(
                        f"Net debit ~${abs(roll.net_credit_per_contract):,.0f} per contract"
                    )
            if roll.open_leg.delta is not None:
                roll_lines.append(f"New leg delta {roll.open_leg.delta:.2f}")
            options.append(
                ComparePathOption(path="roll", title="Roll", lines=roll_lines)
            )

        close_lines: list[str] = []
        if close.cost_per_contract is not None and close.open_pnl is not None:
            close_lines.append(
                f"Pay ~${close.cost_per_contract:,.0f} to buy to close; "
                f"locks in open P/L ${close.open_pnl:,.0f}"
            )
        elif close.cost_per_contract is not None:
            close_lines.append(f"Pay ~${close.cost_per_contract:,.0f} to buy to close")
        elif close.open_pnl is not None:
            close_lines.append(f"Locks in open P/L ${close.open_pnl:,.0f}")
        if close_lines:
            options.append(
                ComparePathOption(path="close", title="Close now", lines=close_lines)
            )

        hold_lines: list[str] = []
        if hold.assignment_note:
            hold_lines.append(hold.assignment_note)
        else:
            if hold.days_to_expiration is not None:
                hold_lines.append(f"{hold.days_to_expiration} DTE remaining")
            if hold.delta is not None:
                hold_lines.append(f"Delta {hold.delta:.2f}")
        if hold_lines:
            options.append(
                ComparePathOption(path="hold", title="Hold to expiration", lines=hold_lines)
            )

        return options

    @staticmethod
    def _action_trigger(
        *,
        open_pnl_pct: float | None,
        delta: float | None,
        dte: int,
    ) -> str | None:
        triggers: list[str] = []
        if open_pnl_pct is not None and open_pnl_pct <= PNL_ACTION_TRIGGER_PCT:
            triggers.append(f"open P/L {open_pnl_pct:+.1f}% (loss rule)")
        if delta is not None and abs(delta) >= ASSIGNMENT_DELTA_THRESHOLD:
            triggers.append(f"delta {delta:.2f} (assignment proximity)")
        if dte <= 3:
            triggers.append(f"{dte} DTE (near expiration)")
        if not triggers:
            return None
        return "; ".join(triggers)

    @staticmethod
    def _format_cash_amount(amount: float) -> str:
        return f"${abs(amount):,.0f}"

    @staticmethod
    def _build_roll_cash_picture(
        *,
        entry_premium_per_contract: float | None,
        close_cost_per_contract: float | None,
        open_collect_per_contract: float | None,
    ) -> RollCashPicture | None:
        roll_net = None
        if close_cost_per_contract is not None and open_collect_per_contract is not None:
            roll_net = round(
                open_collect_per_contract - close_cost_per_contract, 2
            )

        net_cash_after_roll = None
        if (
            entry_premium_per_contract is not None
            and close_cost_per_contract is not None
            and open_collect_per_contract is not None
        ):
            net_cash_after_roll = round(
                entry_premium_per_contract
                - close_cost_per_contract
                + open_collect_per_contract,
                2,
            )
        elif entry_premium_per_contract is not None and roll_net is not None:
            net_cash_after_roll = round(entry_premium_per_contract + roll_net, 2)

        loss_on_closed_put = None
        if (
            entry_premium_per_contract is not None
            and close_cost_per_contract is not None
        ):
            loss_on_closed_put = round(
                entry_premium_per_contract - close_cost_per_contract, 2
            )

        if (
            entry_premium_per_contract is None
            and close_cost_per_contract is None
            and open_collect_per_contract is None
            and roll_net is None
        ):
            return None

        summary = SymbolAnalysisPrecomputedService._roll_cash_summary(
            net_cash_after_roll=net_cash_after_roll,
            loss_on_closed_put=loss_on_closed_put,
            roll_net=roll_net,
            open_collect_per_contract=open_collect_per_contract,
        )

        return RollCashPicture(
            entry_premium_per_contract=entry_premium_per_contract,
            close_cost_per_contract=close_cost_per_contract,
            open_collect_per_contract=open_collect_per_contract,
            roll_net_per_contract=roll_net,
            net_cash_after_roll_per_contract=net_cash_after_roll,
            loss_on_closed_put_per_contract=loss_on_closed_put,
            summary=summary,
        )

    @staticmethod
    def _roll_cash_summary(
        *,
        net_cash_after_roll: float | None,
        loss_on_closed_put: float | None,
        roll_net: float | None,
        open_collect_per_contract: float | None,
    ) -> str | None:
        if net_cash_after_roll is None and roll_net is None:
            return None

        parts: list[str] = []
        fmt = SymbolAnalysisPrecomputedService._format_cash_amount

        if net_cash_after_roll is not None:
            if net_cash_after_roll >= 0:
                parts.append(
                    f"You keep {fmt(net_cash_after_roll)} more cash in your account "
                    "than before you sold the first put."
                )
            else:
                parts.append(
                    f"You are {fmt(net_cash_after_roll)} net behind vs before you "
                    "sold the first put."
                )

        if loss_on_closed_put is not None and loss_on_closed_put < 0:
            parts.append(
                f"Closing the old put realizes a {fmt(loss_on_closed_put)} loss on "
                "that leg."
            )

        if roll_net is not None:
            if roll_net < 0:
                parts.append(
                    f"The roll itself is a {fmt(roll_net)} debit today — you pay "
                    "more to close than you collect on the new put."
                )
            elif roll_net > 0:
                parts.append(
                    f"The roll itself brings in a {fmt(roll_net)} credit today."
                )

        if open_collect_per_contract is not None and open_collect_per_contract > 0:
            parts.append(
                f"You still hold the new short put; you keep its "
                f"{fmt(open_collect_per_contract)} premium if it expires out of the money."
            )

        return " ".join(parts) if parts else None

    @staticmethod
    def _resolve_roll_cash_picture(
        *,
        roll: RollPathOutcome | None,
        entry_premium_per_contract: float | None,
        close_cost_per_contract: float | None,
        matched_suggestion: OptionRollSuggestion | None,
    ) -> RollCashPicture | None:
        if roll is not None and roll.cash_picture is not None:
            return roll.cash_picture

        close_cost = close_cost_per_contract
        open_collect = None
        if roll is not None:
            close_cost = roll.close_leg.cash_per_contract
            open_collect = roll.open_leg.cash_per_contract
        elif matched_suggestion is not None:
            if (
                close_cost is not None
                and matched_suggestion.estimated_credit is not None
            ):
                open_collect = round(
                    close_cost
                    + matched_suggestion.estimated_credit * OPTION_CONTRACT_MULTIPLIER,
                    2,
                )

        return SymbolAnalysisPrecomputedService._build_roll_cash_picture(
            entry_premium_per_contract=entry_premium_per_contract,
            close_cost_per_contract=close_cost,
            open_collect_per_contract=open_collect,
        )

    @staticmethod
    def _build_roll_path(
        *,
        roll_suggestions: list[OptionRollSuggestion],
        strike: float,
        expiration_iso: str,
        side: str,
        put_call: str,
        contracts: float,
        option_chain: OptionChain | None,
        close_bid: float | None,
        close_ask: float | None,
        close_mark: float | None,
        close_dte: int,
        close_delta: float | None,
        entry_premium_per_contract: float | None = None,
    ) -> RollPathOutcome | None:
        suggestion = SymbolAnalysisPrecomputedService._match_roll_suggestion(
            roll_suggestions,
            strike=strike,
            expiration_iso=expiration_iso,
            side=side,
        )
        if suggestion is None:
            return None

        open_expiration = date.fromisoformat(suggestion.suggested_expiration[:10])
        open_contract = None
        if option_chain is not None:
            open_contract = lookup_option_contract(
                option_chain,
                expiration=open_expiration,
                strike=suggestion.suggested_strike,
                put_call=put_call,
            )
        open_bid = quoted_bid(open_contract)
        open_ask = quoted_ask(open_contract)
        open_mark = fair_option_price(open_contract)
        open_delta = sanitize_delta(
            open_contract.delta if open_contract is not None else suggestion.suggested_delta
        )
        open_dte = OptionsScoringService._expiration_dte(suggestion.suggested_expiration)

        close_leg = OptionLegOutcome(
            put_call=put_call,
            side=side,
            strike=suggestion.current_strike,
            expiration=suggestion.current_expiration[:10],
            contracts=contracts,
            days_to_expiration=close_dte,
            delta=close_delta if close_delta is not None else suggestion.current_delta,
            bid=close_bid,
            ask=close_ask,
            mark=close_mark,
            cash_per_contract=round(close_ask * OPTION_CONTRACT_MULTIPLIER, 2)
            if close_ask is not None
            else None,
            cash_direction="pay" if close_ask is not None else None,
        )
        open_leg = OptionLegOutcome(
            put_call=put_call,
            side=side,
            strike=suggestion.suggested_strike,
            expiration=suggestion.suggested_expiration[:10],
            contracts=contracts,
            days_to_expiration=open_dte,
            delta=open_delta,
            bid=open_bid,
            ask=open_ask,
            mark=open_mark,
            cash_per_contract=round(open_bid * OPTION_CONTRACT_MULTIPLIER, 2)
            if open_bid is not None
            else None,
            cash_direction="collect" if open_bid is not None else None,
        )

        net_per_share = (
            round(open_bid - close_ask, 2)
            if open_bid is not None and close_ask is not None
            else suggestion.estimated_credit
        )
        net_per_contract = (
            round(net_per_share * OPTION_CONTRACT_MULTIPLIER, 2)
            if net_per_share is not None
            else None
        )

        cash_picture = SymbolAnalysisPrecomputedService._build_roll_cash_picture(
            entry_premium_per_contract=entry_premium_per_contract,
            close_cost_per_contract=close_leg.cash_per_contract,
            open_collect_per_contract=open_leg.cash_per_contract,
        )

        return RollPathOutcome(
            close_leg=close_leg,
            open_leg=open_leg,
            net_credit_per_share=net_per_share,
            net_credit_per_contract=net_per_contract,
            is_net_credit=net_per_share is None or net_per_share >= 0,
            cash_picture=cash_picture,
        )

    @staticmethod
    def _match_roll_suggestion(
        roll_suggestions: list[OptionRollSuggestion],
        *,
        strike: float,
        expiration_iso: str,
        side: str,
    ) -> OptionRollSuggestion | None:
        exp = expiration_iso[:10]

        def strike_matches(suggestion: OptionRollSuggestion) -> bool:
            return abs(suggestion.current_strike - strike) < 0.01

        def expiration_matches(suggestion: OptionRollSuggestion) -> bool:
            return suggestion.current_expiration[:10] == exp

        for suggestion in roll_suggestions:
            if suggestion.side != side:
                continue
            if not strike_matches(suggestion):
                continue
            if expiration_matches(suggestion):
                return suggestion

        for suggestion in roll_suggestions:
            if suggestion.side != side:
                continue
            if strike_matches(suggestion):
                return suggestion

        side_matches = [s for s in roll_suggestions if s.side == side]
        if len(side_matches) == 1:
            return side_matches[0]

        if len(roll_suggestions) == 1:
            return roll_suggestions[0]

        return None
