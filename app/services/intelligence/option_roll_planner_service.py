from __future__ import annotations

from app.broker.option_chain_table import lookup_option_contract, quoted_ask, quoted_bid
from app.broker.option_greeks import sanitize_delta
from app.broker.option_utils import (
    parse_put_call_from_option_symbol,
    position_expiration_date,
    position_strike_price,
)
from app.models.intelligence_models import (
    OptionRollSuggestion,
    OptionsScorecard,
    OptionsStrikeCandidate,
)
from app.broker.option_delta_preference import (
    OptionDeltaBand,
    assignment_delta_threshold,
    resolve_option_delta_band,
    resolve_option_strategy_preferences,
)
from app.models.schwab_models import Position
from app.models.schwab_option_chain_models import OptionChain
from app.models.strategy_models import UserInvestmentProfile
from app.services.intelligence.options_scoring_service import OptionsScoringService


class OptionRollPlannerService:
    @staticmethod
    def build_roll_suggestions(
        *,
        positions: list[Position],
        symbol: str,
        option_chain: OptionChain | None,
        scorecard: OptionsScorecard | None,
        profile: UserInvestmentProfile | None = None,
        delta_band: OptionDeltaBand | None = None,
    ) -> list[OptionRollSuggestion]:
        if option_chain is None or scorecard is None:
            return []

        band = delta_band or resolve_option_delta_band(profile)
        prefs = resolve_option_strategy_preferences(profile)

        symbol_upper = symbol.upper()
        suggestions: list[OptionRollSuggestion] = []

        for position in positions:
            instrument = position.instrument
            if instrument.assetType != "OPTION":
                continue
            if position.shortQuantity <= 0:
                continue

            underlying = (instrument.underlyingSymbol or instrument.symbol or "").upper()
            if underlying != symbol_upper:
                continue

            strike = position_strike_price(position) or instrument.strikePrice
            expiration_date = position_expiration_date(position)
            put_call = instrument.putCall or parse_put_call_from_option_symbol(
                instrument.symbol or ""
            )
            if strike is None or expiration_date is None or not put_call:
                continue

            side = "call" if put_call == "CALL" else "put"
            expiration = expiration_date.isoformat()
            current_contract = lookup_option_contract(
                option_chain,
                expiration=expiration_date,
                strike=strike,
                put_call=put_call,
            )

            candidates = (
                scorecard.covered_call_candidates
                if side == "call"
                else scorecard.csp_candidates
            )
            close_delta = sanitize_delta(
                current_contract.delta if current_contract is not None else None
            )
            alternative = OptionRollPlannerService._pick_alternative(
                side=side,
                current_strike=strike,
                current_expiration=expiration,
                current_delta=close_delta,
                candidates=candidates,
                delta_band=band,
                preferred_dte=prefs.preferred_dte_days,
            )
            if alternative is None:
                continue

            current_bid = quoted_bid(current_contract)
            current_ask = quoted_ask(current_contract)
            estimated_credit = OptionRollPlannerService._estimate_roll_credit(
                current_contract_ask=current_ask,
                alternative_bid=alternative.bid,
            )

            rationale = OptionRollPlannerService._build_rationale(
                side=side,
                strike=strike,
                expiration=expiration,
                close_delta=close_delta,
                current_bid=current_bid,
                current_ask=current_ask,
                alternative=alternative,
                estimated_credit=estimated_credit,
                open_pnl=position.openProfitLoss,
                entry_credit_per_share=position.averagePrice,
            )

            suggestions.append(
                OptionRollSuggestion(
                    side=side,
                    current_strike=strike,
                    current_expiration=expiration,
                    suggested_strike=alternative.strike,
                    suggested_expiration=alternative.expiration,
                    current_delta=close_delta,
                    suggested_delta=alternative.delta,
                    estimated_credit=estimated_credit,
                    rationale=rationale,
                    action="roll",
                )
            )

        return suggestions

    @staticmethod
    def _pick_alternative(
        *,
        side: str,
        current_strike: float,
        current_expiration: str,
        current_delta: float | None,
        candidates: list[OptionsStrikeCandidate],
        delta_band: OptionDeltaBand,
        preferred_dte: int = 7,
    ) -> OptionsStrikeCandidate | None:
        if not candidates:
            return None

        current_exp = current_expiration[:10]
        current_abs_delta = abs(current_delta) if current_delta is not None else None

        ranked = sorted(
            candidates,
            key=lambda candidate: OptionRollPlannerService._roll_fit_score(
                candidate=candidate,
                side=side,
                current_strike=current_strike,
                current_exp=current_exp,
                current_abs_delta=current_abs_delta,
                delta_band=delta_band,
                preferred_dte=preferred_dte,
            ),
            reverse=True,
        )
        best_score = OptionRollPlannerService._roll_fit_score(
            candidate=ranked[0],
            side=side,
            current_strike=current_strike,
            current_exp=current_exp,
            current_abs_delta=current_abs_delta,
            delta_band=delta_band,
            preferred_dte=preferred_dte,
        )
        if best_score < 0:
            return None
        return ranked[0]

    @staticmethod
    def _roll_fit_score(
        *,
        candidate: OptionsStrikeCandidate,
        side: str,
        current_strike: float,
        current_exp: str,
        current_abs_delta: float | None,
        delta_band: OptionDeltaBand,
        preferred_dte: int = 7,
    ) -> float:
        assignment_threshold = assignment_delta_threshold(delta_band)
        if candidate.side != side:
            return -1.0
        if (
            abs(candidate.strike - current_strike) < 0.01
            and candidate.expiration[:10] == current_exp
        ):
            return -1.0

        score = candidate.score
        candidate_dte = OptionsScoringService._expiration_dte(candidate.expiration)
        score += OptionsScoringService._dte_score(
            candidate_dte,
            preferred_dte=preferred_dte,
        ) * 0.25

        if candidate.expiration[:10] > current_exp:
            score += 0.15

        candidate_delta = sanitize_delta(candidate.delta)
        if current_abs_delta is not None and candidate_delta is not None:
            candidate_abs_delta = abs(candidate_delta)
            if current_abs_delta >= assignment_threshold:
                if candidate_abs_delta < current_abs_delta:
                    score += 0.25
                else:
                    score -= 0.15
            if delta_band.min_delta <= candidate_abs_delta <= delta_band.max_delta:
                score += 0.10

        return score

    @staticmethod
    def _build_rationale(
        *,
        side: str,
        strike: float,
        expiration: str,
        close_delta: float | None,
        current_bid: float | None,
        current_ask: float | None,
        alternative: OptionsStrikeCandidate,
        estimated_credit: float | None,
        open_pnl: float | None = None,
        entry_credit_per_share: float | None = None,
    ) -> str:
        alt_dte = OptionsScoringService._expiration_dte(alternative.expiration)
        rationale = (
            f"Buy to close ${strike:g} {side} exp {expiration[:10]}"
            + (
                f" (delta {close_delta:.2f}"
                if close_delta is not None
                else " (delta n/a"
            )
            + (
                f", bid/ask {current_bid:.2f}/{current_ask:.2f})"
                if current_bid is not None and current_ask is not None
                else ")"
            )
            + f" → sell ${alternative.strike:g} {side} exp {alternative.expiration[:10]}"
            + (
                f" (delta {alternative.delta:.2f}, {alt_dte} DTE"
                if alternative.delta is not None
                else f" ({alt_dte} DTE"
            )
            + (
                f", bid/ask {alternative.bid:.2f}/{alternative.ask:.2f}, "
                f"OI {alternative.open_interest:,})"
                if alternative.bid is not None and alternative.ask is not None
                else ")"
            )
        )
        if close_delta is not None and alternative.delta is not None:
            if abs(close_delta) > abs(alternative.delta):
                rationale += "; lowers assignment delta"
        if current_ask is not None:
            rationale += f"; pay ~${current_ask * 100:,.0f}/contract to close (ask ${current_ask:.2f}/sh)"
        if alternative.bid is not None:
            rationale += (
                f"; collect ~${alternative.bid * 100:,.0f}/contract on new leg "
                f"(bid ${alternative.bid:.2f}/sh)"
            )
        if estimated_credit is not None:
            net_per_contract = estimated_credit * 100
            if estimated_credit >= 0:
                rationale += (
                    f"; estimated net credit ~${estimated_credit:.2f}/sh "
                    f"(~${net_per_contract:,.0f}/contract)"
                )
            else:
                rationale += (
                    f"; estimated net debit ~${abs(estimated_credit):.2f}/sh "
                    f"(~${abs(net_per_contract):,.0f}/contract to roll)"
                )
        if open_pnl is not None:
            rationale += f"; current open P/L ${open_pnl:,.0f}"
        if entry_credit_per_share and entry_credit_per_share > 0:
            rationale += (
                f"; original premium collected ~${entry_credit_per_share * 100:,.0f}/contract"
            )
        return rationale

    @staticmethod
    def _estimate_roll_credit(
        *,
        current_contract_ask: float | None,
        alternative_bid: float | None,
    ) -> float | None:
        if current_contract_ask is None or alternative_bid is None:
            return None
        return round(alternative_bid - current_contract_ask, 2)
