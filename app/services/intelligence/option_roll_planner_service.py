from __future__ import annotations

from app.broker.option_chain_table import lookup_option_contract, quoted_ask, quoted_bid
from app.broker.option_greeks import sanitize_delta
from app.broker.option_utils import (
    parse_put_call_from_option_symbol,
    position_expiration_date,
)
from app.models.intelligence_models import (
    OptionRollSuggestion,
    OptionsScorecard,
    OptionsStrikeCandidate,
)
from app.models.schwab_models import Position
from app.models.schwab_option_chain_models import OptionChain
from app.services.intelligence.options_scoring_service import OptionsScoringService


class OptionRollPlannerService:
    @staticmethod
    def build_roll_suggestions(
        *,
        positions: list[Position],
        symbol: str,
        option_chain: OptionChain | None,
        scorecard: OptionsScorecard | None,
    ) -> list[OptionRollSuggestion]:
        if option_chain is None or scorecard is None:
            return []

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

            strike = instrument.strikePrice
            expiration = instrument.expirationDate
            put_call = instrument.putCall or parse_put_call_from_option_symbol(
                instrument.symbol or ""
            )
            if strike is None or not expiration or not put_call:
                continue

            side = "call" if put_call == "CALL" else "put"
            expiration_date = position_expiration_date(position)
            current_contract = None
            if expiration_date is not None:
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
            )
            if alternative is None:
                continue

            current_bid = quoted_bid(current_contract)
            current_ask = quoted_ask(current_contract)
            estimated_credit = OptionRollPlannerService._estimate_roll_credit(
                current_contract_bid=current_bid,
                alternative_ask=alternative.ask,
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
            ),
            reverse=True,
        )
        best_score = OptionRollPlannerService._roll_fit_score(
            candidate=ranked[0],
            side=side,
            current_strike=current_strike,
            current_exp=current_exp,
            current_abs_delta=current_abs_delta,
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
    ) -> float:
        if candidate.side != side:
            return -1.0
        if (
            abs(candidate.strike - current_strike) < 0.01
            and candidate.expiration[:10] == current_exp
        ):
            return -1.0

        score = candidate.score
        candidate_dte = OptionsScoringService._expiration_dte(candidate.expiration)
        score += OptionsScoringService._dte_score(candidate_dte) * 0.25

        if candidate.expiration[:10] > current_exp:
            score += 0.15

        candidate_delta = sanitize_delta(candidate.delta)
        if current_abs_delta is not None and candidate_delta is not None:
            candidate_abs_delta = abs(candidate_delta)
            if current_abs_delta >= OptionsScoringService.ASSIGNMENT_DELTA_THRESHOLD:
                if candidate_abs_delta < current_abs_delta:
                    score += 0.25
                else:
                    score -= 0.15
            if (
                OptionsScoringService.TARGET_DELTA_MIN
                <= candidate_abs_delta
                <= OptionsScoringService.TARGET_DELTA_MAX
            ):
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
        if estimated_credit is not None:
            rationale += f"; estimated net credit ~${estimated_credit:.2f}/contract"
        return rationale

    @staticmethod
    def _estimate_roll_credit(
        *,
        current_contract_bid: float | None,
        alternative_ask: float | None,
    ) -> float | None:
        if current_contract_bid is None or alternative_ask is None:
            return None
        credit = current_contract_bid - alternative_ask
        return round(max(credit, 0.0), 2)
