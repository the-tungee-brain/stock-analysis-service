from __future__ import annotations

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
            put_call = instrument.putCall
            if strike is None or not expiration or not put_call:
                continue

            side = "call" if put_call == "CALL" else "put"
            exp_map = (
                option_chain.callExpDateMap
                if side == "call"
                else option_chain.putExpDateMap
            )

            current_contract = None
            for contracts_by_strike in exp_map.values():
                match = OptionsScoringService._find_contract(
                    contracts_by_strike, strike
                )
                if match is not None:
                    current_contract = match
                    break

            candidates = (
                scorecard.covered_call_candidates
                if side == "call"
                else scorecard.csp_candidates
            )
            alternative = OptionRollPlannerService._pick_alternative(
                side=side,
                current_strike=strike,
                current_expiration=expiration,
                candidates=candidates,
            )
            if alternative is None:
                continue

            estimated_credit = OptionRollPlannerService._estimate_roll_credit(
                current_contract_bid=getattr(current_contract, "bidPrice", None),
                alternative_ask=alternative.ask,
            )

            rationale = (
                f"Roll short {side} from ${strike:g} ({expiration[:10]}) to "
                f"${alternative.strike:g} ({alternative.expiration[:10]}) "
                f"for better delta/OI profile"
            )
            if estimated_credit is not None:
                rationale += f"; estimated net credit ~${estimated_credit:.2f}/contract"

            suggestions.append(
                OptionRollSuggestion(
                    side=side,
                    current_strike=strike,
                    current_expiration=expiration,
                    suggested_strike=alternative.strike,
                    suggested_expiration=alternative.expiration,
                    current_delta=getattr(current_contract, "delta", None),
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
        candidates: list[OptionsStrikeCandidate],
    ) -> OptionsStrikeCandidate | None:
        for candidate in candidates:
            if candidate.side != side:
                continue
            if abs(candidate.strike - current_strike) < 0.01:
                continue
            if candidate.expiration[:10] == current_expiration[:10]:
                continue
            return candidate
        return candidates[0] if candidates else None

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
