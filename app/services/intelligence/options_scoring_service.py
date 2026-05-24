from __future__ import annotations

from datetime import datetime

from app.models.intelligence_models import OptionsScorecard, OptionsStrikeCandidate
from app.models.schwab_option_chain_models import OptionChain, OptionContract


class OptionsScoringService:
    TARGET_DELTA_MIN = 0.18
    TARGET_DELTA_MAX = 0.32
    MIN_OPEN_INTEREST = 100
    ASSIGNMENT_DELTA_THRESHOLD = 0.40

    @staticmethod
    def build_scorecard(
        chain: OptionChain,
        *,
        short_call_strikes: list[float] | None = None,
        short_put_strikes: list[float] | None = None,
    ) -> OptionsScorecard | None:
        if not chain.callExpDateMap and not chain.putExpDateMap:
            return None

        underlying_price = chain.underlyingPrice or (
            chain.underlying.last if chain.underlying else None
        )

        def parse_exp_key(key: str) -> datetime:
            return datetime.fromisoformat(key.split(":")[0])

        all_exp_keys = list(
            set(chain.callExpDateMap.keys()) | set(chain.putExpDateMap.keys())
        )
        if not all_exp_keys:
            return None

        all_exp_keys.sort(key=parse_exp_key)
        first_exp = all_exp_keys[0]

        calls = chain.callExpDateMap.get(first_exp, {})
        puts = chain.putExpDateMap.get(first_exp, {})

        covered_calls = OptionsScoringService._rank_candidates(
            side="call",
            contracts_by_strike=calls,
            underlying_price=underlying_price,
            rationale_prefix="Covered call candidate",
        )
        csps = OptionsScoringService._rank_candidates(
            side="put",
            contracts_by_strike=puts,
            underlying_price=underlying_price,
            rationale_prefix="Cash-secured put candidate",
        )

        assignment_flags: list[str] = []
        if short_call_strikes and underlying_price:
            for strike in short_call_strikes:
                call = OptionsScoringService._find_contract(calls, strike)
                if call and call.delta and call.delta >= OptionsScoringService.ASSIGNMENT_DELTA_THRESHOLD:
                    assignment_flags.append(
                        f"Short call at ${strike:g} has delta {call.delta:.2f} — "
                        "elevated assignment risk."
                    )
        if short_put_strikes and underlying_price:
            for strike in short_put_strikes:
                put = OptionsScoringService._find_contract(puts, strike)
                if put and put.delta and abs(put.delta) >= OptionsScoringService.ASSIGNMENT_DELTA_THRESHOLD:
                    assignment_flags.append(
                        f"Short put at ${strike:g} has delta {put.delta:.2f} — "
                        "elevated assignment risk."
                    )

        return OptionsScorecard(
            underlying_price=underlying_price,
            covered_call_candidates=covered_calls[:3],
            csp_candidates=csps[:3],
            assignment_flags=assignment_flags,
        )

    @staticmethod
    def _find_contract(
        contracts_by_strike: dict[str, list[OptionContract]], strike: float
    ) -> OptionContract | None:
        for strike_str, contracts in contracts_by_strike.items():
            try:
                if abs(float(strike_str) - strike) < 0.01:
                    return contracts[0] if contracts else None
            except ValueError:
                continue
        return None

    @staticmethod
    def _rank_candidates(
        *,
        side: str,
        contracts_by_strike: dict[str, list[OptionContract]],
        underlying_price: float | None,
        rationale_prefix: str,
    ) -> list[OptionsStrikeCandidate]:
        candidates: list[OptionsStrikeCandidate] = []

        for strike_str, contract_list in contracts_by_strike.items():
            if not contract_list:
                continue
            contract = contract_list[0]
            try:
                strike = float(strike_str)
            except ValueError:
                continue

            delta = contract.delta
            if delta is None:
                continue

            abs_delta = abs(delta)
            if abs_delta < 0.05 or abs_delta > 0.55:
                continue

            oi = contract.openInterest or 0
            if oi < OptionsScoringService.MIN_OPEN_INTEREST:
                continue

            delta_score = OptionsScoringService._delta_score(abs_delta)
            oi_score = min(oi / 1000.0, 1.0)
            spread = OptionsScoringService._spread_pct(contract)
            spread_score = max(0.0, 1.0 - (spread or 0.0) / 0.15)

            score = delta_score * 0.5 + oi_score * 0.3 + spread_score * 0.2

            moneyness = ""
            if underlying_price:
                pct_otm = ((strike / underlying_price) - 1.0) * 100.0
                if side == "call":
                    moneyness = f"{pct_otm:+.1f}% from spot"
                else:
                    moneyness = f"{-pct_otm:+.1f}% from spot"

            candidates.append(
                OptionsStrikeCandidate(
                    side=side,
                    strike=strike,
                    expiration=contract.expirationDate,
                    delta=delta,
                    open_interest=oi,
                    bid=contract.bidPrice,
                    ask=contract.askPrice,
                    iv=contract.volatility,
                    score=round(score, 3),
                    rationale=(
                        f"{rationale_prefix}: delta {delta:.2f}, OI {oi:,}"
                        + (f", {moneyness}" if moneyness else "")
                    ),
                )
            )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    @staticmethod
    def _delta_score(abs_delta: float) -> float:
        target_mid = (
            OptionsScoringService.TARGET_DELTA_MIN
            + OptionsScoringService.TARGET_DELTA_MAX
        ) / 2.0
        distance = abs(abs_delta - target_mid)
        return max(0.0, 1.0 - distance / 0.2)

    @staticmethod
    def _spread_pct(contract: OptionContract) -> float | None:
        bid = contract.bidPrice
        ask = contract.askPrice
        if bid is None or ask is None or ask <= 0:
            return None
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return None
        return (ask - bid) / mid
