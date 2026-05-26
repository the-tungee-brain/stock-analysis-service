from __future__ import annotations

from datetime import date, datetime, timezone

from app.broker.option_delta_preference import (
    DEFAULT_DELTA_BAND,
    OptionDeltaBand,
    OptionStrategyPreferences,
    assignment_delta_threshold,
    delta_band_distance,
    delta_in_band,
    resolve_option_delta_band,
    resolve_option_strategy_preferences,
)
from app.broker.option_greeks import sanitize_delta
from app.models.strategy_models import UserInvestmentProfile
from app.broker.option_chain_table import (
    fair_option_price,
    quoted_ask,
    quoted_bid,
    quoted_last,
)
from app.models.intelligence_models import OptionsScorecard, OptionsStrikeCandidate
from app.models.schwab_option_chain_models import OptionChain, OptionContract


class OptionsScoringService:
    TARGET_DELTA_MIN = DEFAULT_DELTA_BAND.min_delta
    TARGET_DELTA_MAX = DEFAULT_DELTA_BAND.max_delta
    MIN_OPEN_INTEREST = 100
    ASSIGNMENT_DELTA_THRESHOLD = assignment_delta_threshold(DEFAULT_DELTA_BAND)
    TARGET_DTE_MIN = 5
    TARGET_DTE_MAX = 14
    MIN_SCORECARD_DTE = 1
    MAX_SCORECARD_DTE = 45
    MAX_CANDIDATES_PER_SIDE = 3

    @staticmethod
    def build_scorecard(
        chain: OptionChain,
        *,
        short_call_strikes: list[float] | None = None,
        short_put_strikes: list[float] | None = None,
        profile: UserInvestmentProfile | None = None,
        delta_band: OptionDeltaBand | None = None,
        strategy_preferences: OptionStrategyPreferences | None = None,
    ) -> OptionsScorecard | None:
        prefs = strategy_preferences or resolve_option_strategy_preferences(profile)
        band = delta_band or prefs.delta_band
        assignment_threshold = assignment_delta_threshold(band)
        if not chain.callExpDateMap and not chain.putExpDateMap:
            return None

        underlying_price = chain.underlyingPrice or (
            chain.underlying.last if chain.underlying else None
        )

        def parse_exp_key(key: str) -> datetime:
            return datetime.fromisoformat(key.split(":")[0])

        all_exp_keys = sorted(
            set(chain.callExpDateMap.keys()) | set(chain.putExpDateMap.keys()),
            key=parse_exp_key,
        )
        if not all_exp_keys:
            return None

        covered_calls: list[OptionsStrikeCandidate] = []
        csps: list[OptionsStrikeCandidate] = []

        for exp_key in all_exp_keys:
            covered_calls.extend(
                OptionsScoringService._rank_candidates(
                    side="call",
                    contracts_by_strike=chain.callExpDateMap.get(exp_key, {}),
                    underlying_price=underlying_price,
                    rationale_prefix="Covered call candidate",
                    expiration_key=exp_key,
                    strategy_preferences=prefs,
                )
            )
            csps.extend(
                OptionsScoringService._rank_candidates(
                    side="put",
                    contracts_by_strike=chain.putExpDateMap.get(exp_key, {}),
                    underlying_price=underlying_price,
                    rationale_prefix="Cash-secured put candidate",
                    expiration_key=exp_key,
                    strategy_preferences=prefs,
                )
            )

        covered_calls = OptionsScoringService._select_top_strategy_candidates(
            covered_calls,
            band=band,
            limit=OptionsScoringService.MAX_CANDIDATES_PER_SIDE,
        )
        csps = OptionsScoringService._select_top_strategy_candidates(
            csps,
            band=band,
            limit=OptionsScoringService.MAX_CANDIDATES_PER_SIDE,
        )

        assignment_flags: list[str] = []
        if short_call_strikes and underlying_price:
            for strike in short_call_strikes:
                call = OptionsScoringService._find_contract_in_chain(
                    chain, side="call", strike=strike
                )
                delta = sanitize_delta(call.delta if call else None)
                if delta is not None and delta >= assignment_threshold:
                    assignment_flags.append(
                        f"Short call at ${strike:g} has delta {delta:.2f} — "
                        "elevated assignment risk."
                    )
        if short_put_strikes and underlying_price:
            for strike in short_put_strikes:
                put = OptionsScoringService._find_contract_in_chain(
                    chain, side="put", strike=strike
                )
                delta = sanitize_delta(put.delta if put else None)
                if (
                    delta is not None
                    and abs(delta) >= assignment_threshold
                ):
                    assignment_flags.append(
                        f"Short put at ${strike:g} has delta {delta:.2f} — "
                        "elevated assignment risk."
                    )

        return OptionsScorecard(
            underlying_price=underlying_price,
            covered_call_candidates=covered_calls,
            csp_candidates=csps,
            assignment_flags=assignment_flags,
        )

    @staticmethod
    def _select_top_strategy_candidates(
        candidates: list[OptionsStrikeCandidate],
        *,
        band: OptionDeltaBand,
        limit: int,
    ) -> list[OptionsStrikeCandidate]:
        if not candidates or limit <= 0:
            return []

        in_band = [c for c in candidates if delta_in_band(c.delta, band)]
        in_band.sort(key=lambda item: item.score, reverse=True)
        if len(in_band) >= limit:
            return in_band[:limit]

        selected = list(in_band)
        remaining = [c for c in candidates if c not in selected]
        remaining.sort(
            key=lambda item: (delta_band_distance(item.delta, band), -item.score),
        )
        for candidate in remaining:
            if len(selected) >= limit:
                break
            selected.append(candidate)

        selected.sort(key=lambda item: item.score, reverse=True)
        return selected[:limit]

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
    def _find_contract_in_chain(
        chain: OptionChain,
        *,
        side: str,
        strike: float,
    ) -> OptionContract | None:
        exp_map = (
            chain.callExpDateMap if side == "call" else chain.putExpDateMap
        )
        for exp_key in sorted(exp_map.keys()):
            match = OptionsScoringService._find_contract(exp_map[exp_key], strike)
            if match is not None:
                return match
        return None

    @staticmethod
    def _rank_candidates(
        *,
        side: str,
        contracts_by_strike: dict[str, list[OptionContract]],
        underlying_price: float | None,
        rationale_prefix: str,
        expiration_key: str | None = None,
        strategy_preferences: OptionStrategyPreferences | None = None,
    ) -> list[OptionsStrikeCandidate]:
        prefs = strategy_preferences or OptionStrategyPreferences(
            delta_band=DEFAULT_DELTA_BAND
        )
        band = prefs.delta_band
        preferred_dte = prefs.preferred_dte_days
        candidates: list[OptionsStrikeCandidate] = []

        for strike_str, contract_list in contracts_by_strike.items():
            if not contract_list:
                continue
            contract = contract_list[0]
            try:
                strike = float(strike_str)
            except ValueError:
                continue

            delta = sanitize_delta(contract.delta)
            if delta is None:
                continue

            abs_delta = abs(delta)
            if abs_delta < 0.05 or abs_delta > 0.55:
                continue

            oi = contract.openInterest or 0
            if oi < OptionsScoringService.MIN_OPEN_INTEREST:
                continue

            days_to_expiration = OptionsScoringService._contract_dte(
                contract, expiration_key=expiration_key
            )
            if (
                days_to_expiration < OptionsScoringService.MIN_SCORECARD_DTE
                or days_to_expiration > OptionsScoringService.MAX_SCORECARD_DTE
            ):
                continue

            delta_score = OptionsScoringService._delta_score(abs_delta, band=band)
            oi_score = min(oi / 1000.0, 1.0)
            spread = OptionsScoringService._spread_pct(contract)
            spread_score = max(0.0, 1.0 - (spread or 0.0) / 0.15)
            dte_score = OptionsScoringService._dte_score(
                days_to_expiration,
                preferred_dte=preferred_dte,
            )

            score = (
                delta_score * 0.40
                + oi_score * 0.25
                + spread_score * 0.15
                + dte_score * 0.20
            )

            moneyness = ""
            if underlying_price:
                pct_from_spot = ((strike / underlying_price) - 1.0) * 100.0
                if side == "call":
                    if pct_from_spot > 0.5:
                        moneyness = f"{pct_from_spot:.1f}% above stock price"
                    elif pct_from_spot < -0.5:
                        moneyness = f"{abs(pct_from_spot):.1f}% below stock price"
                    else:
                        moneyness = "at the money"
                else:
                    if pct_from_spot < -0.5:
                        moneyness = f"{abs(pct_from_spot):.1f}% below stock price"
                    elif pct_from_spot > 0.5:
                        moneyness = f"{pct_from_spot:.1f}% above stock price"
                    else:
                        moneyness = "at the money"

            summary_parts = [
                f"{days_to_expiration} days to expiry",
                f"delta {abs_delta:.2f}",
                f"{oi:,} open interest",
            ]
            if moneyness:
                summary_parts.append(moneyness)

            candidates.append(
                OptionsStrikeCandidate(
                    side=side,
                    strike=strike,
                    expiration=contract.expirationDate,
                    delta=delta,
                    open_interest=oi,
                    bid=quoted_bid(contract),
                    ask=quoted_ask(contract),
                    last_price=quoted_last(contract),
                    mark=fair_option_price(contract),
                    theta=contract.theta,
                    iv=contract.volatility,
                    score=round(score, 3),
                    rationale=" · ".join(summary_parts),
                )
            )

        return candidates

    @staticmethod
    def _contract_dte(
        contract: OptionContract,
        *,
        expiration_key: str | None = None,
    ) -> int:
        if contract.daysToExpiration is not None:
            return max(int(contract.daysToExpiration), 0)
        if expiration_key and ":" in expiration_key:
            try:
                return max(int(expiration_key.split(":")[1]), 0)
            except ValueError:
                pass
        return OptionsScoringService._expiration_dte(contract.expirationDate)

    @staticmethod
    def _expiration_dte(expiration: str) -> int:
        exp_date = date.fromisoformat(expiration[:10])
        today = datetime.now(timezone.utc).date()
        return max((exp_date - today).days, 0)

    @staticmethod
    def _delta_score(abs_delta: float, *, band: OptionDeltaBand) -> float:
        target_mid = (band.min_delta + band.max_delta) / 2.0
        half_width = max((band.max_delta - band.min_delta) / 2.0, 0.05)
        distance = abs(abs_delta - target_mid)
        return max(0.0, 1.0 - distance / half_width)

    @staticmethod
    def _dte_score(days_to_expiration: int, *, preferred_dte: int) -> float:
        distance = abs(days_to_expiration - preferred_dte)
        window = max(preferred_dte, 7)
        return max(0.0, 1.0 - distance / window)

    @staticmethod
    def _spread_pct(contract: OptionContract) -> float | None:
        bid = quoted_bid(contract)
        ask = quoted_ask(contract)
        if bid is None or ask is None:
            return None
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return None
        return (ask - bid) / mid
