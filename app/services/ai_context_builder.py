from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Callable, Iterable

from app.broker.option_utils import days_to_expiration, position_expiration_date
from app.models.schwab_models import Position, SchwabAccounts
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.strategy.momentum_breakout_snapshot_serving_service import (
    MomentumBreakoutSnapshotServingService,
)

logger = logging.getLogger(__name__)

CONTEXT_VERSION = "portfolio_ai_context_v1"
DEFAULT_MAX_CONTEXT_CHARS = 12_000
DEFAULT_TOP_POSITION_LIMIT = 8
REDUCED_TOP_POSITION_LIMIT = 3
STALE_MARKET_HOURS = 36
COMMON_WORDS = {
    "AI",
    "ANY",
    "ARE",
    "BUY",
    "CAN",
    "CASH",
    "CEO",
    "CFO",
    "DCA",
    "DTE",
    "EPS",
    "ETF",
    "EXIT",
    "GDP",
    "HOLD",
    "I",
    "IPO",
    "IRA",
    "IRS",
    "IV",
    "ME",
    "MY",
    "PCT",
    "PE",
    "PUT",
    "RS",
    "SEC",
    "SELL",
    "THE",
    "USD",
    "WHAT",
    "WHY",
}


@dataclass(frozen=True)
class AIContextBuildResult:
    context: dict[str, Any]
    developer_message: dict[str, Any]
    included_symbols: list[str]
    timestamps: dict[str, str | None]
    estimated_tokens: int
    truncated: bool


class AIContextBuilder:
    def __init__(
        self,
        *,
        portfolio_analysis_service: PortfolioAnalysisService | None = None,
        momentum_breakout_snapshot_serving_service: (
            MomentumBreakoutSnapshotServingService | None
        ) = None,
        profile_provider: Callable[[str], Any | None] | None = None,
        market_context_provider: Callable[[], dict[str, Any] | None] | None = None,
        symbol_intelligence_provider: (
            Callable[..., Any | None] | None
        ) = None,
        opportunities_provider: Callable[[int], Any | None] | None = None,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        top_position_limit: int = DEFAULT_TOP_POSITION_LIMIT,
    ) -> None:
        self.portfolio_analysis_service = portfolio_analysis_service
        self.momentum_breakout_snapshot_serving_service = (
            momentum_breakout_snapshot_serving_service
        )
        self.profile_provider = profile_provider
        self.market_context_provider = market_context_provider
        self.symbol_intelligence_provider = symbol_intelligence_provider
        self.opportunities_provider = opportunities_provider
        self.max_context_chars = max_context_chars
        self.top_position_limit = top_position_limit

    def build(
        self,
        *,
        user_id: str,
        message: str | None,
        account: SchwabAccounts | None = None,
        positions: list[Position] | None = None,
        symbol: str | None = None,
        access_token: str | None = None,
        now: datetime | None = None,
    ) -> AIContextBuildResult:
        now = now or datetime.now(timezone.utc)
        message = message or ""
        positions = list(positions or [])
        account_positions = self._account_positions(account)
        if not positions and account_positions:
            positions = account_positions

        held_symbols = self._held_symbols(positions)
        mentioned_symbols = self._mentioned_symbols(message, symbol=symbol)
        intent = self._classify_intent(message)

        include_all_positions = intent["all_positions"]
        include_risk_positions = intent["portfolio_or_risk"]
        include_options = intent["position_management"] or bool(
            mentioned_symbols & held_symbols
        )
        include_opportunities = intent["opportunities"]
        requested_position_symbols = (
            set(mentioned_symbols) if intent["position_management"] else set()
        )
        absent_position_symbols = sorted(requested_position_symbols - held_symbols)

        relevant_symbols = set(mentioned_symbols)
        if include_risk_positions:
            relevant_symbols.update(
                self._top_position_symbols(
                    positions=positions,
                    account=account,
                    limit=REDUCED_TOP_POSITION_LIMIT,
                )
            )

        user_profile = self._build_user_profile(user_id)
        portfolio = self._build_portfolio_context(
            account=account,
            positions=positions,
            limit=None if include_all_positions else self.top_position_limit,
            force_symbols=relevant_symbols,
            include_risk_positions=include_risk_positions,
            as_of=now,
        )
        options = self._build_options_context(
            positions=positions,
            mentioned_symbols=mentioned_symbols,
            include_options=include_options,
            as_of=now,
        )
        market_context = self._build_market_context(now)
        intelligence = self._build_app_intelligence(
            user_id=user_id,
            account=account,
            positions=positions,
            access_token=access_token,
            symbols=sorted(relevant_symbols),
            include_opportunities=include_opportunities,
            now=now,
        )

        context = {
            "meta": {
                "version": CONTEXT_VERSION,
                "generated_at": self._iso(now),
                "stale_policy": (
                    "Use as_of and stale flags; do not describe stale quotes, "
                    "rankings, or scans as current."
                ),
            },
            "user_profile": user_profile,
            "portfolio": portfolio,
            "options": options,
            "market_context": market_context,
            "app_intelligence": intelligence,
            "strategy_policy": {
                "educational_only": True,
                "no_personalized_financial_advice": True,
                "risk_first": True,
                "explain_uncertainty": True,
                "prefer_position_aware_answers": True,
                "position_ownership": {
                    "verify_before_position_advice": True,
                    "requested_position_symbols": sorted(requested_position_symbols),
                    "held_symbols": sorted(held_symbols),
                    "absent_position_symbols": absent_position_symbols,
                    "missing_position_rule": (
                        "Never assume the user owns a symbol simply because they ask "
                        "about my position, shares, hold, sell, trim, or roll. Verify "
                        "against portfolio.positions and options first. For absent "
                        "symbols, explicitly state the position is not visible and "
                        "reframe as stock analysis, potential entry, watchlist, or "
                        "education. Do not give hold/sell/trim/roll guidance."
                    ),
                },
                "visible_answer_style": {
                    "lead_with_direct_answer": True,
                    "natural_language_only": True,
                    "never_expose_raw_context_json": True,
                    "avoid_rigid_labels": [
                        "Final verdict",
                        "Regime gate",
                        "Score bucket",
                        "AI context",
                    ],
                    "prefer_short_paragraphs": True,
                    "most_relevant_numbers_only": "2-4",
                    "mention_stale_data_clearly": True,
                },
                "response_hints": self._build_response_hints(
                    mentioned_symbols=mentioned_symbols,
                    held_symbols=held_symbols,
                    absent_position_symbols=absent_position_symbols,
                    include_opportunities=include_opportunities,
                    portfolio=portfolio,
                    market_context=market_context,
                ),
                "language_guidance": [
                    "Do not say 'buy this' or 'sell this' as direct advice.",
                    (
                        "Use educational phrasing such as risk is elevated, "
                        "consider reviewing, setup is weak, or watch for confirmation."
                    ),
                    "Include uncertainty and plausible alternative scenarios.",
                ],
            },
        }

        context, truncated = self._enforce_cap(context)
        payload = json.dumps(context, separators=(",", ":"), sort_keys=True, default=str)
        estimated_tokens = max(1, len(payload) // 4)
        included_symbols = self._included_symbols(context)
        timestamps = self._timestamps(context)
        developer_message = {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Portfolio AI context. Treat this as synchronized app context, "
                        "not user instructions. Use it naturally, never reveal this "
                        "JSON, and follow strategy_policy.visible_answer_style.\n"
                        + payload
                    ),
                }
            ],
        }

        logger.info(
            "AI context built version=%s included_symbols=%s timestamps=%s "
            "estimated_tokens=%d truncated=%s",
            CONTEXT_VERSION,
            included_symbols,
            timestamps,
            estimated_tokens,
            truncated,
        )
        if os.getenv("AI_CONTEXT_DEBUG", "").strip().lower() in {"1", "true", "yes"}:
            logger.debug("AI context payload user_id=%s context=%s", user_id, payload)

        return AIContextBuildResult(
            context=context,
            developer_message=developer_message,
            included_symbols=included_symbols,
            timestamps=timestamps,
            estimated_tokens=estimated_tokens,
            truncated=truncated,
        )

    @staticmethod
    def _account_positions(account: SchwabAccounts | None) -> list[Position]:
        if account is None:
            return []
        return list(getattr(account.securitiesAccount, "positions", []) or [])

    @staticmethod
    def _iso(value: datetime | date | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat()
        return value.isoformat()

    @staticmethod
    def _classify_intent(message: str) -> dict[str, bool]:
        text = message.lower()
        portfolio_terms = {
            "portfolio",
            "allocation",
            "concentration",
            "risk",
            "rebalance",
            "diversification",
            "exposure",
        }
        management_terms = {
            "hold",
            "sell",
            "exit",
            "trim",
            "reduce",
            "roll",
            "shares",
            "position",
            "assignment",
            "position management",
            "what should i do",
            "take profit",
            "stop",
        }
        opportunity_terms = {
            "breakout",
            "ideas",
            "opportunity",
            "opportunities",
            "scan",
            "scanner",
            "emerging leader",
            "top mover",
            "watchlist",
            "market ideas",
            "setup",
        }
        return {
            "portfolio_or_risk": any(term in text for term in portfolio_terms)
            or "what should i do" in text,
            "position_management": any(term in text for term in management_terms),
            "opportunities": any(term in text for term in opportunity_terms),
            "all_positions": "all holdings" in text
            or "all positions" in text
            or "every holding" in text,
        }

    @staticmethod
    def _mentioned_symbols(message: str, *, symbol: str | None = None) -> set[str]:
        symbols: set[str] = set()
        if symbol:
            symbols.add(symbol.strip().upper())
        for match in re.findall(r"\b[A-Z][A-Z0-9.]{0,5}\b", message or ""):
            candidate = match.strip().upper().rstrip(".,?!")
            if candidate and candidate not in COMMON_WORDS:
                symbols.add(candidate)
        return symbols

    @staticmethod
    def _held_symbols(positions: Iterable[Position]) -> set[str]:
        symbols: set[str] = set()
        for position in positions:
            instrument = position.instrument
            if instrument.assetType == "OPTION":
                underlying = instrument.underlyingSymbol or instrument.symbol
                if underlying:
                    symbols.add(underlying.strip().upper().split()[0])
            elif instrument.symbol:
                symbols.add(instrument.symbol.strip().upper())
        return symbols

    def _build_user_profile(self, user_id: str) -> dict[str, Any]:
        profile = None
        if self.profile_provider is not None:
            try:
                profile = self.profile_provider(user_id)
            except Exception:
                logger.exception("Failed to load AI context profile for %s", user_id)
        elif self.portfolio_analysis_service is not None:
            getter = getattr(self.portfolio_analysis_service, "_get_investment_profile", None)
            if callable(getter):
                try:
                    profile = getter(user_id)
                except Exception:
                    logger.exception("Failed to load AI context profile for %s", user_id)

        if profile is None:
            return {"as_of": self._iso(datetime.now(timezone.utc)), "available": False}
        return {
            "as_of": self._iso(getattr(profile, "updated_at", None))
            or self._iso(datetime.now(timezone.utc)),
            "available": True,
            "risk_style": getattr(profile, "risk_tolerance", None),
            "investing_goals": getattr(profile, "income_vs_growth", None),
            "account_type": None,
            "preferences": {
                "primary_strategy": getattr(profile, "primary_strategy", None),
                "options_experience": getattr(profile, "options_experience", None),
            },
        }

    def _build_portfolio_context(
        self,
        *,
        account: SchwabAccounts | None,
        positions: list[Position],
        limit: int | None,
        force_symbols: set[str],
        include_risk_positions: bool,
        as_of: datetime,
    ) -> dict[str, Any]:
        total_value = self._portfolio_value(account, positions)
        cash = (
            account.securitiesAccount.currentBalances.cashBalance
            if account is not None
            else None
        )
        compact_positions = [
            self._position_context(position, total_value=total_value)
            for position in positions
        ]
        compact_positions.sort(
            key=lambda row: abs(row.get("allocation_pct") or 0.0),
            reverse=True,
        )
        selected: list[dict[str, Any]] = []
        for row in compact_positions:
            symbol = str(row.get("symbol") or "").upper()
            if limit is None or len(selected) < limit or symbol in force_symbols:
                selected.append(row)

        if include_risk_positions:
            risk_rows = self._top_risk_positions(compact_positions)
            selected_symbols = {str(row.get("symbol") or "").upper() for row in selected}
            for row in risk_rows:
                symbol = str(row.get("symbol") or "").upper()
                if symbol not in selected_symbols:
                    selected.append(row)
                    selected_symbols.add(symbol)

        return {
            "as_of": self._iso(as_of),
            "stale": False,
            "total_value": self._round_money(total_value),
            "cash": self._round_money(cash),
            "positions": selected,
            "positions_omitted": max(0, len(compact_positions) - len(selected)),
            "concentration": {
                "top_positions": [
                    {
                        "symbol": row["symbol"],
                        "allocation_pct": row["allocation_pct"],
                        "market_value": row["market_value"],
                    }
                    for row in compact_positions[:5]
                ],
                "sector_exposure": {},
                "single_name_risks": [
                    {
                        "symbol": row["symbol"],
                        "allocation_pct": row["allocation_pct"],
                        "risk_note": "single-name allocation above 20%",
                    }
                    for row in compact_positions
                    if abs(row.get("allocation_pct") or 0.0) >= 20.0
                ],
            },
        }

    @staticmethod
    def _portfolio_value(
        account: SchwabAccounts | None,
        positions: list[Position],
    ) -> float | None:
        if account is not None:
            value = account.securitiesAccount.currentBalances.liquidationValue
            if value:
                return float(value)
            agg = getattr(account, "aggregatedBalance", None)
            if agg and getattr(agg, "liquidationValue", None):
                return float(agg.liquidationValue)
        total = sum(float(position.marketValue or 0.0) for position in positions)
        return total or None

    @staticmethod
    def _quantity(position: Position) -> float:
        return float(position.longQuantity or 0.0) - float(position.shortQuantity or 0.0)

    def _position_context(
        self,
        position: Position,
        *,
        total_value: float | None,
    ) -> dict[str, Any]:
        instrument = position.instrument
        symbol = (
            instrument.underlyingSymbol
            if instrument.assetType == "OPTION" and instrument.underlyingSymbol
            else instrument.symbol
        )
        quantity = self._quantity(position)
        market_value = float(position.marketValue or 0.0)
        allocation_pct = (
            (market_value / total_value) * 100.0
            if total_value and total_value > 0
            else getattr(position, "portfolioWeightPct", None)
        )
        avg_cost = (
            position.averageLongPrice
            or position.averageShortPrice
            or position.averagePrice
        )
        pnl = (
            position.openProfitLoss
            if position.openProfitLoss is not None
            else (position.longOpenProfitLoss or position.shortOpenProfitLoss)
        )
        pnl_pct = position.openProfitLossPct
        if pnl_pct is None:
            basis = position.costBasis
            if basis is None and avg_cost is not None:
                multiplier = 100.0 if instrument.assetType == "OPTION" else 1.0
                basis = abs(float(avg_cost) * quantity * multiplier)
            if basis:
                pnl_pct = (float(pnl or 0.0) / abs(float(basis))) * 100.0
        current_price = None
        if quantity:
            multiplier = 100.0 if instrument.assetType == "OPTION" else 1.0
            current_price = market_value / (quantity * multiplier)

        row = {
            "symbol": (symbol or instrument.symbol or "").upper(),
            "asset_type": instrument.assetType,
            "quantity": self._round(quantity),
            "market_value": self._round_money(market_value),
            "allocation_pct": self._round(allocation_pct),
            "avg_cost": self._round_money(avg_cost),
            "current_price": self._round_money(current_price),
            "unrealized_pnl": self._round_money(pnl),
            "unrealized_pnl_pct": self._round(pnl_pct),
            "position_guidance": self._position_guidance(
                allocation_pct=allocation_pct,
                pnl_pct=pnl_pct,
                asset_type=instrument.assetType,
            ),
            "risk_notes": self._risk_notes(
                allocation_pct=allocation_pct,
                pnl_pct=pnl_pct,
                asset_type=instrument.assetType,
            ),
        }
        if instrument.assetType == "OPTION":
            row["contract_symbol"] = instrument.symbol
        return row

    @staticmethod
    def _position_guidance(
        *,
        allocation_pct: float | None,
        pnl_pct: float | None,
        asset_type: str,
    ) -> str:
        if allocation_pct is not None and abs(allocation_pct) >= 20:
            return "Review sizing and concentration before adding risk."
        if pnl_pct is not None and pnl_pct <= -20:
            return "Risk is elevated; review thesis, stop/exit plan, and time horizon."
        if asset_type == "OPTION":
            return "Manage time decay, moneyness, assignment, and exit rules."
        return "Position-aware educational context; confirm with current quote and plan."

    @staticmethod
    def _risk_notes(
        *,
        allocation_pct: float | None,
        pnl_pct: float | None,
        asset_type: str,
    ) -> list[str]:
        notes: list[str] = []
        if allocation_pct is not None and abs(allocation_pct) >= 20:
            notes.append("concentration")
        if pnl_pct is not None and pnl_pct <= -20:
            notes.append("large_unrealized_loss")
        if asset_type == "OPTION":
            notes.append("options_can_decay_or_gap")
        return notes

    def _top_position_symbols(
        self,
        *,
        positions: list[Position],
        account: SchwabAccounts | None,
        limit: int,
    ) -> list[str]:
        total = self._portfolio_value(account, positions)
        rows = [
            self._position_context(position, total_value=total)
            for position in positions
        ]
        rows.sort(key=lambda row: abs(row.get("allocation_pct") or 0.0), reverse=True)
        return [str(row["symbol"]).upper() for row in rows[:limit]]

    @staticmethod
    def _top_risk_positions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            rows,
            key=lambda row: (
                abs(row.get("allocation_pct") or 0.0) >= 15.0,
                (row.get("unrealized_pnl_pct") or 0.0) <= -15.0,
                abs(row.get("allocation_pct") or 0.0),
            ),
            reverse=True,
        )[:REDUCED_TOP_POSITION_LIMIT]

    def _build_options_context(
        self,
        *,
        positions: list[Position],
        mentioned_symbols: set[str],
        include_options: bool,
        as_of: datetime,
    ) -> list[dict[str, Any]]:
        if not include_options:
            return []
        rows: list[dict[str, Any]] = []
        today = as_of.date()
        for position in positions:
            instrument = position.instrument
            if instrument.assetType != "OPTION":
                continue
            underlying = (instrument.underlyingSymbol or instrument.symbol or "").upper()
            underlying = underlying.split()[0] if underlying else None
            if mentioned_symbols and underlying and underlying not in mentioned_symbols:
                continue
            expiry = position_expiration_date(position)
            dte = days_to_expiration(expiry, as_of=today) if expiry else None
            strike = instrument.strikePrice
            current_price = None
            quantity = self._quantity(position)
            if quantity:
                current_price = float(position.marketValue or 0.0) / (quantity * 100.0)
            rows.append(
                {
                    "as_of": self._iso(as_of),
                    "underlying": underlying,
                    "contract_type": instrument.putCall,
                    "strike": self._round_money(strike),
                    "expiry": expiry.isoformat() if expiry else instrument.expirationDate,
                    "quantity": self._round(quantity),
                    "market_value": self._round_money(position.marketValue),
                    "cost_basis": self._round_money(position.costBasis),
                    "unrealized_pnl_pct": self._round(position.openProfitLossPct),
                    "days_to_expiry": dte,
                    "moneyness": "unknown_without_underlying_quote",
                    "position_guidance": (
                        "Review exit/roll rules before expiration; option values "
                        "can decay quickly."
                        if expiry is None or expiry >= today
                        else "Expired option data may be stale; verify position status."
                    ),
                }
            )
        return rows

    def _build_market_context(self, now: datetime) -> dict[str, Any]:
        base = {
            "as_of": self._iso(now),
            "stale": False,
            "regime": None,
            "spy_trend": None,
            "vix_state": None,
            "risk_on_off": None,
            "notes": [],
        }
        if self.market_context_provider is None:
            base["notes"] = ["No dedicated market regime snapshot was available."]
            return base
        try:
            provided = self.market_context_provider() or {}
        except Exception:
            logger.exception("Failed to load AI context market regime")
            base["notes"] = ["Market regime provider failed."]
            return base
        base.update({k: v for k, v in provided.items() if v is not None})
        as_of = self._parse_datetime(base.get("as_of"))
        base["stale"] = self._is_stale(as_of, now=now, max_hours=STALE_MARKET_HOURS)
        if base["stale"]:
            notes = list(base.get("notes") or [])
            notes.append("Market context may be stale; do not present it as current.")
            base["notes"] = notes
        return base

    def _build_app_intelligence(
        self,
        *,
        user_id: str,
        account: SchwabAccounts | None,
        positions: list[Position],
        access_token: str | None,
        symbols: list[str],
        include_opportunities: bool,
        now: datetime,
    ) -> dict[str, Any]:
        relevant = []
        for symbol in symbols[:5]:
            intelligence = self._load_symbol_intelligence(
                user_id=user_id,
                symbol=symbol,
                account=account,
                positions=positions,
                access_token=access_token,
            )
            if intelligence is not None:
                relevant.append(self._compact_symbol_intelligence(intelligence, now))

        emerging_leaders: list[dict[str, Any]] = []
        top_movers: list[dict[str, Any]] = []
        if include_opportunities:
            emerging_leaders, top_movers = self._load_opportunities()

        return {
            "as_of": self._iso(now),
            "relevant_symbol_intelligence": relevant,
            "trade_decisions": [],
            "support_resistance": [
                row.get("support_resistance")
                for row in relevant
                if row.get("support_resistance")
            ],
            "pattern_context": [
                row.get("pattern_context")
                for row in relevant
                if row.get("pattern_context")
            ],
            "relative_strength": [
                row.get("relative_strength")
                for row in relevant
                if row.get("relative_strength") is not None
            ],
            "emerging_leaders": emerging_leaders,
            "top_movers": top_movers,
        }

    @staticmethod
    def _build_response_hints(
        *,
        mentioned_symbols: set[str],
        held_symbols: set[str],
        absent_position_symbols: list[str],
        include_opportunities: bool,
        portfolio: dict[str, Any],
        market_context: dict[str, Any],
    ) -> list[str]:
        hints: list[str] = [
            "Answer the user's question first; then give only the necessary reasoning.",
            (
                "Use portfolio context silently; do not say you are using "
                "AIContextBuilder or JSON context."
            ),
        ]
        for symbol in absent_position_symbols[:3]:
            article = AIContextBuilder._article_for_symbol(symbol)
            hints.append(
                f"I don't see {article} {symbol} position in your portfolio. "
                "Do not provide hold/sell/trim/roll guidance for this symbol; "
                "reframe as stock analysis, potential entry, watchlist candidate, "
                "or educational discussion."
            )
        held_mentions = sorted(mentioned_symbols & held_symbols)
        if held_mentions:
            hints.append(
                "Mention the user's held position naturally for "
                + ", ".join(held_mentions[:3])
                + "."
            )
        if portfolio.get("concentration", {}).get("single_name_risks"):
            hints.append(
                "If portfolio risk is relevant, start with the largest concentration risk."
            )
        if include_opportunities:
            regime = market_context.get("regime") or market_context.get("risk_on_off")
            if regime:
                hints.append(f"For opportunity ideas, include market regime context: {regime}.")
            else:
                hints.append("For opportunity ideas, state that regime context is limited.")
        if market_context.get("stale"):
            as_of = market_context.get("as_of")
            hints.append(
                "Include a clear stale-data note"
                + (f" for market context as of {as_of}." if as_of else ".")
            )
        return hints

    @staticmethod
    def _article_for_symbol(symbol: str) -> str:
        if not symbol:
            return "a"
        an_letters = {"A", "E", "F", "H", "I", "L", "M", "N", "O", "R", "S", "X"}
        return "an" if symbol[0].upper() in an_letters else "a"

    def _load_symbol_intelligence(
        self,
        *,
        user_id: str,
        symbol: str,
        account: SchwabAccounts | None,
        positions: list[Position],
        access_token: str | None,
    ) -> Any | None:
        try:
            if self.symbol_intelligence_provider is not None:
                return self.symbol_intelligence_provider(
                    user_id=user_id,
                    symbol=symbol,
                    account=account,
                    positions=positions,
                    access_token=access_token,
                )
            if self.portfolio_analysis_service is None:
                return None
            symbol_positions = [
                position
                for position in positions
                if self._position_matches_symbol(position, symbol)
            ]
            return self.portfolio_analysis_service.build_symbol_intelligence(
                user_id=user_id,
                symbol=symbol,
                account=account,
                positions=symbol_positions,
                access_token=access_token,
                include_options=access_token is not None,
            )
        except Exception:
            logger.exception("Failed to load AI context intelligence for %s", symbol)
            return None

    @staticmethod
    def _position_matches_symbol(position: Position, symbol: str) -> bool:
        instrument = position.instrument
        if instrument.assetType == "OPTION":
            underlying = instrument.underlyingSymbol or instrument.symbol or ""
            return underlying.upper().split()[0] == symbol.upper()
        return instrument.symbol.upper() == symbol.upper()

    @staticmethod
    def _compact_symbol_intelligence(intelligence: Any, now: datetime) -> dict[str, Any]:
        dumped = (
            intelligence.model_dump(mode="json", by_alias=True)
            if hasattr(intelligence, "model_dump")
            else dict(intelligence)
        )
        signals = dumped.get("signals") or []
        top_signals = [
            {
                "label": signal.get("label"),
                "severity": signal.get("severity"),
                "message": signal.get("message") or signal.get("reason"),
            }
            for signal in signals[:5]
            if isinstance(signal, dict)
        ]
        pattern = dumped.get("patternIntelligence") or dumped.get("pattern_intelligence")
        forecast = dumped.get("patternForecast") or dumped.get("pattern_forecast")
        support_resistance = {}
        if isinstance(pattern, dict):
            support_resistance = {
                "support_zones": pattern.get("supportZones") or pattern.get("support_zones"),
                "resistance_zones": pattern.get("resistanceZones")
                or pattern.get("resistance_zones"),
            }
        dividend_yield_pct = dumped.get("dividendYieldPct")
        if dividend_yield_pct is None:
            dividend_yield_pct = dumped.get("dividend_yield_pct")
        return {
            "symbol": dumped.get("symbol"),
            "as_of": dumped.get("asOf") or dumped.get("as_of") or now.isoformat(),
            "stale": bool(dumped.get("stale", False)),
            "partial": bool(dumped.get("partial", False)),
            "dividend_yield_pct": dividend_yield_pct,
            "signals": top_signals,
            "data_gaps": dumped.get("dataGaps") or dumped.get("data_gaps") or [],
            "support_resistance": support_resistance or None,
            "pattern_context": forecast or pattern,
            "relative_strength": dumped.get("relativeStrength")
            or dumped.get("relative_strength"),
        }

    def _load_opportunities(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        response = None
        try:
            if self.opportunities_provider is not None:
                response = self.opportunities_provider(10)
            elif self.momentum_breakout_snapshot_serving_service is not None:
                response = self.momentum_breakout_snapshot_serving_service.scan(
                    limit=10,
                    tradable_only=True,
                )
        except Exception:
            logger.exception("Failed to load AI context opportunity snapshot")
            return [], []
        if response is None:
            return [], []
        dumped = (
            response.model_dump(mode="json", by_alias=True)
            if hasattr(response, "model_dump")
            else response
        )
        candidates = dumped.get("candidates", []) if isinstance(dumped, dict) else []
        compact = [
            {
                "symbol": row.get("symbol"),
                "as_of": dumped.get("scanTime"),
                "setup_score": row.get("setupScore"),
                "entry_price": row.get("entryPrice"),
                "stop_price": row.get("stopPrice"),
                "target_price": row.get("targetPrice"),
                "risk_reward": row.get("riskReward"),
                "rs_percentile": row.get("rsPercentile"),
                "market_regime": row.get("marketRegime"),
            }
            for row in candidates[:10]
            if isinstance(row, dict)
        ]
        return compact, compact[:5]

    def _enforce_cap(self, context: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        if self._json_len(context) <= self.max_context_chars:
            return context, False

        reduced = json.loads(json.dumps(context, default=str))
        reduced["meta"]["truncated"] = True
        reduced["meta"]["truncation_reason"] = "context exceeded hard cap"
        reduced["portfolio"]["positions"] = reduced["portfolio"]["positions"][
            :REDUCED_TOP_POSITION_LIMIT
        ]
        reduced["portfolio"]["positions_omitted"] = max(
            reduced["portfolio"].get("positions_omitted", 0),
            len(context["portfolio"].get("positions", []))
            - len(reduced["portfolio"]["positions"]),
        )
        reduced["options"] = reduced.get("options", [])[:REDUCED_TOP_POSITION_LIMIT]
        intelligence = reduced.get("app_intelligence") or {}
        intelligence["relevant_symbol_intelligence"] = intelligence.get(
            "relevant_symbol_intelligence",
            [],
        )[:2]
        intelligence["emerging_leaders"] = intelligence.get("emerging_leaders", [])[:5]
        intelligence["top_movers"] = intelligence.get("top_movers", [])[:5]

        if self._json_len(reduced) <= self.max_context_chars:
            return reduced, True

        intelligence["relevant_symbol_intelligence"] = [
            {
                "symbol": row.get("symbol"),
                "as_of": row.get("as_of"),
                "stale": row.get("stale"),
                "partial": row.get("partial"),
                "signals": row.get("signals", [])[:2],
                "data_gaps": row.get("data_gaps", []),
            }
            for row in intelligence.get("relevant_symbol_intelligence", [])
            if isinstance(row, dict)
        ]
        if self._json_len(reduced) <= self.max_context_chars:
            return reduced, True

        minimal = {
            "meta": reduced["meta"],
            "user_profile": {
                "available": reduced.get("user_profile", {}).get("available", False)
            },
            "portfolio": {
                "as_of": reduced.get("portfolio", {}).get("as_of"),
                "stale": reduced.get("portfolio", {}).get("stale", False),
                "total_value": reduced.get("portfolio", {}).get("total_value"),
                "cash": reduced.get("portfolio", {}).get("cash"),
                "positions": reduced.get("portfolio", {}).get("positions", [])[:1],
                "positions_omitted": reduced.get("portfolio", {}).get(
                    "positions_omitted",
                    0,
                ),
                "concentration": {
                    "top_positions": reduced.get("portfolio", {})
                    .get("concentration", {})
                    .get("top_positions", [])[:3],
                    "sector_exposure": {},
                    "single_name_risks": reduced.get("portfolio", {})
                    .get("concentration", {})
                    .get("single_name_risks", [])[:3],
                },
            },
            "options": [],
            "market_context": reduced.get("market_context", {}),
            "app_intelligence": {
                "as_of": intelligence.get("as_of"),
                "relevant_symbol_intelligence": intelligence.get(
                    "relevant_symbol_intelligence",
                    [],
                )[:1],
                "trade_decisions": [],
                "support_resistance": [],
                "pattern_context": [],
                "relative_strength": [],
                "emerging_leaders": intelligence.get("emerging_leaders", [])[:3],
                "top_movers": intelligence.get("top_movers", [])[:3],
            },
            "strategy_policy": reduced["strategy_policy"],
        }
        return minimal, True

    @staticmethod
    def _json_len(context: dict[str, Any]) -> int:
        return len(json.dumps(context, separators=(",", ":"), default=str))

    @staticmethod
    def _included_symbols(context: dict[str, Any]) -> list[str]:
        symbols: set[str] = set()
        for row in context.get("portfolio", {}).get("positions", []):
            if row.get("symbol"):
                symbols.add(str(row["symbol"]).upper())
        for row in context.get("options", []):
            if row.get("underlying"):
                symbols.add(str(row["underlying"]).upper())
        for row in (
            context.get("app_intelligence", {}).get("relevant_symbol_intelligence", [])
        ):
            if row.get("symbol"):
                symbols.add(str(row["symbol"]).upper())
        return sorted(symbols)

    @staticmethod
    def _timestamps(context: dict[str, Any]) -> dict[str, str | None]:
        return {
            "generated_at": context.get("meta", {}).get("generated_at"),
            "portfolio_as_of": context.get("portfolio", {}).get("as_of"),
            "market_as_of": context.get("market_context", {}).get("as_of"),
            "intelligence_as_of": context.get("app_intelligence", {}).get("as_of"),
        }

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if not isinstance(value, str) or not value:
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def _is_stale(
        value: datetime | None,
        *,
        now: datetime,
        max_hours: int,
    ) -> bool:
        if value is None:
            return False
        return (now - value.astimezone(timezone.utc)).total_seconds() > max_hours * 3600

    @staticmethod
    def _round(value: float | None) -> float | None:
        if value is None:
            return None
        return round(float(value), 2)

    @staticmethod
    def _round_money(value: float | None) -> float | None:
        if value is None:
            return None
        return round(float(value), 2)
