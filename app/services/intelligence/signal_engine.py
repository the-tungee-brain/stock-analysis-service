from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from app.core.prompts import AnalysisAction
from app.models.company_research_models import ResearchContext, SecRatioTrendPoint
from app.models.intelligence_models import IntelligenceSignal
from app.models.schwab_models import Position, SchwabAccounts


def _parse_pct(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.strip().replace("%", "").replace("+", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_return_pct(value: str | None) -> float | None:
    return _parse_pct(value)


def _days_until(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        target = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    return (target - date.today()).days


class SignalEngine:
    @staticmethod
    def build_symbol_signals(
        *,
        research: ResearchContext,
        positions: list[Position],
        account: SchwabAccounts | None = None,
        symbol: str,
    ) -> list[IntelligenceSignal]:
        signals: list[IntelligenceSignal] = []
        symbol_upper = symbol.upper()

        signals.extend(
            SignalEngine._earnings_signals(research=research, symbol=symbol_upper)
        )
        signals.extend(
            SignalEngine._valuation_signals(research=research, symbol=symbol_upper)
        )
        signals.extend(
            SignalEngine._sec_trend_signals(research=research, symbol=symbol_upper)
        )
        signals.extend(
            SignalEngine._news_sentiment_signals(
                research=research, symbol=symbol_upper
            )
        )
        signals.extend(
            SignalEngine._position_signals(
                positions=positions,
                account=account,
                symbol=symbol_upper,
            )
        )
        signals.extend(
            SignalEngine._performance_signals(
                research=research, symbol=symbol_upper
            )
        )

        severity_rank = {"critical": 0, "warning": 1, "watch": 2, "info": 3}
        signals.sort(key=lambda s: severity_rank.get(s.severity, 99))
        return signals

    @staticmethod
    def build_portfolio_signals(
        *,
        positions: list[Position],
        account: SchwabAccounts,
        sector_weights: dict[str, float] | None = None,
    ) -> list[IntelligenceSignal]:
        signals: list[IntelligenceSignal] = []
        liquidation = account.securitiesAccount.currentBalances.liquidationValue
        if liquidation <= 0:
            return signals

        by_symbol: dict[str, float] = {}
        for position in positions:
            key = SignalEngine._position_symbol(position)
            if not key:
                continue
            by_symbol[key] = by_symbol.get(key, 0.0) + abs(position.marketValue)

        for sym, mv in sorted(by_symbol.items(), key=lambda item: item[1], reverse=True):
            weight = (mv / liquidation) * 100.0
            if weight >= 30:
                signals.append(
                    IntelligenceSignal(
                        kind="concentration",
                        severity="critical",
                        message=(
                            f"{sym} is {weight:.1f}% of portfolio — above the 30% "
                            "concentration limit; trim is strongly recommended."
                        ),
                        symbol=sym,
                    )
                )
            elif weight >= 20:
                signals.append(
                    IntelligenceSignal(
                        kind="concentration",
                        severity="warning",
                        message=(
                            f"{sym} is {weight:.1f}% of portfolio — elevated "
                            "concentration; consider trimming toward 15%."
                        ),
                        symbol=sym,
                    )
                )

        if sector_weights:
            for sector, weight in sorted(
                sector_weights.items(), key=lambda item: item[1], reverse=True
            ):
                if weight >= 40:
                    signals.append(
                        IntelligenceSignal(
                            kind="sector_concentration",
                            severity="warning",
                            message=(
                                f"{sector} sector is {weight:.1f}% of portfolio — "
                                "high sector concentration risk."
                            ),
                        )
                    )

        severity_rank = {"critical": 0, "warning": 1, "watch": 2, "info": 3}
        signals.sort(key=lambda s: severity_rank.get(s.severity, 99))
        return signals

    @staticmethod
    def _position_symbol(position: Position) -> str | None:
        if position.instrument.assetType == "OPTION":
            return (
                position.instrument.underlyingSymbol or position.instrument.symbol
            )
        return position.instrument.symbol

    @staticmethod
    def _position_signals(
        *,
        positions: list[Position],
        account: SchwabAccounts | None,
        symbol: str,
    ) -> list[IntelligenceSignal]:
        if account is None:
            return []

        signals: list[IntelligenceSignal] = []
        liquidation = account.securitiesAccount.currentBalances.liquidationValue
        if liquidation <= 0:
            return signals

        scoped = [
            p
            for p in positions
            if SignalEngine._position_symbol(p) == symbol
        ]
        if not scoped:
            return signals

        total_mv = sum(abs(p.marketValue) for p in scoped)
        weight = (total_mv / liquidation) * 100.0

        for position in scoped:
            cost = position.averageLongPrice or position.taxLotAverageLongPrice
            if position.longQuantity > 0 and cost and cost > 0:
                # Approximate using market value / quantity for current price
                qty = position.longQuantity
                current = abs(position.marketValue) / qty if qty else None
                if current:
                    pnl_pct = (current / cost - 1.0) * 100.0
                    if pnl_pct <= -30:
                        signals.append(
                            IntelligenceSignal(
                                kind="drawdown",
                                severity="critical",
                                message=(
                                    f"Unrealized loss ~{pnl_pct:.0f}% on {symbol} — "
                                    "urgent risk review recommended."
                                ),
                                symbol=symbol,
                            )
                        )
                    elif pnl_pct <= -20:
                        signals.append(
                            IntelligenceSignal(
                                kind="drawdown",
                                severity="warning",
                                message=(
                                    f"Unrealized loss ~{pnl_pct:.0f}% on {symbol} — "
                                    "thesis and sizing review recommended."
                                ),
                                symbol=symbol,
                            )
                        )

        if weight >= 30:
            signals.append(
                IntelligenceSignal(
                    kind="position_size",
                    severity="critical",
                    message=f"{symbol} position weight is {weight:.1f}% of portfolio.",
                    symbol=symbol,
                )
            )
        elif weight >= 20:
            signals.append(
                IntelligenceSignal(
                    kind="position_size",
                    severity="watch",
                    message=f"{symbol} position weight is {weight:.1f}% of portfolio.",
                    symbol=symbol,
                )
            )

        return signals

    @staticmethod
    def _earnings_signals(
        *, research: ResearchContext, symbol: str
    ) -> list[IntelligenceSignal]:
        earnings = research.earnings
        if earnings is None or not earnings.upcoming_report_date:
            return []

        days = _days_until(earnings.upcoming_report_date)
        if days is None:
            return []

        period = earnings.upcoming_fiscal_period or "upcoming quarter"
        timing = earnings.upcoming_timing or "unknown timing"

        if 0 <= days <= 3:
            severity = "warning" if days <= 1 else "watch"
            return [
                IntelligenceSignal(
                    kind="earnings",
                    severity=severity,
                    message=(
                        f"Earnings in {days} day(s) ({earnings.upcoming_report_date}, "
                        f"{period}, {timing}) — expect elevated volatility."
                    ),
                    symbol=symbol,
                )
            ]
        if 4 <= days <= 7:
            return [
                IntelligenceSignal(
                    kind="earnings",
                    severity="info",
                    message=(
                        f"Earnings next week on {earnings.upcoming_report_date} "
                        f"({period})."
                    ),
                    symbol=symbol,
                )
            ]
        return []

    @staticmethod
    def _valuation_signals(
        *, research: ResearchContext, symbol: str
    ) -> list[IntelligenceSignal]:
        if not research.snapshot or not research.snapshot.range52w:
            return []

        range_text = research.snapshot.range52w
        price = research.snapshot.price
        try:
            parts = range_text.replace("$", "").split("–")
            if len(parts) != 2:
                parts = range_text.replace("$", "").split("-")
            low = float(parts[0].strip())
            high = float(parts[1].strip())
        except (ValueError, IndexError):
            return []

        if high <= low:
            return []

        position_in_range = (price - low) / (high - low)
        if position_in_range >= 0.9:
            return [
                IntelligenceSignal(
                    kind="valuation",
                    severity="watch",
                    message=(
                        f"Price ${price:.2f} is near the 52-week high "
                        f"({range_text}) — limited upside cushion."
                    ),
                    symbol=symbol,
                )
            ]
        if position_in_range <= 0.1:
            return [
                IntelligenceSignal(
                    kind="valuation",
                    severity="info",
                    message=(
                        f"Price ${price:.2f} is near the 52-week low "
                        f"({range_text}) — potential value or broken thesis."
                    ),
                    symbol=symbol,
                )
            ]
        return []

    @staticmethod
    def _sec_trend_signals(
        *, research: ResearchContext, symbol: str
    ) -> list[IntelligenceSignal]:
        trends = research.sec_ratio_trends
        if len(trends) < 2:
            return []

        signals: list[IntelligenceSignal] = []
        latest = trends[0]
        prior = trends[1]

        rev_latest = _parse_pct(latest.revenue_growth_yoy)
        rev_prior = _parse_pct(prior.revenue_growth_yoy)
        if rev_latest is not None and rev_prior is not None:
            if rev_latest < rev_prior - 5 and rev_latest < 5:
                signals.append(
                    IntelligenceSignal(
                        kind="fundamentals",
                        severity="watch",
                        message=(
                            f"SEC revenue growth decelerated to {latest.revenue_growth_yoy} "
                            f"from {prior.revenue_growth_yoy} YoY."
                        ),
                        symbol=symbol,
                    )
                )

        margin_latest = _parse_pct(latest.net_margin)
        margin_prior = _parse_pct(prior.net_margin)
        if margin_latest is not None and margin_prior is not None:
            if margin_latest < margin_prior - 3:
                signals.append(
                    IntelligenceSignal(
                        kind="fundamentals",
                        severity="watch",
                        message=(
                            f"Net margin compressed to {latest.net_margin} "
                            f"from {prior.net_margin} (SEC filed)."
                        ),
                        symbol=symbol,
                    )
                )

        return signals

    @staticmethod
    def _news_sentiment_signals(
        *, research: ResearchContext, symbol: str
    ) -> list[IntelligenceSignal]:
        enriched = research.enriched_news
        if enriched is None:
            return []

        sentiment = enriched.overall_sentiment
        perf = research.performance
        one_month = _parse_return_pct(perf.oneMonth if perf else None)

        if sentiment in {"strongly_bearish", "bearish"} and one_month is not None:
            if one_month < -5:
                return [
                    IntelligenceSignal(
                        kind="thesis_drift",
                        severity="warning",
                        message=(
                            "Bearish news sentiment aligns with recent price weakness "
                            f"({perf.oneMonth} over 1 month) — thesis may be weakening."
                        ),
                        symbol=symbol,
                    )
                ]
        if sentiment in {"strongly_bullish", "bullish"} and one_month is not None:
            if one_month < -8:
                return [
                    IntelligenceSignal(
                        kind="thesis_drift",
                        severity="watch",
                        message=(
                            "Bullish news sentiment but price is down "
                            f"{perf.oneMonth} over 1 month — possible disconnect."
                        ),
                        symbol=symbol,
                    )
                ]
        return []

    @staticmethod
    def _performance_signals(
        *, research: ResearchContext, symbol: str
    ) -> list[IntelligenceSignal]:
        perf = research.performance
        if perf is None:
            return []

        one_year = _parse_return_pct(perf.oneYear)
        one_month = _parse_return_pct(perf.oneMonth)
        if one_year is not None and one_month is not None:
            if one_year > 20 and one_month < -10:
                return [
                    IntelligenceSignal(
                        kind="momentum",
                        severity="watch",
                        message=(
                            f"Strong 1-year return ({perf.oneYear}) but recent "
                            f"pullback ({perf.oneMonth} over 1 month)."
                        ),
                        symbol=symbol,
                    )
                ]
        return []


def build_proactive_alerts(
    *,
    portfolio_signals: list[IntelligenceSignal],
    suggested_actions: list,
    earnings_this_week: list[str],
) -> list:
    from app.models.intelligence_models import ProactiveAlert

    alerts: list[ProactiveAlert] = []
    seen: set[tuple[str, str]] = set()

    def add(action: AnalysisAction, reason: str, priority: int, symbol: str | None):
        key = (action.value, symbol or "")
        if key in seen:
            return
        seen.add(key)
        alerts.append(
            ProactiveAlert(
                action=action,
                label=action.label,
                reason=reason,
                priority=priority,
                symbol=symbol,
            )
        )

    for signal in portfolio_signals:
        if signal.kind == "concentration" and signal.severity == "critical":
            add(
                AnalysisAction.CONCENTRATION_CHECK,
                signal.message,
                priority=1,
                symbol=signal.symbol,
            )
        elif signal.kind == "earnings" and signal.severity in {"warning", "watch"}:
            add(
                AnalysisAction.DAILY_SUMMARY,
                signal.message,
                priority=2,
                symbol=signal.symbol,
            )

    for sym in earnings_this_week[:5]:
        add(
            AnalysisAction.DAILY_SUMMARY,
            f"{sym} reports earnings this week — review volatility exposure.",
            priority=2,
            symbol=sym,
        )

    for suggestion in suggested_actions:
        add(
            suggestion.action,
            suggestion.reason,
            priority=suggestion.priority,
            symbol=None,
        )

    alerts.sort(key=lambda alert: alert.priority)
    return alerts
