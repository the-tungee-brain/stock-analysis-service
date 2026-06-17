from __future__ import annotations

import html
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from app.adapters.email.email_adapter import EmailAdapter
from app.adapters.portfolio.morning_brief_delivery_adapter import (
    MorningBriefDeliveryAdapter,
)
from app.adapters.user.app_user_adapter import AppUserAdapter
from app.models.intelligence_models import HoldingCompanyNewsItem, PortfolioNewsItem
from app.models.portfolio_memory_models import MorningBrief
from app.services.portfolio_analysis_service import PortfolioAnalysisService
from app.services.portfolio_memory_service import PortfolioMemoryService
from app.services.portfolio_service import PortfolioService
from app.services.schwab_auth_service import SchwabAuthService, SchwabReauthRequired
from app.services.transaction_service import DEFAULT_DAYS_BACK, TransactionService

logger = logging.getLogger(__name__)


@dataclass
class MorningBriefDispatchResult:
    attempted: int = 0
    sent: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class MorningBriefPrewarmResult:
    attempted: int = 0
    warmed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)


class MorningBriefDeliveryService:
    def __init__(
        self,
        *,
        app_user_adapter: AppUserAdapter,
        delivery_adapter: MorningBriefDeliveryAdapter,
        email_adapter: EmailAdapter,
        portfolio_analysis_service: PortfolioAnalysisService,
        portfolio_service: PortfolioService,
        transaction_service: TransactionService,
        schwab_auth_service: SchwabAuthService,
        portfolio_memory_service: PortfolioMemoryService,
    ):
        self.app_user_adapter = app_user_adapter
        self.delivery_adapter = delivery_adapter
        self.email_adapter = email_adapter
        self.portfolio_analysis_service = portfolio_analysis_service
        self.portfolio_service = portfolio_service
        self.transaction_service = transaction_service
        self.schwab_auth_service = schwab_auth_service
        self.portfolio_memory_service = portfolio_memory_service
        self.frontend_uri = os.getenv(
            "POWERPOCKET_FRONTEND_URI", "https://tomcrest.com"
        ).rstrip("/")
        self.brand_name = os.getenv("MORNING_BRIEF_BRAND_NAME", "Tomcrest")

    def build_for_user(
        self,
        *,
        user_id: str,
        refresh: bool = False,
        persist: bool = True,
    ) -> MorningBrief | None:
        try:
            schwab_token = self.schwab_auth_service.get_valid_token_by_user_id(
                user_id=user_id
            )
        except SchwabReauthRequired:
            return None

        account_map = self.portfolio_service.get_enriched_account(
            access_token=schwab_token.access_token
        )
        account = account_map["account"]
        positions = account.securitiesAccount.positions
        account_number = account.securitiesAccount.accountNumber

        suggested_actions = []
        try:
            recent_activity = self.transaction_service.build_recent_activity_summary(
                account_number=account_number,
                access_token=schwab_token.access_token,
                user_id=user_id,
                days_back=DEFAULT_DAYS_BACK,
                refresh=refresh,
            )
            if recent_activity:
                suggested_actions = recent_activity.suggested_actions
        except Exception:
            suggested_actions = []

        portfolio_brief = self.portfolio_analysis_service.build_portfolio_brief_with_cache(
            user_id=user_id,
            account=account,
            positions=positions,
            access_token=schwab_token.access_token,
            suggested_actions=suggested_actions,
            assignment_risk_summary=account_map["assignmentRiskSummary"],
            refresh=refresh,
        )

        if persist:
            self.portfolio_memory_service.capture_snapshot(
                user_id=user_id,
                account=account,
                positions=positions,
                portfolio_brief=portfolio_brief,
            )
            self.portfolio_memory_service.record_alerts(
                user_id=user_id,
                alerts=portfolio_brief.alerts,
            )

        return self.portfolio_memory_service.build_morning_brief(
            user_id=user_id,
            portfolio_brief=portfolio_brief,
            current_alerts=portfolio_brief.alerts,
        )

    def _dispatch_one_user(self, user, *, force: bool) -> str:
        """Returns sent | skipped | failed."""
        user_key = user.identity_sub

        if not force and self.delivery_adapter.was_delivered_today(user_key):
            return "skipped"

        try:
            brief = self.build_for_user(user_id=user_key, persist=True)
            if brief is None:
                return "skipped"

            subject, text_body, html_body = self._render_email(
                recipient_name=user.full_name,
                brief=brief,
            )
            self.email_adapter.send_email(
                to_email=str(user.email),
                subject=subject,
                html=html_body,
                text=text_body,
            )
            self.delivery_adapter.record_delivery(
                user_id=user_key,
                email=str(user.email),
                status="sent",
            )
            return "sent"
        except SchwabReauthRequired:
            return "skipped"
        except Exception as exc:
            message = f"{user.email}: {exc}"
            logger.exception("Morning brief delivery failed for user %s", user_key)
            try:
                self.delivery_adapter.record_delivery(
                    user_id=user_key,
                    email=str(user.email),
                    status="failed",
                    error_message=str(exc)[:2000],
                )
            except Exception:
                logger.exception(
                    "Failed to record morning brief delivery failure for %s",
                    user.id,
                )
            raise RuntimeError(message) from exc

    def dispatch_all(self, *, force: bool = False) -> MorningBriefDispatchResult:
        result = MorningBriefDispatchResult()

        if not self.email_adapter.enabled:
            result.errors.append("Email delivery is not configured.")
            return result

        users = self.app_user_adapter.list_users_with_schwab()
        if not users:
            return result

        max_workers = max(
            1,
            int(os.getenv("MORNING_BRIEF_DISPATCH_WORKERS", "20")),
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._dispatch_one_user, user, force=force): user
                for user in users
            }
            for future in as_completed(futures):
                result.attempted += 1
                try:
                    outcome = future.result()
                    if outcome == "sent":
                        result.sent += 1
                    else:
                        result.skipped += 1
                except Exception as exc:
                    result.failed += 1
                    result.errors.append(str(exc))

        logger.info(
            "morning brief dispatch finished attempted=%s sent=%s skipped=%s failed=%s force=%s",
            result.attempted,
            result.sent,
            result.skipped,
            result.failed,
            force,
        )
        return result

    def _prewarm_one_user(self, user) -> str:
        """Returns warmed | skipped | failed."""
        user_key = user.identity_sub
        try:
            brief = self.build_for_user(
                user_id=user_key,
                refresh=True,
                persist=False,
            )
            if brief is None:
                return "skipped"
            return "warmed"
        except SchwabReauthRequired:
            return "skipped"
        except Exception as exc:
            message = f"{user.email}: {exc}"
            logger.exception("Morning brief pre-warm failed for user %s", user_key)
            raise RuntimeError(message) from exc

    def prewarm_all(self) -> MorningBriefPrewarmResult:
        """Warm portfolio brief Redis cache before email dispatch (no snapshots or email)."""
        result = MorningBriefPrewarmResult()
        users = self.app_user_adapter.list_users_with_schwab()
        if not users:
            return result

        max_workers = max(
            1,
            int(os.getenv("MORNING_BRIEF_PREWARM_WORKERS", os.getenv("MORNING_BRIEF_DISPATCH_WORKERS", "20"))),
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._prewarm_one_user, user): user for user in users
            }
            for future in as_completed(futures):
                result.attempted += 1
                try:
                    outcome = future.result()
                    if outcome == "warmed":
                        result.warmed += 1
                    else:
                        result.skipped += 1
                except Exception as exc:
                    result.failed += 1
                    result.errors.append(str(exc))

        logger.info(
            "morning brief prewarm finished attempted=%s warmed=%s skipped=%s failed=%s",
            result.attempted,
            result.warmed,
            result.skipped,
            result.failed,
        )
        return result

    def _render_email(
        self,
        *,
        recipient_name: str | None,
        brief: MorningBrief,
    ) -> tuple[str, str, str]:
        greeting_name = self._first_name(recipient_name)
        subject = f"Your {self.brand_name} morning brief"

        lines: list[str] = [
            f"Good morning, {greeting_name}.",
            "",
        ]

        snapshot_lines = self._portfolio_snapshot_lines(brief)
        if snapshot_lines:
            lines.extend(["Portfolio Snapshot", *snapshot_lines, ""])

        market_lines = self._market_overview_lines(brief.macro_regime)
        if market_lines:
            lines.extend(["Market Overview", *market_lines, ""])

        change_lines = self._portfolio_change_lines(brief)
        if change_lines:
            lines.extend(["Portfolio Changes", *change_lines, ""])

        alert_lines = self._risk_alert_lines(brief)
        if alert_lines:
            lines.extend(["Risk Alerts", *alert_lines, ""])

        news_lines = self._portfolio_news_lines(brief)
        if news_lines:
            lines.extend(["Portfolio News", *news_lines, ""])

        insight = self._actionable_insight(brief)
        if insight:
            lines.extend(["Actionable Insight", f"- {insight}", ""])

        lines.extend(
            [
                f"Open {self.brand_name}: {self.frontend_uri}/portfolio",
                "",
                f"- {self.brand_name}",
            ]
        )

        text_body = "\n".join(lines)

        html_sections: list[str] = [
            f"<p>Good morning, {html.escape(greeting_name)}.</p>",
        ]

        if snapshot_lines:
            html_sections.append(
                self._html_list_section("Portfolio Snapshot", snapshot_lines)
            )

        if market_lines:
            html_sections.append(
                self._html_list_section("Market Overview", market_lines)
            )

        if change_lines:
            html_sections.append(
                self._html_list_section("Portfolio Changes", change_lines)
            )

        if alert_lines:
            html_sections.append(self._html_list_section("Risk Alerts", alert_lines))

        news_items = self._portfolio_news_items(brief)
        if news_items:
            items = "".join(self._html_news_item(item) for item in news_items[:3])
            html_sections.append(f"<h3>Portfolio News</h3><ul>{items}</ul>")
        elif brief.digest and brief.digest.macro_news:
            items = "".join(
                "<li>"
                + self._html_link(item.headline, item.url)
                + f"{f' <em>({html.escape(item.source)})</em>' if item.source else ''}"
                + "<br><span style=\"color: #52525b;\">Why it matters: broad market backdrop when no portfolio-specific news is available.</span>"
                + "</li>"
                for item in brief.digest.macro_news[:3]
            )
            html_sections.append(f"<h3>Portfolio News</h3><ul>{items}</ul>")

        if insight:
            html_sections.append(
                "<h3>Actionable Insight</h3>"
                f"<p>{html.escape(insight)}</p>"
            )

        html_sections.append(
            f'<p style="margin-top: 32px;"><a href="{html.escape(self.frontend_uri)}/portfolio" '
            f'style="display: inline-block; padding: 12px 20px; background: #18181b; '
            f'color: #ffffff; text-decoration: none; border-radius: 8px; font-weight: 600; '
            f'font-size: 14px;">Open {html.escape(self.brand_name)}</a></p>'
        )
        html_sections.append(
            f'<p style="margin-top: 24px; color: #71717a; font-size: 13px;">'
            f"- {html.escape(self.brand_name)}</p>"
        )

        html_body = (
            '<html><body style="font-family: -apple-system, BlinkMacSystemFont, '
            '"Segoe UI", Roboto, sans-serif; line-height: 1.5; '
            'color: #18181b; background: #f4f4f5; max-width: 640px; margin: 0 auto; '
            'padding: 32px 24px;">'
            '<div style="border: 1px solid #e4e4e7; border-radius: 8px; '
            'background: #ffffff; padding: 28px 24px;">'
            + "".join(html_sections)
            + "</div></body></html>"
        )

        return subject, text_body, html_body

    @staticmethod
    def _first_name(recipient_name: str | None) -> str:
        if not recipient_name:
            return "there"
        return recipient_name.strip().split()[0] or "there"

    @staticmethod
    def _money(value: float | None, *, signed: bool = False) -> str | None:
        if value is None:
            return None
        if signed and value > 0:
            return f"+${value:,.0f}"
        if value < 0:
            return f"-${abs(value):,.0f}"
        return f"${value:,.0f}"

    @staticmethod
    def _pct(value: float | None, *, signed: bool = False) -> str | None:
        if value is None:
            return None
        prefix = "+" if signed and value > 0 else ""
        return f"{prefix}{value:.2f}%"

    def _portfolio_snapshot_lines(self, brief: MorningBrief) -> list[str]:
        snapshot = brief.snapshot
        if snapshot is None:
            return []

        lines: list[str] = []
        if snapshot.portfolio_value is not None:
            lines.append(f"- Value: {self._money(snapshot.portfolio_value)}")
        if snapshot.day_pnl is not None:
            day_pct = self._pct(snapshot.day_pnl_pct, signed=True)
            pct_suffix = f" ({day_pct})" if day_pct else ""
            lines.append(
                f"- Day P/L: {self._money(snapshot.day_pnl, signed=True)}"
                f"{pct_suffix}"
            )
        if snapshot.cash_available is not None:
            lines.append(f"- Cash: {self._money(snapshot.cash_available)}")
        if snapshot.diversification_score is not None:
            rating = (
                f" ({snapshot.diversification_rating})"
                if snapshot.diversification_rating
                else ""
            )
            lines.append(
                f"- Diversification Score: {snapshot.diversification_score}/100{rating}"
            )
        if snapshot.biggest_winner and snapshot.biggest_winner.day_pnl_pct is not None:
            lines.append(
                f"- Top mover: {snapshot.biggest_winner.symbol} "
                f"{self._pct(snapshot.biggest_winner.day_pnl_pct, signed=True)}"
            )
        if snapshot.biggest_loser and snapshot.biggest_loser.day_pnl_pct is not None:
            lines.append(
                f"- Weakest: {snapshot.biggest_loser.symbol} "
                f"{self._pct(snapshot.biggest_loser.day_pnl_pct, signed=True)}"
            )
        return lines

    @staticmethod
    def _market_overview_lines(macro_regime: str | None) -> list[str]:
        if not macro_regime:
            return []
        parts = [part.strip() for part in macro_regime.split(";") if part.strip()]
        allowed = ("VIX", "S&P 500", "Nasdaq")
        return [f"- {part}" for part in parts if part.startswith(allowed)]

    def _portfolio_change_lines(self, brief: MorningBrief) -> list[str]:
        changes = brief.changes
        if changes is None:
            return []

        lines: list[str] = []
        lines.extend(f"- Added {symbol}" for symbol in changes.new_symbols[:3])
        lines.extend(f"- Removed {symbol}" for symbol in changes.removed_symbols[:3])
        for item in changes.weight_changes[:3]:
            direction = "increased" if item.change_pct > 0 else "decreased"
            lines.append(
                f"- Position size {direction} in {item.symbol} "
                f"({item.previous_weight_pct:.1f}% to {item.current_weight_pct:.1f}%)"
            )
        if changes.liquidation_value_change_pct is not None and abs(changes.liquidation_value_change_pct) >= 0.05:
            amount = self._money(changes.liquidation_value_change, signed=True)
            pct = self._pct(changes.liquidation_value_change_pct, signed=True)
            amount_part = f" {amount}" if amount else ""
            lines.append(f"- Portfolio value changed{amount_part} ({pct})")
        return lines

    @staticmethod
    def _alert_severity(alert) -> tuple[str, str]:
        text = f"{alert.label} {alert.reason}".lower()
        if "sector" in text:
            return ("Warning", "⚠️")
        if any(
            term in text
            for term in ("concentration", "leverage", "margin", "maintenance call")
        ):
            return ("Critical", "🚨")
        if any(
            term in text
            for term in (
                "wash",
                "sizing",
                "assignment",
                "short put",
                "short call",
            )
        ):
            return ("Warning", "⚠️")
        return ("Info", "ℹ️")

    def _risk_alert_lines(self, brief: MorningBrief) -> list[str]:
        alerts = sorted(brief.top_alerts, key=lambda alert: alert.priority)[:3]
        synthetic_info: list[str] = []
        if brief.digest and brief.digest.earnings_this_week:
            earnings = ", ".join(brief.digest.earnings_this_week[:5])
            synthetic_info.append(f"- ℹ️ Info: Earnings this week - {earnings}")

        if not alerts and not synthetic_info:
            return []
        grouped: dict[str, list[str]] = {"Critical": [], "Warning": [], "Info": []}
        for alert in alerts:
            severity, icon = self._alert_severity(alert)
            symbol = f" ({alert.symbol})" if alert.symbol else ""
            grouped[severity].append(
                f"- {icon} {severity}: {alert.label}{symbol} - {alert.reason}"
            )
        remaining = max(3 - sum(len(items) for items in grouped.values()), 0)
        grouped["Info"].extend(synthetic_info[:remaining])
        lines: list[str] = []
        for severity in ("Critical", "Warning", "Info"):
            lines.extend(grouped[severity])
        return lines

    def _portfolio_news_items(
        self,
        brief: MorningBrief,
    ) -> list[PortfolioNewsItem | HoldingCompanyNewsItem]:
        if not brief.digest:
            return []
        return [
            *brief.digest.top_holdings_company_news,
            *brief.digest.top_news,
        ]

    def _portfolio_news_lines(self, brief: MorningBrief) -> list[str]:
        items = self._portfolio_news_items(brief)
        if items:
            lines: list[str] = []
            for item in items[:3]:
                source = self._news_source(item)
                why = self._why_news_matters(item)
                lines.append(f"- {item.headline} ({source})")
                lines.append(f"  Why it matters: {why}")
            return lines

        if brief.digest and brief.digest.macro_news:
            lines = []
            for item in brief.digest.macro_news[:3]:
                source = f" ({item.source})" if item.source else ""
                lines.append(f"- {item.headline}{source}")
                lines.append(
                    "  Why it matters: broad market backdrop when no portfolio-specific news is available."
                )
            return lines
        return []

    @staticmethod
    def _why_news_matters(item: PortfolioNewsItem | HoldingCompanyNewsItem) -> str:
        weight = getattr(item, "weight_pct", None)
        if isinstance(item, HoldingCompanyNewsItem) and item.summary:
            return item.summary
        if weight is not None:
            return f"{item.symbol} is a current holding at about {weight:.1f}% exposure."
        return f"{item.symbol} is currently in the portfolio."

    @staticmethod
    def _news_source(item: PortfolioNewsItem | HoldingCompanyNewsItem) -> str:
        if isinstance(item, HoldingCompanyNewsItem) and item.source:
            return item.source
        return "Portfolio analysis"

    def _actionable_insight(self, brief: MorningBrief) -> str | None:
        snapshot = brief.snapshot
        if snapshot and snapshot.diversification_score is not None:
            if snapshot.diversification_score < 40:
                return (
                    "Diversification remains poor; review the largest position before adding new exposure."
                )
            if snapshot.diversification_score < 60:
                return (
                    "Diversification is fair; prioritize adds that reduce single-name or sector concentration."
                )
        if brief.digest and brief.digest.sector_weights:
            top_sector = max(
                brief.digest.sector_weights,
                key=lambda item: item.weight_pct,
            )
            if top_sector.weight_pct > 35:
                return (
                    f"{top_sector.sector} exposure is {top_sector.weight_pct:.1f}% "
                    "of portfolio value; "
                    "new buys should diversify away from that sector."
                )
        if snapshot and snapshot.cash_available is not None and snapshot.portfolio_value:
            cash_pct = (snapshot.cash_available / snapshot.portfolio_value) * 100.0
            if 5 <= cash_pct <= 20:
                return f"Cash is within a balanced range at {cash_pct:.1f}%."
            if cash_pct > 20:
                return f"Cash is elevated at {cash_pct:.1f}%; consider a staged deployment plan."
        return None

    @staticmethod
    def _html_link(label: str, url: str | None) -> str:
        escaped = html.escape(label)
        if not url:
            return escaped
        return f'<a href="{html.escape(url)}">{escaped}</a>'

    @staticmethod
    def _html_list_section(title: str, lines: list[str]) -> str:
        items = "".join(
            f"<li>{html.escape(line.removeprefix('- ').strip())}</li>"
            for line in lines
            if line.startswith("- ")
        )
        return f"<h3>{html.escape(title)}</h3><ul>{items}</ul>"

    def _html_news_item(self, item: PortfolioNewsItem | HoldingCompanyNewsItem) -> str:
        source = (
            f" <em>({html.escape(self._news_source(item))})</em>"
        )
        return (
            "<li>"
            + self._html_link(item.headline, item.url)
            + source
            + f'<br><span style="color: #52525b;">Why it matters: {html.escape(self._why_news_matters(item))}</span>'
            + "</li>"
        )
