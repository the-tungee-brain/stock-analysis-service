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
        greeting_name = recipient_name or "there"
        subject = f"Your {self.brand_name} morning brief"

        lines: list[str] = [
            f"Good morning, {greeting_name}.",
            "",
        ]

        if brief.macro_regime:
            lines.extend(["Macro", brief.macro_regime, ""])

        if brief.digest and brief.digest.macro_news:
            lines.append("Market headlines")
            for item in brief.digest.macro_news:
                source = f" ({item.source})" if item.source else ""
                link = f" {item.url}" if item.url else ""
                lines.append(f"- {item.headline}{source}{link}")
            lines.append("")

        if brief.changes and brief.changes.summary:
            lines.extend(["Since yesterday", brief.changes.summary, ""])

        if brief.top_alerts:
            lines.append("Top alerts")
            for alert in brief.top_alerts[:5]:
                symbol = f" ({alert.symbol})" if alert.symbol else ""
                lines.append(f"- [{alert.label}{symbol}] {alert.reason}")
            lines.append("")

        if brief.digest and brief.digest.earnings_this_week:
            lines.append(
                "Earnings this week: " + ", ".join(brief.digest.earnings_this_week[:8])
            )
            lines.append("")

        lines.extend(
            [
                f"Open {self.brand_name}: {self.frontend_uri}/portfolio",
                "",
                f"— {self.brand_name}",
            ]
        )

        text_body = "\n".join(lines)

        html_sections: list[str] = [
            f"<p>Good morning, {html.escape(greeting_name)}.</p>",
        ]

        if brief.macro_regime:
            html_sections.append(
                "<h3>Macro</h3>" f"<p>{html.escape(brief.macro_regime)}</p>"
            )

        if brief.digest and brief.digest.macro_news:
            items = "".join(
                "<li>"
                + (
                    f'<a href="{html.escape(item.url)}">{html.escape(item.headline)}</a>'
                    if item.url
                    else html.escape(item.headline)
                )
                + f"{f' <em>({html.escape(item.source)})</em>' if item.source else ''}"
                + "</li>"
                for item in brief.digest.macro_news
            )
            html_sections.append(f"<h3>Market headlines</h3><ul>{items}</ul>")

        if brief.changes and brief.changes.summary:
            html_sections.append(
                "<h3>Since yesterday</h3>"
                f"<p>{html.escape(brief.changes.summary)}</p>"
            )

        if brief.top_alerts:
            items = "".join(
                "<li>"
                f"<strong>{html.escape(alert.label)}</strong>"
                f"{f' ({html.escape(alert.symbol)})' if alert.symbol else ''}: "
                f"{html.escape(alert.reason)}"
                "</li>"
                for alert in brief.top_alerts[:5]
            )
            html_sections.append(f"<h3>Top alerts</h3><ul>{items}</ul>")

        if brief.digest and brief.digest.earnings_this_week:
            html_sections.append(
                "<h3>Earnings this week</h3><p>"
                + html.escape(", ".join(brief.digest.earnings_this_week[:8]))
                + "</p>"
            )

        html_sections.append(
            f'<p style="margin-top: 32px;"><a href="{html.escape(self.frontend_uri)}/portfolio" '
            f'style="display: inline-block; padding: 12px 20px; background: #6366f1; '
            f'color: #ffffff; text-decoration: none; border-radius: 10px; font-weight: 600; '
            f'font-size: 14px;">Open {html.escape(self.brand_name)}</a></p>'
        )
        html_sections.append(
            f'<p style="margin-top: 24px; color: #71717a; font-size: 13px;">'
            f"— {html.escape(self.brand_name)}</p>"
        )

        html_body = (
            '<html><body style="font-family: -apple-system, BlinkMacSystemFont, '
            '"Segoe UI", Roboto, sans-serif; line-height: 1.6; '
            'color: #18181b; background: #fafafa; max-width: 640px; margin: 0 auto; '
            'padding: 32px 24px;">'
            '<div style="border: 1px solid #e4e4e7; border-radius: 12px; '
            'background: #ffffff; padding: 28px 24px;">'
            + "".join(html_sections)
            + "</div></body></html>"
        )

        return subject, text_body, html_body
