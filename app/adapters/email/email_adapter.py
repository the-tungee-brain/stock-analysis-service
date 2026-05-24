from __future__ import annotations

import logging
import os
from datetime import date

import requests

logger = logging.getLogger(__name__)


class EmailAdapter:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        from_email: str | None = None,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key or os.getenv("RESEND_API_KEY")
        self.from_email = from_email or os.getenv(
            "MORNING_BRIEF_FROM_EMAIL", "PowerPocket <brief@powerpocket.app>"
        )
        self.session = session or requests.Session()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.from_email)

    def send_email(
        self,
        *,
        to_email: str,
        subject: str,
        html: str,
        text: str,
    ) -> None:
        if not self.enabled:
            raise RuntimeError(
                "Email delivery is not configured (RESEND_API_KEY / MORNING_BRIEF_FROM_EMAIL)."
            )

        response = self.session.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "from": self.from_email,
                "to": [to_email],
                "subject": subject,
                "html": html,
                "text": text,
            },
            timeout=30,
        )

        if response.status_code >= 400:
            logger.error(
                "Resend API error %s: %s",
                response.status_code,
                response.text[:500],
            )
            response.raise_for_status()
