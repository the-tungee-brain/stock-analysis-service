from app.builders.app_user_builder import AppUserBuilder
from app.builders.waitlist_builder import WaitlistBuilder
from app.models.user_models import AppUserItem
from typing import Optional
from app.models.user_models import IdentityPayload
from datetime import datetime, timezone
import uuid

from app.services.access_control_errors import WaitlistRequiredError


class UserService:
    def __init__(
        self,
        app_user_builder: AppUserBuilder,
        waitlist_builder: WaitlistBuilder,
        *,
        max_active_users: int = 5,
    ):
        self.app_user_builder = app_user_builder
        self.waitlist_builder = waitlist_builder
        self.max_active_users = max_active_users

    def get_user_by_identity_sub(self, identity_sub: str) -> Optional[AppUserItem]:
        return self.app_user_builder.get_user_by_identity_sub(identity_sub=identity_sub)

    def get_persisted_user_by_identity_sub(
        self, identity_sub: str
    ) -> Optional[AppUserItem]:
        return self.app_user_builder.get_persisted_user_by_identity_sub(
            identity_sub=identity_sub
        )

    def create_or_link_user(self, payload: IdentityPayload) -> AppUserItem:
        existing = self.app_user_builder.get_user_by_identity_sub(
            identity_sub=payload.identity_sub
        )
        if existing:
            return existing

        active_count = self.app_user_builder.count_active_users()
        if active_count < self.max_active_users:
            if self.waitlist_builder.get_by_identity_sub(payload.identity_sub):
                self.waitlist_builder.mark_promoted(payload.identity_sub)
            return self._create_user(payload)

        self.waitlist_builder.save_waiting(payload)
        raise WaitlistRequiredError()

    def _create_user(self, payload: IdentityPayload) -> AppUserItem:
        now = datetime.now(timezone.utc)

        user = AppUserItem(
            id=self._generate_id(),
            identity_sub=payload.identity_sub,
            identity_provider=payload.identity_provider,
            email=payload.email,
            full_name=payload.full_name,
            avatar_url=payload.avatar_url,
            created_at=now,
            last_login_at=now,
        )

        self.app_user_builder.save_user(user)
        return user

    def _generate_id(self) -> str:
        return str(uuid.uuid4())
