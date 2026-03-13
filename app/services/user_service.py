from app.builders.app_user_builder import AppUserBuilder
from app.models.user_models import AppUserItem
from typing import Optional
from app.models.user_models import IdentityPayload
from datetime import datetime, timezone
import uuid


class UserService:
    def __init__(self, app_user_builder: AppUserBuilder):
        self.app_user_builder = app_user_builder

    def get_user_by_identity_sub(self, identity_sub: str) -> Optional[AppUserItem]:
        return self.app_user_builder.get_user_by_identity_sub(identity_sub=identity_sub)

    def create_or_link_user(self, payload: IdentityPayload) -> AppUserItem:
        existing = self.app_user_builder.get_user_by_identity_sub(
            identity_sub=payload.identity_sub
        )
        if existing:
            return existing

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
