from app.adapters.user.app_user_adapter import AppUserAdapter
from app.models.user_models import AppUserItem
from typing import Optional


class AppUserBuilder:
    def __init__(self, app_user_adapter: AppUserAdapter):
        self.app_user_adapter = app_user_adapter

    def get_user_by_identity_sub(self, identity_sub: str) -> Optional[AppUserItem]:
        return self.app_user_adapter.get_by_identity_sub(identity_sub=identity_sub)

    def save_user(self, item: AppUserItem):
        return self.app_user_adapter.save(item=item)
