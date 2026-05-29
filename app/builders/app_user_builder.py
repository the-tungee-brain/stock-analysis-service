from typing import Optional

from app.adapters.cache.app_user_cache import AppUserCache
from app.adapters.user.app_user_adapter import AppUserAdapter
from app.models.user_models import AppUserItem


class AppUserBuilder:
    def __init__(
        self,
        app_user_adapter: AppUserAdapter,
        app_user_cache: AppUserCache | None = None,
    ):
        self.app_user_adapter = app_user_adapter
        self.app_user_cache = app_user_cache

    def get_user_by_identity_sub(self, identity_sub: str) -> Optional[AppUserItem]:
        if self.app_user_cache is not None:
            cached = self.app_user_cache.get(identity_sub)
            if cached is not None:
                return cached

        user = self.app_user_adapter.get_by_identity_sub(identity_sub=identity_sub)
        if user is not None and self.app_user_cache is not None:
            self.app_user_cache.put(identity_sub, user)
        return user

    def count_active_users(self) -> int:
        return self.app_user_adapter.count_active_users()

    def save_user(self, item: AppUserItem):
        saved = self.app_user_adapter.save(item=item)
        if self.app_user_cache is not None:
            self.app_user_cache.put(item.identity_sub, item)
        return saved
