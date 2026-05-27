from typing import Optional

from app.adapters.user.waitlist_adapter import WaitlistAdapter
from app.models.user_models import IdentityPayload
from app.models.waitlist_models import WaitlistEntryItem


class WaitlistBuilder:
    def __init__(self, waitlist_adapter: WaitlistAdapter):
        self.waitlist_adapter = waitlist_adapter

    def get_by_identity_sub(self, identity_sub: str) -> Optional[WaitlistEntryItem]:
        return self.waitlist_adapter.get_by_identity_sub(identity_sub=identity_sub)

    def save_waiting(self, payload: IdentityPayload) -> WaitlistEntryItem:
        return self.waitlist_adapter.save_waiting(payload)

    def mark_promoted(self, identity_sub: str) -> None:
        self.waitlist_adapter.mark_promoted(identity_sub=identity_sub)

    def get_queue_position(self, identity_sub: str) -> Optional[int]:
        return self.waitlist_adapter.get_queue_position(identity_sub=identity_sub)
