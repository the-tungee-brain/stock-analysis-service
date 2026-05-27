from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class WaitlistEntryItem(BaseModel):
    id: str
    identity_sub: str
    identity_provider: str
    email: EmailStr
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    status: str = "waiting"
    created_at: datetime
    updated_at: datetime
