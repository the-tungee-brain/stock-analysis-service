from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class AppUserItem(BaseModel):
    id: str
    identity_sub: str
    identity_provider: str
    email: EmailStr
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: datetime
    last_login_at: Optional[datetime] = None


class IdentityPayload(BaseModel):
    identity_sub: str
    identity_provider: str
    email: EmailStr
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
