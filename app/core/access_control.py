import os

DEFAULT_MAX_ACTIVE_USERS = 5


def max_active_users() -> int:
    raw = os.getenv("MAX_ACTIVE_USERS", str(DEFAULT_MAX_ACTIVE_USERS))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_ACTIVE_USERS
    return max(1, value)
