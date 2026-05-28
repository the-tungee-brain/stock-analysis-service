from datetime import datetime, timezone


def yahoo_snapshot_as_of() -> str:
    return datetime.now(timezone.utc).isoformat()
