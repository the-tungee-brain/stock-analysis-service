import os

from fastapi import APIRouter, Depends, Header, HTTPException

from app.dependencies.service_dependencies import get_morning_brief_delivery_service
from app.services.morning_brief_delivery_service import MorningBriefDeliveryService

router = APIRouter()


def _verify_cron_secret(x_cron_secret: str | None = Header(default=None)) -> None:
    expected = os.getenv("CRON_SECRET")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="CRON_SECRET is not configured on the server.",
        )
    if not x_cron_secret or x_cron_secret != expected:
        raise HTTPException(status_code=401, detail="Invalid cron secret.")


@router.post("/internal/dispatch-morning-briefs")
def dispatch_morning_briefs(
    force: bool = False,
    _: None = Depends(_verify_cron_secret),
    delivery_service: MorningBriefDeliveryService = Depends(
        get_morning_brief_delivery_service
    ),
):
    result = delivery_service.dispatch_all(force=force)
    return {
        "attempted": result.attempted,
        "sent": result.sent,
        "skipped": result.skipped,
        "failed": result.failed,
        "errors": result.errors,
    }


@router.post("/internal/prewarm-morning-briefs")
def prewarm_morning_briefs(
    _: None = Depends(_verify_cron_secret),
    delivery_service: MorningBriefDeliveryService = Depends(
        get_morning_brief_delivery_service
    ),
):
    result = delivery_service.prewarm_all()
    return {
        "attempted": result.attempted,
        "warmed": result.warmed,
        "skipped": result.skipped,
        "failed": result.failed,
        "errors": result.errors,
    }
