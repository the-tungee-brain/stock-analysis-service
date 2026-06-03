"""Product-facing operational health API."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.product.models import SystemHealthResponseV1
from app.services.product_api_service import get_system_health_v1

router = APIRouter(tags=["Product — Health"])


@router.get("/health", response_model=SystemHealthResponseV1)
async def system_health() -> SystemHealthResponseV1:
    """
    Pipeline and snapshot health for Web/iOS status indicators.

    ``system_status``: ``ok`` | ``degraded`` | ``failing`` based on run freshness.
    """
    return get_system_health_v1()
