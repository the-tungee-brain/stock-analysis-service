from __future__ import annotations

from fastapi import APIRouter

from app.core.momentum_breakout_feature_flags import get_momentum_breakout_feature_flags
from app.models.momentum_breakout_feature_models import (
    MomentumBreakoutFeatureFlagsDto,
    MomentumBreakoutFeatureStatusResponse,
)

router = APIRouter()


@router.get(
    "/strategy/momentum-breakout/feature-status",
    response_model=MomentumBreakoutFeatureStatusResponse,
    response_model_by_alias=True,
)
async def get_momentum_breakout_feature_status() -> MomentumBreakoutFeatureStatusResponse:
    flags = get_momentum_breakout_feature_flags()
    return MomentumBreakoutFeatureStatusResponse(
        flags=MomentumBreakoutFeatureFlagsDto(
            alertsEnabled=flags.alerts_enabled,
            alertCreationEnabled=flags.alert_creation_enabled,
            alertNotificationsEnabled=flags.alert_notifications_enabled,
            paperAnalyticsEnabled=flags.paper_analytics_enabled,
        ),
    )
