import asyncio

from fastapi import APIRouter

from app.models.emerging_leaders_validation_models import (
    EmergingLeadersValidationResponse,
    ValidationBucketMetrics,
)
from app.services.emerging_leaders_validation_service import (
    build_emerging_leaders_validation_dashboard,
)

router = APIRouter()


@router.get(
    "/research/emerging-leaders-validation",
    response_model=EmergingLeadersValidationResponse,
)
async def get_emerging_leaders_validation() -> EmergingLeadersValidationResponse:
    payload = await asyncio.to_thread(build_emerging_leaders_validation_dashboard)
    return EmergingLeadersValidationResponse(
        snapshot_dates=payload["snapshotDates"],
        snapshot_rows=payload["snapshotRows"],
        labeled_rows=payload["labeledRows"],
        setup_score_buckets=[
            ValidationBucketMetrics.model_validate(b)
            for b in payload["setupScoreBuckets"]
        ],
        compression_velocity_buckets=[
            ValidationBucketMetrics.model_validate(b)
            for b in payload["compressionVelocityBuckets"]
        ],
        stage_buckets=[
            ValidationBucketMetrics.model_validate(b) for b in payload["stageBuckets"]
        ],
        top_decile=ValidationBucketMetrics.model_validate(payload["topDecile"]),
        methodology=payload["methodology"],
    )
