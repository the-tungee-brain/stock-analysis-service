from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ValidationBucketMetrics(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    bucket: str
    count: int
    avg_ret_5d: float | None = Field(default=None, alias="avgRet5D")
    avg_ret_10d: float | None = Field(default=None, alias="avgRet10D")
    avg_ret_20d: float | None = Field(default=None, alias="avgRet20D")
    avg_excess_5d: float | None = Field(default=None, alias="avgExcess5D")
    avg_excess_10d: float | None = Field(default=None, alias="avgExcess10D")
    avg_excess_20d: float | None = Field(default=None, alias="avgExcess20D")
    hit_rate_5d: float | None = Field(default=None, alias="hitRate5D")
    hit_rate_10d: float | None = Field(default=None, alias="hitRate10D")
    hit_rate_20d: float | None = Field(default=None, alias="hitRate20D")


class EmergingLeadersValidationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    snapshot_dates: int = Field(alias="snapshotDates")
    snapshot_rows: int = Field(alias="snapshotRows")
    labeled_rows: int = Field(alias="labeledRows")
    setup_score_buckets: list[ValidationBucketMetrics] = Field(alias="setupScoreBuckets")
    compression_velocity_buckets: list[ValidationBucketMetrics] = Field(
        alias="compressionVelocityBuckets"
    )
    stage_buckets: list[ValidationBucketMetrics] = Field(alias="stageBuckets")
    top_decile: ValidationBucketMetrics = Field(alias="topDecile")
    methodology: str
