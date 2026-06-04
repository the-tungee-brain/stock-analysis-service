from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.strategy_models import _STRATEGY_MODEL_CONFIG


class HealthCheckDto(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    name: str
    ok: bool
    detail: str


class LaunchReadinessResponse(BaseModel):
    model_config = _STRATEGY_MODEL_CONFIG

    disclaimer: str = (
        "Operational diagnostics only. Not investment advice or performance marketing."
    )
    admin_diagnostics: bool = Field(alias="adminDiagnostics")
    ready: bool
    alert_store_type: str = Field(alias="alertStoreType")
    paper_trade_store_type: str = Field(alias="paperTradeStoreType")
    scheduler_enabled: bool = Field(alias="schedulerEnabled")
    refresh_interval_sec: int = Field(alias="refreshIntervalSec")
    raw_refresh_interval_sec: int = Field(alias="rawRefreshIntervalSec")
    oracle_alert_ddl_required: bool = Field(alias="oracleAlertDdlRequired")
    oracle_alert_ddl_applied: bool | None = Field(alias="oracleAlertDdlApplied")
    oracle_paper_ddl_required: bool = Field(alias="oraclePaperDdlRequired")
    oracle_paper_ddl_applied: bool | None = Field(alias="oraclePaperDdlApplied")
    ddl_detail_alert: str = Field(alias="ddlDetailAlert")
    ddl_detail_paper: str = Field(alias="ddlDetailPaper")
    active_alerts_count: int = Field(alias="activeAlertsCount")
    paper_trade_rows_count: int = Field(alias="paperTradeRowsCount")
    total_alerts_count: int = Field(alias="totalAlertsCount")
    alerts_missing_paper_rows: int = Field(alias="alertsMissingPaperRows")
    latest_lifecycle_update_at: datetime | None = Field(
        default=None, alias="latestLifecycleUpdateAt"
    )
    latest_paper_trade_update_at: datetime | None = Field(
        default=None, alias="latestPaperTradeUpdateAt"
    )
    quote_provider_available: bool = Field(alias="quoteProviderAvailable")
    health_checks: list[HealthCheckDto] = Field(alias="healthChecks")
    warnings: list[str]
