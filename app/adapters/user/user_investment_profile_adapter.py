from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

import oracledb

from app.models.strategy_models import (
    DividendStrategyConfig,
    EtfCoreStrategyConfig,
    IncomeVsGrowth,
    InvestmentStrategy,
    OptionsExperience,
    RiskTolerance,
    UserInvestmentProfile,
    UserInvestmentProfileUpdate,
    WheelStrategyConfig,
)


class UserInvestmentProfileAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "USER_INVESTMENT_PROFILE"

    @staticmethod
    def _parse_config(config_json: str | None) -> dict[str, Any]:
        if not config_json:
            return {}
        try:
            payload = json.loads(config_json)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _config_to_json(
        *,
        wheel: WheelStrategyConfig | None,
        dividend: DividendStrategyConfig | None,
        etf_core: EtfCoreStrategyConfig | None,
    ) -> str | None:
        payload: dict[str, Any] = {}
        if wheel is not None:
            payload["wheel"] = wheel.model_dump(mode="json", by_alias=True)
        if dividend is not None:
            payload["dividend"] = dividend.model_dump(mode="json", by_alias=True)
        if etf_core is not None:
            payload["etfCore"] = etf_core.model_dump(mode="json", by_alias=True)
        if not payload:
            return None
        return json.dumps(payload)

    def _row_to_profile(self, row: tuple) -> UserInvestmentProfile:
        (
            user_id,
            primary_strategy,
            risk_tolerance,
            options_experience,
            income_vs_growth,
            config_json,
            onboarding_completed_at,
            created_at,
            updated_at,
        ) = row

        config = self._parse_config(config_json)
        wheel_raw = config.get("wheel")
        dividend_raw = config.get("dividend")
        etf_raw = config.get("etfCore")

        return UserInvestmentProfile(
            user_id=user_id,
            primary_strategy=(
                InvestmentStrategy(primary_strategy) if primary_strategy else None
            ),
            risk_tolerance=risk_tolerance or "moderate",
            options_experience=options_experience or "beginner",
            income_vs_growth=income_vs_growth or "balanced",
            wheel=WheelStrategyConfig.model_validate(wheel_raw) if wheel_raw else None,
            dividend=(
                DividendStrategyConfig.model_validate(dividend_raw)
                if dividend_raw
                else None
            ),
            etf_core=EtfCoreStrategyConfig.model_validate(etf_raw) if etf_raw else None,
            onboarding_completed_at=(
                onboarding_completed_at.replace(tzinfo=timezone.utc)
                if onboarding_completed_at
                else None
            ),
            created_at=created_at.replace(tzinfo=timezone.utc) if created_at else None,
            updated_at=updated_at.replace(tzinfo=timezone.utc) if updated_at else None,
        )

    def get_by_user_id(self, user_id: str) -> Optional[UserInvestmentProfile]:
        sql = f"""
            SELECT user_id, primary_strategy, risk_tolerance, options_experience,
                   income_vs_growth, config_json, onboarding_completed_at,
                   created_at, updated_at
            FROM {self.table_name}
            WHERE user_id = :user_id
        """
        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(sql, {"user_id": user_id})
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_profile(row)
        finally:
            self.client.release(con)

    def upsert(
        self,
        *,
        user_id: str,
        update: UserInvestmentProfileUpdate,
        existing: UserInvestmentProfile | None = None,
    ) -> UserInvestmentProfile:
        current = existing or self.get_by_user_id(user_id)

        primary_strategy = (
            update.primary_strategy
            if update.primary_strategy is not None
            else (current.primary_strategy if current else None)
        )
        risk_tolerance: RiskTolerance = (
            update.risk_tolerance
            if update.risk_tolerance is not None
            else (current.risk_tolerance if current else "moderate")
        )
        options_experience: OptionsExperience = (
            update.options_experience
            if update.options_experience is not None
            else (current.options_experience if current else "beginner")
        )
        income_vs_growth: IncomeVsGrowth = (
            update.income_vs_growth
            if update.income_vs_growth is not None
            else (current.income_vs_growth if current else "balanced")
        )
        wheel = (
            update.wheel
            if update.wheel is not None
            else (current.wheel if current else None)
        )
        dividend = (
            update.dividend
            if update.dividend is not None
            else (current.dividend if current else None)
        )
        etf_core = (
            update.etf_core
            if update.etf_core is not None
            else (current.etf_core if current else None)
        )

        onboarding_completed_at = (
            current.onboarding_completed_at if current else None
        )
        if update.complete_onboarding:
            onboarding_completed_at = datetime.now(timezone.utc)

        config_json = self._config_to_json(
            wheel=wheel,
            dividend=dividend,
            etf_core=etf_core,
        )

        sql = f"""
            MERGE INTO {self.table_name} t
            USING (
                SELECT :user_id AS user_id FROM dual
            ) s
            ON (t.user_id = s.user_id)
            WHEN MATCHED THEN UPDATE SET
                primary_strategy = :primary_strategy,
                risk_tolerance = :risk_tolerance,
                options_experience = :options_experience,
                income_vs_growth = :income_vs_growth,
                config_json = :config_json,
                onboarding_completed_at = :onboarding_completed_at,
                updated_at = systimestamp
            WHEN NOT MATCHED THEN INSERT (
                user_id,
                primary_strategy,
                risk_tolerance,
                options_experience,
                income_vs_growth,
                config_json,
                onboarding_completed_at
            ) VALUES (
                :user_id,
                :primary_strategy,
                :risk_tolerance,
                :options_experience,
                :income_vs_growth,
                :config_json,
                :onboarding_completed_at
            )
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(
                sql,
                {
                    "user_id": user_id,
                    "primary_strategy": (
                        primary_strategy.value if primary_strategy else None
                    ),
                    "risk_tolerance": risk_tolerance,
                    "options_experience": options_experience,
                    "income_vs_growth": income_vs_growth,
                    "config_json": config_json,
                    "onboarding_completed_at": onboarding_completed_at,
                },
            )
            con.commit()
        finally:
            self.client.release(con)

        profile = self.get_by_user_id(user_id)
        if profile is None:
            raise RuntimeError(f"Failed to upsert investment profile for {user_id}")
        return profile
