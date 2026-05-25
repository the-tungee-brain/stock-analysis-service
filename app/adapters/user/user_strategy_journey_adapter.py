from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import oracledb

from app.models.strategy_models import (
    InvestmentStrategy,
    JourneyStep,
    UserStrategyJourney,
)


class UserStrategyJourneyAdapter:
    def __init__(self, client: oracledb.ConnectionPool):
        self.client = client
        self.table_name = "USER_STRATEGY_JOURNEY"

    @staticmethod
    def _completion_pct(steps: list[JourneyStep]) -> float:
        if not steps:
            return 0.0
        completed = sum(
            1 for step in steps if step.status.value in {"completed", "skipped"}
        )
        return round((completed / len(steps)) * 100.0, 1)

    def _row_to_journey(self, row: tuple) -> UserStrategyJourney:
        (
            record_id,
            user_id,
            strategy,
            current_step_id,
            steps_json,
            started_at,
            completed_at,
            _updated_at,
        ) = row

        raw_steps = json.loads(steps_json or "[]")
        steps = [JourneyStep.model_validate(item) for item in raw_steps]
        completion_pct = self._completion_pct(steps)

        return UserStrategyJourney(
            id=record_id,
            user_id=user_id,
            strategy=InvestmentStrategy(strategy),
            current_step_id=current_step_id,
            steps=steps,
            completion_pct=completion_pct,
            started_at=started_at.replace(tzinfo=timezone.utc) if started_at else None,
            completed_at=(
                completed_at.replace(tzinfo=timezone.utc) if completed_at else None
            ),
        )

    def get_by_user_and_strategy(
        self, user_id: str, strategy: InvestmentStrategy
    ) -> Optional[UserStrategyJourney]:
        sql = f"""
            SELECT id, user_id, strategy, current_step_id, steps_json,
                   started_at, completed_at, updated_at
            FROM {self.table_name}
            WHERE user_id = :user_id
              AND strategy = :strategy
        """
        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(
                sql,
                {"user_id": user_id, "strategy": strategy.value},
            )
            row = cur.fetchone()
            if not row:
                return None
            return self._row_to_journey(row)
        finally:
            self.client.release(con)

    def upsert(
        self,
        *,
        user_id: str,
        strategy: InvestmentStrategy,
        steps: list[JourneyStep],
        current_step_id: str | None,
        completed_at: datetime | None = None,
    ) -> UserStrategyJourney:
        record_id = str(uuid4())
        steps_json = json.dumps(
            [step.model_dump(mode="json", by_alias=True) for step in steps]
        )

        sql = f"""
            MERGE INTO {self.table_name} t
            USING (
                SELECT :user_id AS user_id, :strategy AS strategy FROM dual
            ) s
            ON (t.user_id = s.user_id AND t.strategy = s.strategy)
            WHEN MATCHED THEN UPDATE SET
                current_step_id = :current_step_id,
                steps_json = :steps_json,
                completed_at = :completed_at,
                updated_at = systimestamp
            WHEN NOT MATCHED THEN INSERT (
                id,
                user_id,
                strategy,
                current_step_id,
                steps_json,
                completed_at
            ) VALUES (
                :record_id,
                :user_id,
                :strategy,
                :current_step_id,
                :steps_json,
                :completed_at
            )
        """

        con = self.client.acquire()
        try:
            cur = con.cursor()
            cur.execute(
                sql,
                {
                    "record_id": record_id,
                    "user_id": user_id,
                    "strategy": strategy.value,
                    "current_step_id": current_step_id,
                    "steps_json": steps_json,
                    "completed_at": completed_at,
                },
            )
            con.commit()
        finally:
            self.client.release(con)

        journey = self.get_by_user_and_strategy(user_id, strategy)
        if journey is None:
            raise RuntimeError(
                f"Failed to upsert strategy journey for {user_id}/{strategy.value}"
            )
        return journey
