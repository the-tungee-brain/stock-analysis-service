"""Price-based alerts for trade plan levels."""

from __future__ import annotations

from trade_planner.models import Alert, AlertType, TradePlan, utc_now
from trade_planner.types import StockData


class AlertEngine:
    def evaluate(
        self,
        *,
        plan: TradePlan,
        stock_data: StockData,
        prior_price: float | None = None,
    ) -> list[Alert]:
        current = stock_data.current.close
        alerts: list[Alert] = []
        now = utc_now()

        if plan.direction == "LONG":
            if self._crossed_above(prior_price, current, plan.entry_price):
                alerts.append(
                    self._alert(
                        plan,
                        AlertType.ENTRY_TRIGGERED,
                        current,
                        now,
                        f"{plan.symbol} long entry triggered at {plan.entry_price:.2f}",
                    )
                )
            if current <= plan.stop_price:
                alerts.append(
                    self._alert(
                        plan,
                        AlertType.STOP_HIT,
                        current,
                        now,
                        f"{plan.symbol} stop hit at {plan.stop_price:.2f}",
                    )
                )
            if current >= plan.target_price:
                alerts.append(
                    self._alert(
                        plan,
                        AlertType.TARGET_HIT,
                        current,
                        now,
                        f"{plan.symbol} target hit at {plan.target_price:.2f}",
                    )
                )
        else:
            if self._crossed_below(prior_price, current, plan.entry_price):
                alerts.append(
                    self._alert(
                        plan,
                        AlertType.ENTRY_TRIGGERED,
                        current,
                        now,
                        f"{plan.symbol} short entry triggered at {plan.entry_price:.2f}",
                    )
                )
            if current >= plan.stop_price:
                alerts.append(
                    self._alert(
                        plan,
                        AlertType.STOP_HIT,
                        current,
                        now,
                        f"{plan.symbol} stop hit at {plan.stop_price:.2f}",
                    )
                )
            if current <= plan.target_price:
                alerts.append(
                    self._alert(
                        plan,
                        AlertType.TARGET_HIT,
                        current,
                        now,
                        f"{plan.symbol} target hit at {plan.target_price:.2f}",
                    )
                )

        return alerts

    @staticmethod
    def _crossed_above(
        prior: float | None, current: float, level: float
    ) -> bool:
        if prior is None:
            return current >= level
        return prior < level <= current

    @staticmethod
    def _crossed_below(
        prior: float | None, current: float, level: float
    ) -> bool:
        if prior is None:
            return current <= level
        return prior > level >= current

    @staticmethod
    def _alert(
        plan: TradePlan,
        alert_type: AlertType,
        price: float,
        triggered_at,
        message: str,
    ) -> Alert:
        return Alert(
            symbol=plan.symbol,
            setup_name=plan.setup_name,
            alert_type=alert_type,
            message=message,
            triggered_at=triggered_at,
            reference_price=round(price, 4),
            plan=plan,
        )
