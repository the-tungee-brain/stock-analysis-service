from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.intelligence_models import (
    OptionRollSuggestion,
    OptionsScorecard,
)

_PRECOMPUTED_MODEL_CONFIG = ConfigDict(populate_by_name=True)


class OptionLegOutcome(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    put_call: Literal["CALL", "PUT"] | None = Field(
        default=None, serialization_alias="putCall"
    )
    side: Literal["call", "put"] | None = None
    strike: float
    expiration: str
    contracts: float = 1.0
    days_to_expiration: int | None = Field(
        default=None, serialization_alias="daysToExpiration"
    )
    delta: float | None = None
    bid: float | None = None
    ask: float | None = None
    mark: float | None = None
    cash_per_contract: float | None = Field(
        default=None, serialization_alias="cashPerContract"
    )
    cash_direction: Literal["pay", "collect"] | None = Field(
        default=None, serialization_alias="cashDirection"
    )


class RollCashPicture(BaseModel):
    """Roll cash ledger: original premium through close/open legs to net cash."""

    model_config = _PRECOMPUTED_MODEL_CONFIG

    entry_premium_per_contract: float | None = Field(
        default=None, serialization_alias="entryPremiumPerContract"
    )
    close_cost_per_contract: float | None = Field(
        default=None, serialization_alias="closeCostPerContract"
    )
    open_collect_per_contract: float | None = Field(
        default=None, serialization_alias="openCollectPerContract"
    )
    roll_net_per_contract: float | None = Field(
        default=None, serialization_alias="rollNetPerContract"
    )
    net_cash_after_roll_per_contract: float | None = Field(
        default=None, serialization_alias="netCashAfterRollPerContract"
    )
    loss_on_closed_put_per_contract: float | None = Field(
        default=None, serialization_alias="lossOnClosedPutPerContract"
    )
    summary: str | None = None


class RollPathOutcome(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    close_leg: OptionLegOutcome = Field(serialization_alias="closeLeg")
    open_leg: OptionLegOutcome = Field(serialization_alias="openLeg")
    net_credit_per_share: float | None = Field(
        default=None, serialization_alias="netCreditPerShare"
    )
    net_credit_per_contract: float | None = Field(
        default=None, serialization_alias="netCreditPerContract"
    )
    is_net_credit: bool = Field(default=True, serialization_alias="isNetCredit")
    cash_picture: RollCashPicture | None = Field(
        default=None, serialization_alias="cashPicture"
    )


class ClosePathOutcome(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    cost_per_share: float | None = Field(
        default=None, serialization_alias="costPerShare"
    )
    cost_per_contract: float | None = Field(
        default=None, serialization_alias="costPerContract"
    )
    open_pnl: float | None = Field(default=None, serialization_alias="openPnl")


class HoldPathOutcome(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    days_to_expiration: int | None = Field(
        default=None, serialization_alias="daysToExpiration"
    )
    delta: float | None = None
    underlying_price: float | None = Field(
        default=None, serialization_alias="underlyingPrice"
    )
    in_the_money: bool | None = Field(default=None, serialization_alias="inTheMoney")
    assignment_note: str | None = Field(
        default=None, serialization_alias="assignmentNote"
    )


class HeldOptionDecisionDrivers(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    portfolio_weight_pct: float | None = Field(
        default=None, serialization_alias="portfolioWeightPct"
    )
    open_pnl: float | None = Field(default=None, serialization_alias="openPnl")
    open_pnl_pct: float | None = Field(default=None, serialization_alias="openPnlPct")
    entry_premium_per_share: float | None = Field(
        default=None, serialization_alias="entryPremiumPerShare"
    )
    entry_premium_per_contract: float | None = Field(
        default=None, serialization_alias="entryPremiumPerContract"
    )
    action_trigger: str | None = Field(default=None, serialization_alias="actionTrigger")


class ComparePathOption(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    path: Literal["roll", "close", "hold"]
    title: str
    lines: list[str] = Field(default_factory=list)


class HeldOptionOutcomes(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    drivers: HeldOptionDecisionDrivers
    current_leg: OptionLegOutcome = Field(serialization_alias="currentLeg")
    roll: RollPathOutcome | None = None
    roll_cash_picture: RollCashPicture | None = Field(
        default=None, serialization_alias="rollCashPicture"
    )
    close: ClosePathOutcome
    hold: HoldPathOutcome
    compare_paths: list[ComparePathOption] = Field(
        default_factory=list, serialization_alias="comparePaths"
    )


class SymbolAnalysisPrecomputed(BaseModel):
    model_config = _PRECOMPUTED_MODEL_CONFIG

    symbol: str
    underlying_price: float | None = Field(
        default=None, serialization_alias="underlyingPrice"
    )
    options_scorecard: OptionsScorecard | None = Field(
        default=None, serialization_alias="optionsScorecard"
    )
    roll_suggestions: list[OptionRollSuggestion] = Field(
        default_factory=list, serialization_alias="rollSuggestions"
    )
    held_option_outcomes: list[HeldOptionOutcomes] = Field(
        default_factory=list, serialization_alias="heldOptionOutcomes"
    )
