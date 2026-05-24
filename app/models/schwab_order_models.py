from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

OrderStatus = Literal[
    "AWAITING_PARENT_ORDER",
    "AWAITING_CONDITION",
    "AWAITING_STOP_CONDITION",
    "AWAITING_MANUAL_REVIEW",
    "ACCEPTED",
    "AWAITING_UR_OUT",
    "PENDING_ACTIVATION",
    "QUEUED",
    "WORKING",
    "REJECTED",
    "PENDING_CANCEL",
    "CANCELED",
    "PENDING_REPLACE",
    "REPLACED",
    "FILLED",
    "EXPIRED",
    "NEW",
    "AWAITING_RELEASE_TIME",
    "PENDING_ACKNOWLEDGEMENT",
    "PENDING_RECALL",
    "UNKNOWN",
]


class Instrument(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cusip: Optional[str] = None
    symbol: Optional[str] = None
    description: Optional[str] = None
    instrumentId: Optional[int] = None
    netChange: Optional[float] = None
    type: Optional[str] = None
    assetType: Optional[str] = None
    putCall: Optional[str] = None
    underlyingSymbol: Optional[str] = None


class OrderLeg(BaseModel):
    model_config = ConfigDict(extra="ignore")

    orderLegType: Optional[str] = None
    legId: Optional[int] = None
    instrument: Optional[Instrument] = None
    instruction: Optional[str] = None
    positionEffect: Optional[str] = None
    quantity: Optional[float] = None
    quantityType: Optional[str] = None
    divCapGains: Optional[str] = None
    toSymbol: Optional[str] = None


class ExecutionLeg(BaseModel):
    model_config = ConfigDict(extra="ignore")

    legId: Optional[int] = None
    price: Optional[float] = None
    quantity: Optional[float] = None
    mismarkedQuantity: Optional[float] = None
    instrumentId: Optional[int] = None
    time: Optional[datetime] = None


class OrderActivity(BaseModel):
    model_config = ConfigDict(extra="ignore")

    activityType: Optional[str] = None
    executionType: Optional[str] = None
    quantity: Optional[float] = None
    orderRemainingQuantity: Optional[float] = None
    executionLegs: Optional[List[ExecutionLeg]] = None


class SchwabOrder(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session: Optional[str] = None
    duration: Optional[str] = None
    orderType: Optional[str] = None
    cancelTime: Optional[datetime] = None
    complexOrderStrategyType: Optional[str] = None

    quantity: Optional[float] = None
    filledQuantity: Optional[float] = None
    remainingQuantity: Optional[float] = None

    requestedDestination: Optional[str] = None
    destinationLinkName: Optional[str] = None

    releaseTime: Optional[datetime] = None

    stopPrice: Optional[float] = None
    stopPriceLinkBasis: Optional[str] = None
    stopPriceLinkType: Optional[str] = None
    stopPriceOffset: Optional[float] = None
    stopType: Optional[str] = None

    priceLinkBasis: Optional[str] = None
    priceLinkType: Optional[str] = None
    price: Optional[float] = None

    taxLotMethod: Optional[str] = None

    orderLegCollection: Optional[List[OrderLeg]] = None

    activationPrice: Optional[float] = None
    specialInstruction: Optional[str] = None
    orderStrategyType: Optional[str] = None

    orderId: Optional[int] = None
    cancelable: Optional[bool] = None
    editable: Optional[bool] = None

    status: Optional[OrderStatus] = None
    enteredTime: Optional[datetime] = None
    closeTime: Optional[datetime] = None

    tag: Optional[str] = None
    accountNumber: Optional[int] = None

    orderActivityCollection: Optional[List[OrderActivity]] = None

    replacingOrderCollection: Optional[List[str]] = Field(default=None)
    childOrderStrategies: Optional[List[str]] = Field(default=None)

    statusDescription: Optional[str] = None
