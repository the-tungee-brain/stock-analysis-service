from typing import List, Optional, Literal
from pydantic import BaseModel
from datetime import datetime

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
    cusip: Optional[str]
    symbol: Optional[str]
    description: Optional[str]
    instrumentId: Optional[int]
    netChange: Optional[float]
    type: Optional[str]


class OrderLeg(BaseModel):
    orderLegType: Optional[str]
    legId: Optional[int]
    instrument: Optional[Instrument]
    instruction: Optional[str]  # BUY / SELL
    positionEffect: Optional[str]  # OPENING / CLOSING
    quantity: Optional[float]
    quantityType: Optional[str]
    divCapGains: Optional[str]
    toSymbol: Optional[str]


class ExecutionLeg(BaseModel):
    legId: Optional[int]
    price: Optional[float]
    quantity: Optional[float]
    mismarkedQuantity: Optional[float]
    instrumentId: Optional[int]
    time: Optional[datetime]


class OrderActivity(BaseModel):
    activityType: Optional[str]
    executionType: Optional[str]
    quantity: Optional[float]
    orderRemainingQuantity: Optional[float]
    executionLegs: Optional[List[ExecutionLeg]]


class SchwabOrder(BaseModel):
    session: Optional[str]
    duration: Optional[str]
    orderType: Optional[str]
    cancelTime: Optional[datetime]
    complexOrderStrategyType: Optional[str]

    quantity: Optional[float]
    filledQuantity: Optional[float]
    remainingQuantity: Optional[float]

    requestedDestination: Optional[str]
    destinationLinkName: Optional[str]

    releaseTime: Optional[datetime]

    stopPrice: Optional[float]
    stopPriceLinkBasis: Optional[str]
    stopPriceLinkType: Optional[str]
    stopPriceOffset: Optional[float]
    stopType: Optional[str]

    priceLinkBasis: Optional[str]
    priceLinkType: Optional[str]
    price: Optional[float]

    taxLotMethod: Optional[str]

    orderLegCollection: Optional[List[OrderLeg]]

    activationPrice: Optional[float]
    specialInstruction: Optional[str]
    orderStrategyType: Optional[str]

    orderId: Optional[int]
    cancelable: Optional[bool]
    editable: Optional[bool]

    status: Optional[OrderStatus]
    enteredTime: Optional[datetime]
    closeTime: Optional[datetime]

    tag: Optional[str]
    accountNumber: Optional[int]

    orderActivityCollection: Optional[List[OrderActivity]]

    replacingOrderCollection: Optional[List[str]]
    childOrderStrategies: Optional[List[str]]

    statusDescription: Optional[str]
