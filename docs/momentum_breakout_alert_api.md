# Momentum Breakout Alert API

Educational trade-plan alerts only. Not investment advice. No orders are placed.

All endpoints require authentication (`Authorization: Bearer <token>`) under `/api/v1`.

Camel-case JSON field names are used in responses (`populate_by_name`).

---

## 1. Active Alerts

**`GET /api/v1/strategy/momentum-breakout/alerts/active`**

Returns persisted alerts in active lifecycle states (`PENDING_ENTRY`, `ENTRY_TRIGGERED`, `OPEN`).

### Response example

```json
{
  "disclaimer": "Educational alert tracking only. Not investment advice. No orders are placed.",
  "alerts": [
    {
      "alertId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "symbol": "NVDA",
      "setupName": "momentum_breakout",
      "direction": "LONG",
      "status": "OPEN",
      "createdAt": "2024-06-01T15:00:00Z",
      "signalDate": "2024-06-01",
      "entryPrice": 100.0,
      "stopPrice": 95.0,
      "targetPrice": 110.0,
      "riskReward": 2.0,
      "entryIsStop": true,
      "expiresAt": "2024-06-02T23:59:59Z",
      "triggeredAt": "2024-06-02T14:00:00Z",
      "exitAt": null,
      "exitPrice": null,
      "outcomeReturnPct": null,
      "riskGateAction": "ALLOW",
      "riskGateReasons": [
        "Educational trade plan alert only — not investment advice. No orders are placed."
      ],
      "priority": "HIGH",
      "historicalWinRate": 0.42,
      "historicalProfitFactor": 1.35,
      "historicalTotalTrades": 136,
      "nextActionMessage": "Track price versus stop and target; monitoring only.",
      "lifecycleEvents": [
        {
          "eventId": "e1",
          "eventType": "CREATED",
          "fromStatus": null,
          "toStatus": "PENDING_ENTRY",
          "price": null,
          "recordedAt": "2024-06-01T15:00:00Z",
          "message": "Alert created for NVDA momentum_breakout."
        },
        {
          "eventId": "e2",
          "eventType": "ENTRY_TRIGGERED",
          "fromStatus": "PENDING_ENTRY",
          "toStatus": "ENTRY_TRIGGERED",
          "price": 101.0,
          "recordedAt": "2024-06-02T14:00:00Z",
          "message": "Entry triggered at 100.0000."
        }
      ]
    }
  ]
}
```

### Client notes

| Field | Use |
|-------|-----|
| `status` | Lifecycle badge (map colors in frontend) |
| `riskGateAction` / `riskGateReasons` | Blocked/warning UI |
| `priority` | Sorting emphasis (`HIGH`, `MEDIUM`, `LOW`) |
| `nextActionMessage` | Optional helper text; clients may override with local copy |
| `riskReward`, `historical*` | Display-only stats from backend |

**Refresh:** `POST /api/v1/strategy/momentum-breakout/alerts/refresh` (manual, ignores market-hours guard).

**Cancel:** `POST /api/v1/strategy/momentum-breakout/alerts/{alertId}/cancel` — active alerts only (`PENDING_ENTRY`, `ENTRY_TRIGGERED`, `OPEN`). Returns updated alert DTO. `409` if already terminal. Records lifecycle event `CANCELLED` with message `Alert cancelled by user.` Updates paper-trade row when present. No brokerage orders.

---

## 2. Alert History

**`GET /api/v1/strategy/momentum-breakout/alerts/history?limit=100`**

Same alert object shape as active alerts, including terminal states (`TARGET_HIT`, `STOP_HIT`, `EXPIRED`, `CANCELLED`).

### Response example

```json
{
  "disclaimer": "Educational alert tracking only. Not investment advice. No orders are placed.",
  "alerts": [
    {
      "alertId": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
      "symbol": "NVDA",
      "setupName": "momentum_breakout",
      "direction": "LONG",
      "status": "TARGET_HIT",
      "createdAt": "2024-06-01T15:00:00Z",
      "signalDate": "2024-06-01",
      "entryPrice": 100.0,
      "stopPrice": 95.0,
      "targetPrice": 110.0,
      "riskReward": 2.0,
      "entryIsStop": true,
      "expiresAt": "2024-06-02T23:59:59Z",
      "triggeredAt": "2024-06-02T14:00:00Z",
      "exitAt": "2024-06-03T14:00:00Z",
      "exitPrice": 110.0,
      "outcomeReturnPct": 0.1,
      "riskGateAction": "ALLOW",
      "riskGateReasons": [],
      "priority": "HIGH",
      "historicalWinRate": 0.42,
      "historicalProfitFactor": 1.35,
      "historicalTotalTrades": 136,
      "nextActionMessage": "Review outcome in alert history.",
      "lifecycleEvents": []
    }
  ]
}
```

---

## 3. Notifications

**`GET /api/v1/strategy/momentum-breakout/notifications?unreadOnly=false&limit=100`**

**`POST /api/v1/strategy/momentum-breakout/notifications/{notificationId}/read`**

### List response example

```json
{
  "disclaimer": "Educational trade plan notifications only. Not investment advice. No orders are placed.",
  "notifications": [
    {
      "notificationId": "n1a2b3c4-d5e6-7890-abcd-ef1234567890",
      "eventType": "EntryTriggered",
      "title": "NVDA trade plan entry triggered",
      "body": "NVDA momentum_breakout entry triggered at $100.00. Stop: $95.00. Target: $110.00.",
      "severity": "watch",
      "nextActionMessage": "Track price versus stop and target; monitoring only.",
      "symbol": "NVDA",
      "alertId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "read": false,
      "createdAt": "2024-06-02T14:00:05Z",
      "alert": {
        "alertId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "symbol": "NVDA",
        "setupName": "momentum_breakout",
        "direction": "LONG",
        "status": "OPEN",
        "createdAt": "2024-06-01T15:00:00Z",
        "signalDate": "2024-06-01",
        "entryPrice": 100.0,
        "stopPrice": 95.0,
        "targetPrice": 110.0,
        "riskReward": 2.0,
        "entryIsStop": true,
        "expiresAt": "2024-06-02T23:59:59Z",
        "triggeredAt": "2024-06-02T14:00:00Z",
        "exitAt": null,
        "exitPrice": null,
        "outcomeReturnPct": null,
        "riskGateAction": "ALLOW",
        "riskGateReasons": [],
        "priority": "HIGH",
        "historicalWinRate": 0.42,
        "historicalProfitFactor": 1.35,
        "historicalTotalTrades": 136,
        "nextActionMessage": "Track price versus stop and target; monitoring only.",
        "lifecycleEvents": []
      }
    }
  ]
}
```

### `eventType` values

| `eventType` | Typical `severity` |
|-------------|-------------------|
| `AlertCreated` | `info` |
| `EntryTriggered` | `watch` |
| `TargetHit` | `info` |
| `StopHit` | `warning` |
| `Expired` | `info` |
| `BlockedByRiskGate` | `critical` |
| `WarningByRiskGate` | `warning` |

### Mark read response

```json
{
  "disclaimer": "Educational trade plan notifications only. Not investment advice. No orders are placed.",
  "notification": {
    "notificationId": "n1a2b3c4-d5e6-7890-abcd-ef1234567890",
    "eventType": "BlockedByRiskGate",
    "title": "NVDA alert blocked by risk controls",
    "body": "momentum_breakout alert blocked by risk controls: max open positions reached.",
    "severity": "critical",
    "nextActionMessage": "No further action for this cancelled alert.",
    "symbol": "NVDA",
    "alertId": null,
    "read": true,
    "createdAt": "2024-06-02T14:00:05Z",
    "alert": { "...": "same MomentumBreakoutAlertDto shape" }
  }
}
```

---

## 4. Trade Plan Alert (evaluate + optional persist)

**`POST /api/v1/strategy/momentum-breakout/trade-plan-alert`**

Evaluates the current setup, applies risk gate, returns plan levels and optional persisted alert id.

### Request example

```json
{
  "symbol": "NVDA",
  "openTrades": [],
  "recentClosed": [],
  "persistAlert": true
}
```

### Response example (plan available, allowed)

```json
{
  "disclaimer": "Educational trade plan information only. Not investment advice. Tomcrest does not place orders or manage your portfolio.",
  "planAvailable": true,
  "plan": {
    "symbol": "NVDA",
    "setupName": "momentum_breakout",
    "direction": "LONG",
    "entryPrice": 100.0,
    "stopPrice": 95.0,
    "targetPrice": 110.0,
    "riskReward": 2.0,
    "confidenceScore": 0.72,
    "entryIsStop": true
  },
  "historicalStats": {
    "totalTrades": 136,
    "winRatePct": 42.0,
    "profitFactor": 1.35,
    "averageHoldingDays": 8.2
  },
  "riskGate": {
    "allowed": true,
    "action": "ALLOW",
    "reasons": [
      "Educational trade plan alert only — not investment advice. No orders are placed."
    ],
    "recommendedPositionRiskPct": 0.01,
    "maxNotionalUsd": 5000.0,
    "alertPriority": "HIGH",
    "educationalOnly": true
  },
  "priceAlerts": [],
  "alertId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "lifecycleStatus": "PENDING_ENTRY"
}
```

### Response example (blocked by risk gate)

```json
{
  "disclaimer": "Educational trade plan information only. Not investment advice. Tomcrest does not place orders or manage your portfolio.",
  "planAvailable": true,
  "plan": {
    "symbol": "NVDA",
    "setupName": "momentum_breakout",
    "direction": "LONG",
    "entryPrice": 100.0,
    "stopPrice": 95.0,
    "targetPrice": 110.0,
    "riskReward": 2.0,
    "confidenceScore": 0.72,
    "entryIsStop": true
  },
  "historicalStats": { "totalTrades": 136, "winRatePct": 42.0, "profitFactor": 1.35, "averageHoldingDays": 8.2 },
  "riskGate": {
    "allowed": false,
    "action": "BLOCK",
    "reasons": [
      "Max open positions reached (5/5 active momentum_breakout trades).",
      "Educational trade plan alert only — not investment advice. No orders are placed."
    ],
    "recommendedPositionRiskPct": 0.0,
    "maxNotionalUsd": null,
    "alertPriority": "LOW",
    "educationalOnly": true
  },
  "priceAlerts": [],
  "alertId": null,
  "lifecycleStatus": null
}
```

A `BlockedByRiskGate` notification is emitted when `allowed` is false (no alert persisted).

---

## Related endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/strategy/momentum-breakout/alerts/{alertId}/price-update` | Manual price push (testing / integrations) |
| `POST` | `/strategy/momentum-breakout/alerts/refresh` | Refresh active alerts for current user |

---

## Single-symbol setup check

**`GET /api/v1/strategy/momentum-breakout/check/{symbol}`**

Checks one ticker against Momentum Breakout rules (same engine as the market scan).

| `status` | Meaning |
|----------|---------|
| `TRADABLE_BREAKOUT` | Valid setup + passes quality/safety filters |
| `REJECTED_BREAKOUT` | Valid setup but failed tradable filters |
| `NO_BREAKOUT_SETUP` | Does not meet breakout criteria (`failedSetupRules` in plain English) |
| `DATA_UNAVAILABLE` | No local OHLCV for symbol |

`canTrackBreakoutPlan` is true for tradable breakouts when alert creation is enabled; for `REJECTED_BREAKOUT` it is true only when `riskGate.allowed` is true.

---

## Custom educational plan (not Momentum Breakout)

**`POST /api/v1/strategy/custom-trade-plan`**

```json
{ "symbol": "META", "direction": "LONG", "accountEquityUsd": null }
```

Returns `setupName: "Custom Trade Plan"`, entry/stop/target (2R), `warnings`, `educationalOnly: true`. Does not create a Momentum Breakout alert.

---

## Scan universe diagnostics

**`GET /api/v1/strategy/momentum-breakout/universe`**

Describes which symbols the scanner evaluates when `symbols` is omitted on `/strategy/momentum-breakout/scan`.

Default ordering uses the **latest daily ranking run** (`ranking_results` by `rank` / `final_score`). If that output is stale, the API falls back to liquidity-sorted `universe_members` and sets `warning`.

| Field | Meaning |
|-------|---------|
| `universeSource` | `daily_ranking_results` or fallback/config source |
| `selectionMethod` | Active sort description |
| `rankingRunId` | Latest `ranking_runs.run_id` when present |
| `rankingSnapshotId` | Universe snapshot id tied to the run |
| `rankingGeneratedAt` | `ranking_runs.created_at` |
| `totalRankedSymbols` | Rows in `ranking_results` for the latest run |
| `totalEligibleSymbols` | Symbols with local OHLCV, before cap |
| `scanCap` | `MB_SCAN_MAX_UNIVERSE` (default 500) |
| `symbolsScanned` | Count after cap |
| `excludedByCap` | Eligible symbols not scanned due to cap |
| `first50ScannedSymbols` | First 50 in scan order |
| `topExcludedSample` | Next symbols after the cap (up to 20) |
| `warning` | Stale-ranking message when fallback is used |

Env: `MB_SCAN_UNIVERSE_ORDER` (`ranking_score` \| `liquidity` \| `market_cap` \| `alphabetical`).

---

## Ownership split

| Layer | Responsibility |
|-------|----------------|
| **Backend** | Lifecycle state, prices, risk gate, historical stats, R-multiple, notifications, `severity` / `nextActionMessage` hints |
| **Web / iOS** | Layout, typography, colors, badges, polling UX, empty states, navigation |
