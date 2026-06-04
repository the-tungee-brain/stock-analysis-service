# Momentum Breakout — Frontend Implementation Tasks

API reference: [momentum_breakout_alert_api.md](./momentum_breakout_alert_api.md)

Backend owns state and calculations. Frontend owns layout, visual design, interaction, and UX. Do not duplicate lifecycle rules client-side.

---

## Web

### Components

- [ ] **`MomentumBreakoutAlertsPanel`**
  - Container on strategy / research surface
  - Fetches `GET /alerts/active` on mount
  - Shows disclaimer from API
  - Empty state when `alerts.length === 0`
  - Error + retry for failed fetch

- [ ] **`AlertCard`**
  - Props: `MomentumBreakoutAlertDto` (TypeScript interface matching API)
  - Display: symbol, status badge, entry / stop / target, `riskReward`, `outcomeReturnPct` (history)
  - **Risk gate section:** if `riskGateAction` is `BLOCK` or `WARN` / `SIZE_DOWN`, list `riskGateReasons` (filter boilerplate disclaimers if desired)
  - Optional `nextActionMessage` as subdued helper text
  - Link to alert detail / lifecycle timeline from `lifecycleEvents` if shown

- [ ] **`AlertNotificationBell`**
  - Poll `GET /notifications?unreadOnly=true` (interval TBD, e.g. 60s in session)
  - Badge count = unread length
  - Dropdown / drawer list: `title`, `body`, `severity`, timestamp
  - On item click: `POST /notifications/{notificationId}/read`, navigate to related alert if `alertId` set
  - Use `severity` for icon/color only (map `info` | `watch` | `warning` | `critical`)

### Behavior

- [ ] **Lifecycle status badges**
  - Map `status` → label + color in frontend (e.g. `PENDING_ENTRY`, `OPEN`, `TARGET_HIT`, …)
  - Do not hard-code status transition rules

- [ ] **Risk gate blocked / warning**
  - `BLOCK`: card border or banner using `critical`-style treatment; show summarized reasons
  - `WARN` / `SIZE_DOWN`: warning banner; still show plan levels if present from trade-plan response

- [ ] **Poll or refresh active alerts**
  - Option A: poll `GET /alerts/active` every 1–5 min during market hours
  - Option B: user-triggered refresh calling `POST /alerts/refresh` then refetch active
  - Show last-updated timestamp in panel header

### TypeScript models (suggested)

```ts
export type AlertLifecycleStatus =
  | "PENDING_ENTRY" | "ENTRY_TRIGGERED" | "OPEN"
  | "TARGET_HIT" | "STOP_HIT" | "EXPIRED" | "CANCELLED";

export interface MomentumBreakoutAlertDto { /* mirror API camelCase */ }

export interface MomentumBreakoutNotificationDto { /* mirror API */ }
```

---

## iOS (SwiftUI)

### Models

- [ ] **Swift structs** matching API with `CodingKeys` for camelCase JSON
  - `MomentumBreakoutAlertDTO`
  - `MomentumBreakoutNotificationDTO`
  - `AlertRiskGateResultDTO`, `TradePlanLevelsDTO` (trade-plan endpoint)
  - `NotificationSeverity`: `info`, `watch`, `warning`, `critical`

### Views

- [ ] **`MomentumBreakoutAlertCard`**
  - Inputs: `MomentumBreakoutAlertDTO`
  - Symbol headline, status badge, three price rows (entry / stop / target)
  - `riskReward` and historical stats when non-nil
  - Risk reasons stack when `riskGateReasons` non-empty

- [ ] **Notification list view**
  - `GET /notifications` via existing API client
  - Row: `title`, `body`, relative `createdAt`, unread indicator
  - Swipe or tap to mark read (`POST .../read`)
  - Deep link to alert detail using `alertId` + embedded `alert` snapshot

### UX

- [ ] **Status badge colors**
  - Local `enum` mapping `status` → `Color` (e.g. pending = orange, open = blue, target = green, stop = red)
  - Keep mapping in SwiftUI theme / design system

- [ ] **Risk warning section**
  - Section header “Risk controls” when `riskGateAction != ALLOW`
  - Bullet list from `riskGateReasons`

- [ ] **Manual refresh**
  - Toolbar button → `POST /alerts/refresh` → refresh active list
  - `ProgressView` while in flight; surface `warnings` from refresh response if non-empty

### Networking

- [ ] Reuse authenticated `URLSession` / API service pattern from app
- [ ] Parse ISO-8601 dates for `createdAt`, `triggeredAt`, `exitAt`

---

## Out of scope (frontend)

- Order placement, brokerage actions, “buy now” CTAs
- Guaranteed-return copy
- Reimplementing lifecycle transitions or risk gate logic
