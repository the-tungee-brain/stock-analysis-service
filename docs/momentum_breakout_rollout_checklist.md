# Momentum Breakout rollout checklist

Educational trade-plan monitoring only. No brokerage execution or auto-trading.

## 1. Oracle DDL

Apply in order:

1. `app/sql/momentum_breakout_alert.sql` (alerts + events + context columns)
2. `app/sql/momentum_breakout_paper_trade.sql` (paper performance table)

Verify with launch readiness (`oracleAlertDdlApplied`, `oraclePaperDdlApplied`).

## 2. Environment variables

| Variable | Purpose | Suggested production |
|----------|---------|----------------------|
| `MB_ALERTS_ENABLED` | Master switch for alerts API/UI | `true` after validation |
| `MB_ALERT_CREATION_ENABLED` | Allow new persisted alerts | `false` during read-only phase |
| `MB_ALERT_NOTIFICATIONS_ENABLED` | Emit in-app notifications | `true` when alerts on |
| `MB_PAPER_ANALYTICS_ENABLED` | Paper performance API/UI | `true` after backfill |
| `MB_ALERT_STORE` | `oracle` / `sqlite` / `memory` | `oracle` |
| `MB_PAPER_TRADE_STORE` | Paper persistence backend | `oracle` |
| `MB_ALERT_SCHEDULER_ENABLED` | Background price refresh | `true` |
| `MB_ALERT_REFRESH_INTERVAL_SEC` | Poll interval (60–300) | `180` |
| `MB_ADMIN_TOKEN` | Admin metrics / diagnostics header | Set secret |
| `MB_LAUNCH_READINESS_PUBLIC` | Expose full readiness warnings | `false` in prod |
| `MB_PRODUCTION` or `ENV=production` | Production safety warnings | Set in prod |
| `MB_SCAN_MAX_UNIVERSE` | Max symbols evaluated per scan when `symbols` omitted | `500` |
| `MB_SCAN_UNIVERSE_ORDER` | Universe ordering override | `ranking_score` (default) |

## 2b. Daily ranking before market open

Momentum Breakout **does not** run its own universe rank. Default scan selection depends on the nightly jobs:

1. `scripts/run_ranking_daily.py` — OHLCV, features, cross-section rank → `ranking_pipeline.db`
2. `scripts/run_portfolio_with_risk.py` — portfolio layer (separate from scan universe)

**Operational rule:** The ranking workflow (GitHub Actions **Ranking pipeline (VM)** or equivalent cron) must **finish successfully before US market open** on trading days. If ranking is late or missing, the scanner still runs but uses a **liquidity fallback** universe and surfaces a stale-ranking warning on `GET /api/v1/strategy/momentum-breakout/universe`.

Verify after daily job:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  https://<host>/api/v1/strategy/momentum-breakout/universe
```

Expect `universeSource: daily_ranking_results`, no `warning`, and a recent `rankingRunId` / `rankingGeneratedAt`.

## 3. Paper-trade backfill

After DDL and before enabling paper analytics.

### Local

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

export POWERPOCKETDB_USER=...
export POWERPOCKETDB_PASSWORD=...
export POWERPOCKETDB_TP_TNS='(description= ...)'

# Preview missing rows (no writes)
.venv/bin/python scripts/backfill_momentum_breakout_paper_trades.py \
  --environment staging \
  --dry-run

# Apply backfill
.venv/bin/python scripts/backfill_momentum_breakout_paper_trades.py \
  --environment staging
```

Script flags:

| Flag | Description |
|------|-------------|
| `--environment` | `production` or `staging` — sets `MB_ALERT_STORE` / `MB_PAPER_TRADE_STORE` to `oracle` and `ENV` |
| `--dry-run` | Report `rows to create` without writing |
| `--limit` | Max alerts to scan (default 10000) |

Summary output:

- `alerts scanned`
- `rows created` (or `rows to create` in dry-run)
- `rows skipped` (paper row already exists)
- `rows failed`

Exit code is **non-zero** if `rows failed` > 0.

### GitHub Actions (manual)

Workflow: **Momentum Breakout paper-trade backfill** (`.github/workflows/momentum-breakout-backfill.yml`)

- Trigger: **workflow_dispatch** only (never on push)
- Inputs:
  - **environment:** `staging` or `production` (uses matching GitHub Environment for secrets)
  - **dry_run:** `true` to preview; `false` to write rows

**Required secrets** on each GitHub Environment (`staging`, `production`):

| Secret | Purpose |
|--------|---------|
| `POWERPOCKETDB_USER` | Oracle DB user (same as deploy) |
| `POWERPOCKETDB_PASSWORD` | Oracle DB password |
| `POWERPOCKETDB_TP_TNS` | Oracle connect descriptor (TNS) |

**Recommended run order:**

1. Run with `dry_run: true` on **staging** — confirm counts look reasonable
2. Run with `dry_run: false` on **staging** — verify `rows failed: 0`
3. Repeat dry-run then live on **production**

The job fails if the script exits non-zero (`rows failed` > 0).

Expect `rows created` ≥ alerts missing paper rows; `rows failed` = 0 on live runs.

## 4. Launch readiness

```http
GET /api/v1/strategy/momentum-breakout/launch-readiness
```

Optional header: `X-MB-Admin-Token: <MB_ADMIN_TOKEN>`

Confirm `ready: true` and no blocking warnings before widening rollout.

## 5. Feature flags (controlled rollout)

Phased example:

1. **Dark launch:** `MB_ALERTS_ENABLED=false` (default off in prod until ready)
2. **Read-only:** `MB_ALERTS_ENABLED=true`, `MB_ALERT_CREATION_ENABLED=false`
3. **Creation:** `MB_ALERT_CREATION_ENABLED=true`
4. **Notifications:** `MB_ALERT_NOTIFICATIONS_ENABLED=true`
5. **Paper analytics:** `MB_PAPER_ANALYTICS_ENABLED=true` (after backfill)

Clients poll:

```http
GET /api/v1/strategy/momentum-breakout/feature-status
```

## 6. Smoke tests

- [ ] Feature status returns expected flags
- [ ] Active/history alerts load for test user
- [ ] Trade-plan POST with `persistAlert: false` works when creation disabled
- [ ] Trade-plan POST with `persistAlert: true` blocked when creation disabled
- [ ] Manual refresh updates lifecycle during market hours
- [ ] Notifications list loads; new notifications appear when enabled
- [ ] Paper performance summary loads (when enabled)
- [ ] Launch readiness `ready: true`
- [ ] Admin metrics (`X-MB-Admin-Token`) shows counters

## 7. Monitoring

Structured logs (`momentum_breakout.ops`):

- `alert_created`, `risk_gate_blocked`, `entry_triggered`, `target_hit`, `stop_hit`, `expired`
- `notification_emitted`
- `scheduler_refresh_completed`, `scheduler_refresh_failed`
- `launch_readiness_failed`

Admin metrics:

```http
GET /api/v1/strategy/momentum-breakout/admin/metrics
```

## 8. Rollback

1. Set `MB_ALERTS_ENABLED=false` (hides API/UI immediately)
2. Set `MB_ALERT_SCHEDULER_ENABLED=false` (stops background refresh)
3. Optionally set `MB_ALERT_CREATION_ENABLED=false` for read-only mode instead of full off
4. Investigate logs and launch readiness warnings
5. No order cancellation required (paper monitoring only)

## 9. Web / iOS

- Web nav/bell: hidden when `alertsEnabled=false` (`NEXT_PUBLIC_MB_*` + feature-status)
- Direct route visits show “temporarily unavailable”
- iOS: DEBUG or `MB_LAUNCH_READINESS=1` for internal diagnostics panel
