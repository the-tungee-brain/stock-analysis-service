# Deployment Runbook — Ranking + Portfolio + Risk

**Deploy** = new Docker image + restart API. **Ranking** = separate workflow only.

## Automation overview

| Trigger | Workflow | What runs |
|---------|----------|-----------|
| Push to `main` | `deploy.yml` | Build image, deploy container, import check — **no ranking** |
| Tue–Sat 07:30 UTC | `ranking-pipeline-vm.yml` | **daily** + portfolio |
| Sun 06:00 UTC | `ranking-pipeline-vm.yml` | **weekly** universe refresh |
| Manual (one-time) | `ranking-pipeline-vm.yml` → **bootstrap** | Universe → SPY → daily → portfolio |
| Manual | `ranking-pipeline-vm.yml` → **daily** / **bootstrap-resume** | Ad hoc |

Script on VM: `/home/ubuntu/ranking_pipeline_remote.sh` (copied on deploy and before each pipeline run)  
Logs: `/home/ubuntu/logs/ranking-bootstrap.log`, `ranking-daily.log`

---

## A. First-time data (bootstrap — once)

1. **Deploy** latest `main` (`deploy.yml`).

2. GitHub → **Actions** → **Ranking pipeline (VM)** → **Run workflow** → mode **`bootstrap`**.

3. Monitor on VM:
   ```bash
   tail -f /home/ubuntu/logs/ranking-bootstrap.log
   ```

4. Done when log shows `=== bootstrap complete ===` (marker:
   `/home/ubuntu/sas-ranking-persist/.pipeline_bootstrapped`).

5. Verify API (authenticated):
   ```bash
   curl -H "Authorization: Bearer $TOKEN" https://<host>/api/v1/rankings/top?limit=5
   curl -H "Authorization: Bearer $TOKEN" https://<host>/api/v1/portfolio/latest
   ```

If universe finished but ranking failed: run workflow mode **`bootstrap-resume`** (not full bootstrap).

---

## B. Daily ranking (ongoing)

**Automatic:** `ranking-pipeline-vm.yml` every **Tue–Sat 07:30 UTC**.

**Manual:** same workflow → **daily** (runs `scripts/run_ranking_daily.py` then `scripts/run_portfolio_with_risk.py`).

Deploy does **not** re-run ranking.

### Momentum Breakout scanner dependency

The default Momentum Breakout scan universe (`GET /api/v1/strategy/momentum-breakout/scan` without `symbols`) uses the **latest daily ranking run** in `data/ranking/ranking_pipeline.db` (`ranking_results` ordered by `rank`), intersected with local OHLCV parquet, then capped by `MB_SCAN_MAX_UNIVERSE` (default 500).

| Requirement | Why |
|-------------|-----|
| Complete **daily ranking before US market open** | Scanner expects a fresh run for the latest completed bar day |
| Persisted `ranking_pipeline.db` on the API volume | Same path as `RANKING_DB_PATH` / `data/ranking` mount |
| `scripts/run_ranking_daily.py` success | Writes `ranking_runs` + `ranking_results` used for universe ordering |

If ranking is **stale** (older than one trading day vs the latest bar, empty results, or no run created on the current trading day before the open), the scanner **falls back** to liquidity-sorted universe members and returns warning: `Ranking output is stale; scanner is using fallback universe.`

Diagnostics: `GET /api/v1/strategy/momentum-breakout/universe` (fields: `universeSource`, `selectionMethod`, `rankingRunId`, `warning`, …).

Config: `MB_SCAN_UNIVERSE_ORDER` — default `ranking_score` (`liquidity` \| `market_cap` \| `alphabetical` to override).

---

## C. Deploy (app only)

Push to `main` → `deploy.yml`:

- `-v .../data/raw` and `.../data/ranking` (never mount whole `/app/data`)
- `PYTHONPATH=/app`, `RANKING_MAX_WORKERS=4`, `YFINANCE_MIN_INTERVAL_SEC=0.4`
- `pipeline imports ok` — then nginx + health check

Existing Parquet/SQLite on the volume are unchanged.

### Oracle watchlist workspace migration

Existing Oracle databases created before watchlist workspace versioning need one additive
table migration before deploying the API image that reads/writes watchlist versions.
The script is idempotent and only creates `WATCHLIST_WORKSPACE` when it is missing;
it does not drop, truncate, rewrite, or backfill `WATCHLIST_FOLDER` or
`WATCHLIST_ITEM`.

Run on the production VM with the same Oracle credentials used by the API:

```bash
cd /path/to/stock-analysis
sqlplus -s "$POWERPOCKETDB_USER/$POWERPOCKETDB_PASSWORD@$POWERPOCKETDB_TP_TNS" <<'SQL'
whenever sqlerror exit sql.sqlcode
@app/sql/migrations/20260604_watchlist_workspace.sql
exit
SQL
```

It is safe to run the script more than once. Verify the table exists:

```sql
select table_name
from user_tables
where table_name = 'WATCHLIST_WORKSPACE';
```

Then deploy/restart the API. Old web and iOS clients can continue syncing without
`baseVersion`; optimistic concurrency is enforced only for clients that send it.

---

## D. Monitoring

| Check | Where |
|-------|--------|
| Deploy | Actions → Deploy SAS Server to OCI |
| Daily ranking | Actions → Ranking pipeline (VM) |
| Bootstrap progress | `tail -f /home/ubuntu/logs/ranking-bootstrap.log` |
| Daily progress | `tail -f /home/ubuntu/logs/ranking-daily.log` |

---

## E. Failure recovery

| Issue | Action |
|-------|--------|
| `No module named 'data.benchmarks'` | Redeploy with split volume mounts |
| `No feature rows` / short OHLCV | Workflow **bootstrap-resume** or **daily** after deploy with fix |
| `No data returned for SPY` | Wait 10 min → **bootstrap-resume** |
| Deploy Actions timeout | Expected if pipeline was wrongly on deploy; fixed — use **Ranking pipeline (VM)** |

---

## F. Clients

Web/iOS: `rankings/top`, `portfolio/latest`, `health` — see [`frontend_integration.md`](frontend_integration.md).
