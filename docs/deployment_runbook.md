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

**Manual:** same workflow → **daily**.

Deploy does **not** re-run ranking.

---

## C. Deploy (app only)

Push to `main` → `deploy.yml`:

- `-v .../data/raw` and `.../data/ranking` (never mount whole `/app/data`)
- `PYTHONPATH=/app`, `RANKING_MAX_WORKERS=4`, `YFINANCE_MIN_INTERVAL_SEC=0.4`
- `pipeline imports ok` — then nginx + health check

Existing Parquet/SQLite on the volume are unchanged.

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
