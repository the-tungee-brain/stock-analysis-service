# Deployment Runbook — Ranking + Portfolio + Risk

All pipeline steps run **automatically on the OCI VM** inside `sas-server` via GitHub Actions (no manual `docker exec`).

## Automation overview

| Trigger | Workflow | What runs |
|---------|----------|-----------|
| Push to `main` | `deploy.yml` | Build image, deploy container, import check, then **daily** or **bootstrap** (background) |
| Tue–Sat 07:30 UTC | `ranking-pipeline-vm.yml` | **daily** + portfolio |
| Sun 06:00 UTC | `ranking-pipeline-vm.yml` | **weekly** universe refresh |
| Manual | `ranking-pipeline-vm.yml` → Run workflow | `daily`, `weekly`, or `bootstrap` |

Script on VM: `/home/ubuntu/ranking_pipeline_remote.sh`  
Logs: `/home/ubuntu/logs/ranking-bootstrap.log`, `ranking-daily.log`

---

## A. Initial deployment

1. **Push to `main`** — `deploy.yml` runs tests, builds Docker image, deploys `sas-server` with:
   - `-v .../data/raw:/app/data/raw` and `.../data/ranking:/app/data/ranking` (not the whole `/app/data` tree — that breaks `import data.benchmarks`)
   - `PYTHONPATH=/app`
   - `ranking_pipeline` + `scripts` in the image

2. **First deploy** automatically starts **bootstrap in background** (universe → SPY → ranking → portfolio). Monitor:
   ```bash
   ssh ubuntu@<vm> tail -f /home/ubuntu/logs/ranking-bootstrap.log
   ```

3. **When bootstrap finishes**, marker file exists:
   `/home/ubuntu/sas-ranking-persist/.pipeline_bootstrapped`

4. **Verify API** (authenticated):
   ```bash
   curl -H "Authorization: Bearer $TOKEN" https://<host>/api/v1/health
   curl -H "Authorization: Bearer $TOKEN" https://<host>/api/v1/rankings/top?limit=5
   curl -H "Authorization: Bearer $TOKEN" https://<host>/api/v1/portfolio/latest
   ```

Manual bootstrap (only if automation failed):

```bash
docker exec -w /app sas-server python scripts/run_ranking_universe_weekly.py
# … see scripts/vm/ranking_pipeline_remote.sh bootstrap
```

---

## B. Daily operations

**Automated** — no action required. `ranking-pipeline-vm.yml` at 07:30 UTC Tue–Sat, plus incremental **daily** after each deploy.

---

## C. Weekly operations

**Automated** — Sunday 06:00 UTC universe refresh via `ranking-pipeline-vm.yml`.

Optional ML retrain: `train-pattern-model.yml` or `scripts/train_ranking_model.py` on VM.

---

## D. Monitoring

| Check | Where |
|-------|--------|
| Deploy succeeded | GitHub Actions → Deploy SAS Server to OCI |
| Pipeline imports | Deploy log: `pipeline imports ok` |
| Bootstrap progress | `tail -f /home/ubuntu/logs/ranking-bootstrap.log` |
| Daily runs | GitHub → Ranking pipeline (VM) |
| API health | `GET /api/v1/health` |

---

## E. Failure recovery

| Issue | Action |
|-------|--------|
| `ModuleNotFoundError: ranking_pipeline` | Redeploy latest image from `main` |
| `No module named 'data.benchmarks'` (Gunicorn crash) | Whole-volume mount shadowed `/app/data`. Redeploy with split mounts (`raw` + `ranking` only) |
| Log: `ranking_pipeline not in image` but deploy is new | Log may be **stale** from an old bootstrap. Verify imports (below), then **re-run bootstrap** |
| Bootstrap stuck | SSH → log file; re-run workflow **bootstrap** |
| Ranking fails | API serves last run; fix and run workflow **daily** |
| Portfolio fails | Previous snapshot remains; run workflow **daily** |

### Verify container (on VM)

```bash
docker exec -w /app sas-server python -c "import ranking_pipeline, data.download; print('ok')"
docker exec sas-server sh -c 'echo PYTHONPATH=$PYTHONPATH; test -d /app/ranking_pipeline && echo ranking_pipeline present'
```

If `ok` prints, restart bootstrap (overwrites the old log):

```bash
nohup /home/ubuntu/ranking_pipeline_remote.sh bootstrap > /home/ubuntu/logs/ranking-bootstrap.log 2>&1 &
tail -f /home/ubuntu/logs/ranking-bootstrap.log
```

Or GitHub Actions → **Ranking pipeline (VM)** → mode **bootstrap**.

---

## F. Clients

Web/iOS: `rankings/top`, `portfolio/latest`, `health` only — see [`frontend_integration.md`](frontend_integration.md).
