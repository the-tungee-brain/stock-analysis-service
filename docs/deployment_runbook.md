# Deployment Runbook — Ranking + Portfolio + Risk

Post-deployment operations for the batch pipeline and product API (`/api/v1/rankings`, `/api/v1/portfolio`, `/api/v1/health`).

---

## A. Initial deployment

1. **Deploy backend** (FastAPI + gunicorn/uvicorn) with env:
   - `RANKING_DB_PATH` → persistent volume for `data/ranking/ranking_pipeline.db`
   - `RANKING_MODEL_BACKEND` → `xgboost` or `composite`
   - `PORTFOLIO_SIZING_MODE` → `volatility_adjusted` (recommended)
   - `PORTFOLIO_TARGET_VOL` → `0.12`–`0.15`

2. **Initialize schema** — first API or pipeline start creates SQLite tables via existing stores.

3. **Universe (weekly job)**:
   ```bash
   python scripts/run_ranking_universe_weekly.py
   ```
   Dev smoke: `--max-candidates 100`

4. **OHLCV backfill** — ensure SPY + universe symbols:
   ```bash
   python data/download.py --symbols SPY
   python scripts/run_ranking_daily.py  # will incremental-fill universe
   ```

5. **First ranking run**:
   ```bash
   python scripts/run_ranking_daily.py
   ```

6. **First portfolio + risk**:
   ```bash
   python scripts/run_portfolio_with_risk.py
   ```

7. **Verify API** (authenticated):
   ```bash
   curl -H "Authorization: Bearer $TOKEN" https://<host>/api/v1/health
   curl -H "Authorization: Bearer $TOKEN" https://<host>/api/v1/rankings/top?limit=5
   curl -H "Authorization: Bearer $TOKEN" https://<host>/api/v1/portfolio/latest
   ```
   Expect `system_status: "ok"` and non-empty rankings/portfolio.

---

## B. Daily operations (cron, trading days)

| UTC | Job | Command |
|-----|-----|---------|
| 02:00 | OHLCV incremental | `python scripts/run_ranking_daily.py` (step 1–2 only if split; full script OK) |
| 02:30 | Ranking + regime | `python scripts/run_ranking_daily.py` |
| 03:00 | Portfolio + risk | `python scripts/run_portfolio_with_risk.py` |
| 03:15 | Portfolio backtest (optional) | included in `run_portfolio_with_risk.py` |
| 03:30 | **API ready** | Clients poll; health should show fresh timestamps |

GitHub Actions reference: `.github/workflows/ranking-daily.yml` (extend with portfolio step in your scheduler).

---

## C. Weekly operations

| Task | Command / check |
|------|-----------------|
| Refresh universe | `python scripts/run_ranking_universe_weekly.py` |
| Feature warmup | Full re-run if universe grew >20% |
| ML retrain (optional) | `python scripts/train_ranking_model.py --backend xgboost` |
| Drift validation | Compare hit rate / Sharpe from `portfolio_backtest_metrics` week-over-week |
| Regime sanity | `regime_id` distribution via `market_regime_daily` |

---

## D. Monitoring checklist

| Metric | Target | Source |
|--------|--------|--------|
| Ranking run success | 100% trading days | Cron logs + `ranking_runs` row/day |
| Portfolio run success | 100% trading days | `portfolio_snapshots` row/day |
| API latency | <200ms p95 | APM on `/api/v1/*` |
| Universe size | Stable ±10% | `/api/v1/health` → `universe_size` |
| Turnover spikes | <50% daily | `portfolio.metrics.turnover_estimate` |
| Regime distribution | No weeks stuck in `risk_off` only | `health.regime_id` history |
| Health status | `ok` during market week | `/api/v1/health` |

**Degraded** if ranking age >36h; **failing** if >72h or DB missing.

---

## E. Failure recovery

| Scenario | Action |
|----------|--------|
| Ranking fails | API serves **last** successful `ranking_runs` (default). Fix logs, re-run `run_ranking_daily.py`. |
| Portfolio fails | **Previous** `portfolio_snapshots` remains; clients keep last weights. Re-run `run_portfolio_with_risk.py`. |
| ML model missing | Set `RANKING_MODEL_BACKEND=composite` or train artifacts under `artifacts/ranking_model/`. |
| SPY data missing | `download_and_store_symbol SPY` then re-run daily. |
| DB corrupt | Restore `ranking_pipeline.db` backup; replay daily from last good OHLCV. |

---

## F. Client contract reminder

- Web/iOS: only `rankings/top`, `portfolio/latest`, `health`.
- No on-request feature computation.
- Use `api_version` field for forward compatibility.

See [`frontend_integration.md`](frontend_integration.md) and [`product_serving_architecture.md`](product_serving_architecture.md).
