# Product Serving Architecture (Web + iOS)

Batch compute remains server-side; clients consume three stable v1 endpoints only.

## System architecture

```
┌─────────────────────────────────────────────────────────┐
│  Web (Next.js)          iOS (SwiftUI)                   │
│  poll 60–120s           poll + local cache              │
└───────────────────────────┬─────────────────────────────┘
                            │ HTTPS + auth
                            ▼
┌─────────────────────────────────────────────────────────┐
│  API Layer (app/api/product/)                           │
│  GET /api/v1/rankings/top                               │
│  GET /api/v1/portfolio/latest                           │
│  GET /api/v1/health                                     │
└───────────────────────────┬─────────────────────────────┘
                            │ read SQLite / Parquet snapshots
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Portfolio + Risk Engine (batch)                        │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Ranking Engine + ML (batch)                            │
└───────────────────────────┬─────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Data + Regime + Universe (batch)                       │
└─────────────────────────────────────────────────────────┘
```

## Deployment timeline (first run → steady state)

| Phase | When | Actions |
|-------|------|---------|
| T0 | Day 0 | Deploy API, init DB schema, SPY + universe weekly |
| T1 | Day 0–1 | OHLCV backfill, `run_ranking_daily.py`, `run_portfolio_with_risk.py` |
| T2 | Day 1 | Verify `/api/v1/health` = `ok`, clients poll rankings + portfolio |
| T3 | Daily | Cron 02:00–03:30 UTC pipeline; API serves snapshots by 03:30 |
| T4 | Weekly | Universe refresh, optional ML retrain, drift checks |

## Request/response examples (v1)

### `GET /api/v1/rankings/top?limit=20`

```json
{
  "api_version": "v1",
  "timestamp": "2026-06-02T07:30:12+00:00",
  "run_id": "2026-06-02T073012-a1b2c3d4",
  "as_of_date": "2026-06-01",
  "regime_id": "risk_on_trend",
  "items": [
    {
      "symbol": "NVDA",
      "rank": 1,
      "final_score": 0.82,
      "ml_probability": 0.71,
      "expected_excess_return": 0.018
    }
  ]
}
```

### `GET /api/v1/portfolio/latest`

```json
{
  "api_version": "v1",
  "timestamp": "2026-06-02T08:00:00+00:00",
  "portfolio_id": "2026-06-02T080000-pf-e5f6",
  "ranking_run_id": "2026-06-02T073012-a1b2c3d4",
  "as_of_date": "2026-06-01",
  "sizing_mode": "volatility_adjusted",
  "holdings": [
    {
      "symbol": "NVDA",
      "weight": 0.08,
      "score_contribution": 0.00144,
      "final_score": 0.82,
      "expected_excess_return": 0.018
    }
  ],
  "metrics": {
    "expected_return_5d": 0.012,
    "expected_excess_5d": 0.009,
    "volatility": 0.14,
    "beta_vs_spy": 0.95,
    "correlation_risk_score": 0.42,
    "sector_breakdown": { "Technology": 0.35, "Health": 0.22 },
    "turnover_estimate": 0.18,
    "concentration_hhi": 0.09
  },
  "risk_layer": {
    "portfolio_beta": 0.95,
    "portfolio_volatility": 0.14,
    "target_volatility": 0.12,
    "correlation_risk_score": 0.42,
    "sector_breakdown": { "Technology": 0.35 },
    "vol_scale_factor": 0.86
  },
  "top_contributors": []
}
```

### `GET /api/v1/health`

```json
{
  "api_version": "v1",
  "last_pipeline_run_time": "2026-06-02T08:00:00+00:00",
  "universe_size": 1842,
  "last_successful_ranking_run": "2026-06-02T073012-a1b2c3d4",
  "last_successful_portfolio_run": "2026-06-02T080000-pf-e5f6",
  "system_status": "ok",
  "last_ranking_run_at": "2026-06-02T07:30:12+00:00",
  "last_portfolio_run_at": "2026-06-02T08:00:00+00:00",
  "regime_id": "risk_on_trend"
}
```

## Failure fallback (read path)

| Failure | Client behavior |
|---------|-----------------|
| Ranking batch fails | API serves **last** `ranking_runs` row (omit `run_id`) |
| Portfolio batch fails | Previous `portfolio_snapshots` row remains until new success |
| ML missing | Ranking still has `composite` backend rows; `ml_probability` may be null |
| Health `degraded` | Show stale badge; keep polling |
