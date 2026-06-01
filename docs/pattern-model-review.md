# Pattern model review workflow

Use this checklist before promoting a new pattern model to production.

## 1. Backtest the production portfolio

Validate ranking portfolio performance on the TOP20 universe:

```bash
.venv/bin/python scripts/run_phase3_backtest.py --universe top20
```

Review the compact summary:

- Portfolio **Sharpe** should stay competitive with Phase 2 ranking baseline.
- **Max drawdown** should remain acceptable vs buy-and-hold.
- Concentration and rebalance diagnostics should not show persistent drift.

For Phase 5 minimal-model comparison:

```bash
.venv/bin/python scripts/run_phase5_audit.py --skip-data-prep
```

Confirm **Model C** (relative strength + trend, 11 features) remains the production baseline.

## 2. Train production artifacts

When backtest results look acceptable, train Model C on TOP20:

```bash
.venv/bin/python scripts/train_tradeable_model.py
```

This runs download → features → XGBoost with:

- Universe: `top20` (production training panel)
- Model: **C** — relative strength + trend (11 features)
- Labels: `binary_outperform_spy`
- Class weights: enabled
- Portfolio metadata: ranking strategy, top 10, 5d rebalance/hold, 15% max weight

Artifacts land in `artifacts/` (or `PATTERN_ARTIFACT_DIR`).

## 3. Smoke-test locally

```bash
.venv/bin/python -m pytest tests/test_pattern_api_route.py tests/test_pattern_forecast_service.py tests/test_pattern_train_and_save.py -q
curl -sS -H "Authorization: Bearer $TOKEN" \
  "$API_BASE/api/v1/pattern/health" | jq .
curl -sS -H "Authorization: Bearer $TOKEN" \
  "$API_BASE/api/v1/pattern/predict?symbol=MSFT" | jq .
```

Confirm `/predict` returns `rankingScore`, `upProb`, `inTrainingUniverse`, RS/trend `indicators`, and `portfolioStrategy`.

Research overview / intelligence should expose the same forecast as `patternForecast`:

```bash
curl -sS -H "Authorization: Bearer $TOKEN" \
  "$API_BASE/api/v1/research/intelligence?symbol=MSFT" | jq .patternForecast
```

## 4. Deploy

**Manual:** copy `model_xgb.joblib` and `model_meta.json` to the VM artifact dir and restart `sas-server`.

**CI:** GitHub Actions → **Train Pattern Model** (weekly cron or manual dispatch). The workflow trains Model C on TOP20 and SCPs artifacts to `/home/ubuntu/sas-pattern-artifacts`, then restarts the container.

## 5. Post-deploy checks

See [deploy-smoke.md](./deploy-smoke.md#pattern-model).

Frontend (my-pocket) should render the research overview ranking card using `rankingScore` and `portfolioStrategy` — not threshold `tradeSignal` alone.

## Config reference

| Setting | Production value |
|---------|------------------|
| Model | **C** — relative strength + trend |
| Features | 11 (RS vs SPY + trend) |
| Training universe | `top20` |
| Label scheme | `binary_outperform_spy` |
| Class weights | on |
| Portfolio strategy | ranking, top 10, 5d rebalance |
| Max position weight | 15% |
| Legacy min P(up) | 0.65 (informational only) |

Shared constants live in `models/pattern_production.py`.
