# Pattern model review workflow

Use this checklist before promoting a new pattern model to production.

## 1. Backtest the tradeable universe

Validate walk-forward performance with the production strategy config:

```bash
.venv/bin/python scripts/run_tradeable_backtest.py
```

Review the compact summary:

- Strategy **profit factor** should stay above ~1.3.
- Per-symbol **Sharpe** should be ≥ 1.0 for names you intend to trade.
- Compare strategy **Sharpe / max drawdown** vs buy-and-hold; weak aggregate numbers mean the model is not ready to drive live signals.

For a full report (per-window tables, recommended symbols):

```bash
.venv/bin/python scripts/run_tradeable_backtest.py --full-report
```

Trial extra symbols before adding them permanently:

```bash
.venv/bin/python scripts/run_tradeable_backtest.py --extra-symbols GOOGL JNJ
```

## 2. Train production artifacts

When backtest results look acceptable, train on the same universe and label scheme:

```bash
.venv/bin/python scripts/train_tradeable_model.py
```

This runs download → features → XGBoost with:

- Universe: `UNIVERSE_TRADEABLE_V1` (`COST`, `JPM`, `MSFT`, `NVDA`, `SPY`)
- Labels: `binary_updown`
- Class weights: enabled
- Metadata: `min_up_prob=0.65`, `universe=tradeable_v1`

Artifacts land in `artifacts/` (or `PATTERN_ARTIFACT_DIR`).

## 3. Smoke-test locally

```bash
.venv/bin/python -m pytest tests/test_pattern_api_route.py tests/test_pattern_train_and_save.py -q
curl -sS -H "Authorization: Bearer $TOKEN" \
  "$API_BASE/api/v1/pattern/health" | jq .
curl -sS -H "Authorization: Bearer $TOKEN" \
  "$API_BASE/api/v1/pattern/predict?symbol=MSFT" | jq .
```

Confirm `/predict` returns `upProb`, `tradeSignal`, and `inTrainingUniverse`.

Research overview / intelligence should expose the same forecast as `patternForecast`:

```bash
curl -sS -H "Authorization: Bearer $TOKEN" \
  "$API_BASE/api/v1/research/intelligence?symbol=MSFT" | jq .patternForecast
```

## 4. Deploy

**Manual:** copy `model_xgb.joblib` and `model_meta.json` to the VM artifact dir and restart `sas-server`.

**CI:** GitHub Actions → **Train Pattern Model** (weekly cron or manual dispatch). The workflow uses the tradeable universe and production label config, then SCPs artifacts to `/home/ubuntu/sas-pattern-artifacts` and restarts the container.

## 5. Post-deploy checks

See [deploy-smoke.md](./deploy-smoke.md#pattern-model).

## Config reference

| Setting | Production value |
|---------|------------------|
| Universe | `tradeable_v1` |
| Label scheme | `binary_updown` |
| Class weights | on |
| Min P(up) for trade signal | 0.65 |
| Backtest trade cost | 10 bps |
| Walk-forward train / test | 3y / 1y |

Shared constants live in `models/pattern_production.py`.
