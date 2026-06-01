# Post-deploy smoke checks

Run after deploying **stock-analysis** (and the matching **my-pocket** build when UI changed).

## Morning brief pipeline

1. Confirm `CRON_SECRET` is set on the API and in GitHub Actions secrets.
2. Trigger or wait for **Morning brief** workflow (`morning-brief.yml` at 13:23 UTC). Its `prewarm` job runs first, then `dispatch` (`needs: prewarm`).
3. Check API logs for `morning brief prewarm finished` with non-zero `warmed` when users have Schwab linked.
4. Check `morning brief dispatch finished` for expected `sent` / low `failed`.
5. Dispatch logs should show `attempted` / `sent` without spikes in `failed`.

Manual prewarm (production):

```bash
curl -sS -X POST "$API_BASE/api/v1/internal/prewarm-morning-briefs" \
  -H "X-Cron-Secret: $CRON_SECRET"
```

## Portfolio positions fast path

1. Sign in and open **Portfolio** (or any view that loads positions).
2. First load may return `dataFreshness.briefStatus=pending`; brief should populate on refresh or within background warm.
3. API logs should include `positions load` with `brief_status=cached` or `ready` on subsequent loads.

## Research overview bundle

1. Open `/research/{SYMBOL}/overview` for a stock (e.g. `AAPL`) and an ETF (e.g. `SPY`).
2. Network tab: one `GET /api/v1/research/overview-bundle?symbol=...` (no duplicate snapshot/performance/intelligence calls on overview).
3. API logs: `research overview bundle` with `latency_ms` under a few seconds without `include_summary=true` unless explicitly refreshing AI summary.
4. Repeat overview load within 2 minutes: second request may return **304** (`research overview bundle … not_modified` in logs) when the client sends `If-None-Match`.

## Load test (optional, staging)

Automated prewarm timing: [loadtests/README.md](../loadtests/README.md).

- Local: `k6 run loadtests/morning_brief_prewarm.js` with `API_BASE_URL` + `CRON_SECRET`
- CI: **Actions → Load test — morning brief** (prewarm only; dispatch opt-in sends email)

## Pattern model

After training or deploying a new pattern model, see [pattern-model-review.md](./pattern-model-review.md).

1. Confirm artifacts exist on the VM: `/home/ubuntu/sas-pattern-artifacts/model_xgb.joblib`.
2. `GET /api/v1/pattern/health` returns `status=ok` with:
   - `labelScheme=binary_outperform_spy`
   - `modelKey=C`, `modelLabel=Relative strength + trend`
   - `featureGroups=["relative_strength","trend"]`, `nFeatures=11`
   - `portfolioStrategy.strategyType=ranking`, `portfolioStrategy.universe=top20`, `topN=10`
3. `GET /api/v1/pattern/predict?symbol=MSFT` includes `rankingScore`, `upProb`, `inTrainingUniverse`, RS/trend `indicators`, and `portfolioStrategy`.
4. Research intelligence exposes the same payload: `GET /api/v1/research/intelligence?symbol=MSFT` → `patternForecast`.
5. Frontend (my-pocket): research overview should render the 5-day ranking card using `rankingScore` / `portfolioStrategy` (not threshold `tradeSignal` alone).

## Regression

```bash
cd stock-analysis && pytest -q
cd ../my-pocket && npm run build
```
