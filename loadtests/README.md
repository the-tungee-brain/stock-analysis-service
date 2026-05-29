# Load tests (k6)

Automated **timing and pass/fail checks** for scale-sensitive cron paths. These are not run on every PR — use **workflow_dispatch** or run locally against **staging**.

## Prerequisites

- [k6](https://grafana.com/docs/k6/latest/set-up/install-k6/) installed locally, or use the GitHub Action workflow.
- `API_BASE_URL` — e.g. `https://your-host/api/v1` (no trailing slash required).
- `CRON_SECRET` — must match server `CRON_SECRET`.

## Morning brief pre-warm (safe default)

Exercises the same endpoint as production cron (`prewarm-morning-briefs`). Does **not** send email.

```bash
export API_BASE_URL="https://staging.example.com/api/v1"
export CRON_SECRET="your-secret"

k6 run loadtests/morning_brief_prewarm.js
```

Optional env:

| Variable | Default | Meaning |
|----------|---------|---------|
| `PREWARM_MAX_SECONDS` | `600` | k6 request timeout and p95 threshold |

**Pass criteria:** HTTP 200, `failed === 0`, wall time within `PREWARM_MAX_SECONDS`.

Compare `warmed` / `attempted` in the k6 log to your Schwab-linked user count. For a **5k-user SLO (~10 min)**, prewarm alone should finish with comfortable headroom before dispatch.

## Morning brief dispatch (staging only)

**Sends real emails.** Requires explicit opt-in:

```bash
export ALLOW_DISPATCH=true
# optional: export DISPATCH_FORCE=true
k6 run loadtests/morning_brief_dispatch.js
```

## Shell helper (no k6)

```bash
./scripts/loadtest_morning_brief_prewarm.sh
```

## CI

`.github/workflows/loadtest-morning-brief.yml` — manual run, **prewarm only** by default. Enable dispatch via workflow input only on non-production targets.

## Production

Do **not** run dispatch load tests against production. Prewarm against production is still heavy (builds all briefs); prefer staging or off-peak manual runs with ops approval.
