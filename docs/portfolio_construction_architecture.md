# Portfolio Construction Layer

Downstream consumer of precomputed ranking runs. Does not modify ranking, feature, or ML modules.

## Architecture

```mermaid
flowchart LR
  subgraph rank [Ranking batch - unchanged]
    SQL[(ranking_results)]
  end
  subgraph port [Portfolio layer - new]
    LOAD[Load top-N ranked rows]
    AUX[Read ATR from feature Parquet]
    ADV[Read ADV from universe_members]
    SIZE[sizing.py]
    CON[constraints.py]
    SM[rebalancer smoothing]
    MET[metrics.py]
    PSQL[(portfolio_snapshots)]
  end
  subgraph bt [Backtest extension - new files only]
    PSIM[portfolio_sim.py]
    BCOST[costs.py reuse]
  end
  subgraph api [API]
    PAPI["GET /portfolio/latest"]
  end
  SQL --> LOAD
  LOAD --> SIZE
  AUX --> SIZE
  ADV --> CON
  SIZE --> CON --> SM --> MET --> PSQL
  PSQL --> PAPI
  PSQL --> PSIM --> BCOST
```

## Portfolio risk layer (downstream)

See [`portfolio_risk_layer.md`](portfolio_risk_layer.md). Run `scripts/run_portfolio_with_risk.py` after ranking; extends API `risk_layer` on `GET /portfolio/latest`.

## Data flow: ranking → portfolio → execution

```mermaid
sequenceDiagram
  participant Rank as ranking_runs/results
  participant Cons as portfolio/constructor
  participant DB as SQLite portfolio tables
  participant API as /portfolio/latest
  participant BT as backtest/portfolio_sim
  Rank->>Cons: top N symbols + scores + E[r]
  Cons->>Cons: sizing → constraints → smooth
  Cons->>DB: weights + metrics + trades JSON
  DB->>API: latest snapshot
  Cons->>BT: weights + realized labels
  BT->>DB: portfolio_backtest_metrics
```

## Weighting strategy comparison

| Mode | Formula | Pros | Cons |
|------|---------|------|------|
| **Equal weight** | `w_i = 1/N` | Simple, diversified | Ignores signal strength |
| **Score weighted** | `w_i ∝ final_score` | Tilts to best ranks | Can concentrate if scores skewed |
| **Vol adjusted** | `w_i ∝ final_score / ATR_14` | Less exposure to volatile names | Needs ATR from stored features |

All modes pass through: liquidity filter → max weight cap → turnover cap → EWM smooth vs prior day.

## Smoothing

`w_t = (1 - α) * w_{t-1} + α * w_new` with default `α = 0.3` (70% prior, 30% new target).

## Module layout

```
ranking_pipeline/portfolio/
  config.py
  sizing.py
  constraints.py
  rebalancer.py
  metrics.py
  constructor.py
  persistence.py
  schema.sql
ranking_pipeline/backtest/
  portfolio_sim.py      # new — does not edit evaluate.py
```
