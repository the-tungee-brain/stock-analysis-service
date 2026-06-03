CREATE TABLE IF NOT EXISTS universe_snapshots (
  snapshot_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  symbol_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS universe_members (
  snapshot_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  last_close REAL,
  market_cap REAL,
  avg_dollar_volume_20d REAL,
  passed_filters INTEGER NOT NULL,
  PRIMARY KEY (snapshot_id, symbol)
);

CREATE TABLE IF NOT EXISTS ohlcv_sync (
  symbol TEXT PRIMARY KEY,
  last_bar_date TEXT,
  row_count INTEGER,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ranking_runs (
  run_id TEXT PRIMARY KEY,
  as_of_date TEXT NOT NULL,
  model_backend TEXT NOT NULL,
  universe_snapshot_id TEXT,
  symbol_count INTEGER,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ranking_results (
  run_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  rank INTEGER NOT NULL,
  composite_score REAL,
  ml_probability REAL,
  expected_excess_return REAL,
  final_score REAL NOT NULL,
  contributions_json TEXT NOT NULL,
  PRIMARY KEY (run_id, symbol)
);

CREATE INDEX IF NOT EXISTS idx_ranking_results_run_rank
  ON ranking_results(run_id, rank);

CREATE TABLE IF NOT EXISTS market_regime_daily (
  date TEXT PRIMARY KEY,
  regime_id TEXT NOT NULL,
  regime_multiplier REAL NOT NULL,
  metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS backtest_runs (
  backtest_id TEXT PRIMARY KEY,
  ranking_run_id TEXT NOT NULL,
  as_of_date TEXT NOT NULL,
  top_n INTEGER NOT NULL,
  hold_days INTEGER NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_metrics (
  backtest_id TEXT PRIMARY KEY,
  avg_return REAL,
  avg_excess_return REAL,
  hit_rate_vs_spy REAL,
  sharpe_ratio REAL,
  max_drawdown REAL,
  slippage_bps REAL,
  costs_json TEXT
);
