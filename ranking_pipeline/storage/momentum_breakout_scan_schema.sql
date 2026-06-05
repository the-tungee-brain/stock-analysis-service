CREATE TABLE IF NOT EXISTS momentum_breakout_scan_runs (
  run_id TEXT PRIMARY KEY,
  as_of_date TEXT,
  generated_at TEXT NOT NULL,
  ranking_run_id TEXT,
  ranking_snapshot_id TEXT,
  universe_source TEXT,
  selection_method TEXT,
  total_ranked_symbols INTEGER NOT NULL DEFAULT 0,
  total_eligible_symbols INTEGER NOT NULL DEFAULT 0,
  symbols_scanned INTEGER NOT NULL DEFAULT 0,
  excluded_by_cap INTEGER NOT NULL DEFAULT 0,
  valid_setups_found INTEGER NOT NULL DEFAULT 0,
  tradable_candidates_found INTEGER NOT NULL DEFAULT 0,
  blocked_candidates_count INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  error_message TEXT,
  duration_ms INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_mb_scan_runs_status_generated
  ON momentum_breakout_scan_runs (status, generated_at DESC);

CREATE TABLE IF NOT EXISTS momentum_breakout_scan_results (
  run_id TEXT NOT NULL
    REFERENCES momentum_breakout_scan_runs (run_id) ON DELETE CASCADE,
  rank INTEGER NOT NULL,
  symbol TEXT NOT NULL,
  entry_price REAL NOT NULL,
  stop_price REAL NOT NULL,
  target_price REAL NOT NULL,
  risk_reward REAL NOT NULL,
  historical_win_rate REAL,
  historical_profit_factor REAL,
  historical_total_trades INTEGER,
  setup_score REAL NOT NULL,
  stop_distance_pct REAL NOT NULL,
  volume_ratio REAL,
  rs_percentile REAL,
  market_regime TEXT,
  risk_gate_json TEXT NOT NULL,
  PRIMARY KEY (run_id, rank),
  UNIQUE (run_id, symbol)
);

CREATE INDEX IF NOT EXISTS idx_mb_scan_results_run_rank
  ON momentum_breakout_scan_results (run_id, rank);
