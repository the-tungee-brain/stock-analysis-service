CREATE TABLE IF NOT EXISTS emerging_leaders_snapshot_runs (
  run_id TEXT PRIMARY KEY,
  as_of_date TEXT,
  generated_at TEXT NOT NULL,
  universe_snapshot_id TEXT,
  ranking_run_id TEXT,
  symbols_with_data INTEGER NOT NULL DEFAULT 0,
  candidates_scanned INTEGER NOT NULL DEFAULT 0,
  excluded_top_movers INTEGER NOT NULL DEFAULT 0,
  evaluations_computed INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  error_message TEXT,
  duration_ms INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_el_snapshot_runs_status_generated
  ON emerging_leaders_snapshot_runs (status, generated_at DESC);

CREATE TABLE IF NOT EXISTS emerging_leaders_snapshot_results (
  run_id TEXT NOT NULL
    REFERENCES emerging_leaders_snapshot_runs (run_id) ON DELETE CASCADE,
  rank INTEGER NOT NULL,
  symbol TEXT NOT NULL,
  setup_quality_score INTEGER NOT NULL,
  setup_stage TEXT NOT NULL,
  setup_stage_label TEXT NOT NULL,
  compression_velocity INTEGER NOT NULL,
  compression_velocity_label TEXT NOT NULL,
  why_it_ranks TEXT NOT NULL,
  positive_factors_json TEXT NOT NULL,
  missing_factors_json TEXT NOT NULL,
  next_confirmation_json TEXT NOT NULL,
  components_json TEXT NOT NULL,
  PRIMARY KEY (run_id, rank),
  UNIQUE (run_id, symbol)
);

CREATE INDEX IF NOT EXISTS idx_el_snapshot_results_run_rank
  ON emerging_leaders_snapshot_results (run_id, rank);
