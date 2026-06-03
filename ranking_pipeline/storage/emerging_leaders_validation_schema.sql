CREATE TABLE IF NOT EXISTS emerging_leaders_daily_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  snapshot_date TEXT NOT NULL,
  symbol TEXT NOT NULL,
  rank INTEGER NOT NULL,
  setup_score INTEGER NOT NULL,
  compression_velocity INTEGER NOT NULL,
  setup_purity REAL NOT NULL,
  stage TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE (snapshot_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_el_snapshots_date
  ON emerging_leaders_daily_snapshots (snapshot_date);

CREATE TABLE IF NOT EXISTS emerging_leaders_forward_outcomes (
  snapshot_id INTEGER PRIMARY KEY
    REFERENCES emerging_leaders_daily_snapshots (id) ON DELETE CASCADE,
  ret_5d REAL,
  ret_10d REAL,
  ret_20d REAL,
  spy_ret_5d REAL,
  spy_ret_10d REAL,
  spy_ret_20d REAL,
  excess_ret_5d REAL,
  excess_ret_10d REAL,
  excess_ret_20d REAL,
  universe_pct_rank_5d REAL,
  universe_pct_rank_10d REAL,
  universe_pct_rank_20d REAL,
  computed_at TEXT NOT NULL
);
