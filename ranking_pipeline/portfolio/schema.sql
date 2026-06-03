CREATE TABLE IF NOT EXISTS portfolio_snapshots (
  portfolio_id TEXT PRIMARY KEY,
  ranking_run_id TEXT NOT NULL,
  as_of_date TEXT NOT NULL,
  sizing_mode TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_holdings (
  portfolio_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  weight REAL NOT NULL,
  final_score REAL,
  ml_probability REAL,
  expected_excess_return REAL,
  atr_14 REAL,
  PRIMARY KEY (portfolio_id, symbol)
);

CREATE TABLE IF NOT EXISTS portfolio_metrics (
  portfolio_id TEXT PRIMARY KEY,
  expected_return_5d REAL,
  expected_excess_5d REAL,
  portfolio_volatility REAL,
  turnover REAL,
  concentration_hhi REAL,
  metrics_json TEXT
);

CREATE TABLE IF NOT EXISTS portfolio_trades (
  portfolio_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  side TEXT NOT NULL,
  weight_change REAL NOT NULL,
  target_weight REAL NOT NULL,
  previous_weight REAL NOT NULL,
  PRIMARY KEY (portfolio_id, symbol)
);

CREATE TABLE IF NOT EXISTS portfolio_backtest_runs (
  portfolio_backtest_id TEXT PRIMARY KEY,
  portfolio_id TEXT NOT NULL,
  ranking_run_id TEXT NOT NULL,
  as_of_date TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS portfolio_backtest_metrics (
  portfolio_backtest_id TEXT PRIMARY KEY,
  portfolio_return REAL,
  excess_vs_spy REAL,
  sharpe_ratio REAL,
  max_drawdown REAL,
  turnover REAL,
  slippage_bps REAL,
  metrics_json TEXT
);
