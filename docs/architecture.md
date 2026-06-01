# Architecture: Stock Daily Pattern Recognition System

## Goal

Build a long-term, production-grade system to:
- Download and maintain daily OHLCV data for US stocks.
- Compute technical indicators and candlestick patterns.
- Train ML models to predict the next 5-day trend (up / down / neutral).
- Validate models with walk-forward validation (no lookahead).
- Serve predictions via a FastAPI API.

## Scope

- Asset class: Stocks only (US-listed, e.g., S&P 500 universe or a custom symbol list).
- Timeframe: Daily bars.
- Horizon: Predict direction of the next 5 trading days.

## Project structure

- `data/`
  - `symbols.py` – returns the list of symbols to use.
  - `download.py` – downloads 10–15 years of daily OHLCV (using yfinance for now).
  - `store.py` – stores data as Parquet/CSV files (or in a DB, but start with files).
  - `loader.py` – loads data into pandas DataFrames for modeling.

- `features/`
  - `indicators.py` – computes RSI, SMA (20/50/200), EMA, MACD, Bollinger Bands, ATR.
  - `patterns.py` – computes candlestick pattern features (pandas-ta or similar).
  - `build_features.py` – combines price, indicators, patterns into a feature DataFrame with aligned dates.

- `models/`
  - `labels.py` – defines the target:
    - Next 5-day return = Close[t+5] / Close[t] - 1.
    - Label = 1 if return > +0.5%, -1 if return < -0.5%, else 0.
  - `xgb_model.py` – trains an XGBoost or LightGBM classifier on tabular features.
  - `walk_forward.py` – runs walk-forward validation:
    - For each window:
      - Train on 5 years of data.
      - Test on the following 1 year (out-of-sample).
    - Aggregates metrics across windows.

- `backtest/`
  - `metrics.py` – computes accuracy, precision/recall, Sharpe ratio (using a simple long/flat strategy), max drawdown, and profit factor.
  - `run_backtest.py` – ties together data, features, labels, modeling, and metrics in a reproducible experiment.

- `api/`
  - `main.py` – FastAPI app that:
    - Loads the latest trained model and feature config at startup.
    - For a given symbol:
      - Loads the latest daily data.
      - Computes the latest feature row.
      - Returns predicted class (up/down/neutral) and probabilities.
    - Exposes a `/metrics` endpoint with recent live performance statistics (stubbed at first).

## Modeling details

- Features (per symbol, per day):
  - Price-based:
    - 1-day, 5-day, 10-day returns.
    - Rolling 20-day volatility of daily returns.
    - Relative position vs SMA20 and SMA200.
  - Technical indicators:
    - RSI (14).
    - SMA (20, 50, 200).
    - EMA (12, 26).
    - MACD (12, 26, 9), MACD signal, MACD histogram.
    - Bollinger Bands (20, 2).
    - ATR (14).
  - Candlestick patterns:
    - Use pandas-ta candlestick pattern functions (or equivalent) to generate binary features like hammer, engulfing, doji, etc.

- Labels:
  - For each day t, compute:
    - `future_ret_5d = Close[t+5] / Close[t] - 1`.
    - `label`:
      - 1 if `future_ret_5d > +0.005` (up).
      - -1 if `future_ret_5d < -0.005` (down).
      - 0 otherwise (neutral).
  - Drop rows where future data is not available.

- Model:
  - XGBoost or LightGBM multiclass classifier.
  - Input: features described above.
  - Output: probabilities for classes [-1, 0, 1].

## Validation

- Use **walk-forward validation**:
  - Example:
    - Window 1: Train = 2010–2014, Test = 2015.
    - Window 2: Train = 2011–2015, Test = 2016.
    - Window 3: Train = 2012–2016, Test = 2017.
    - ... continue until last full window.
  - For each window:
    - Train model only on the train period.
    - Evaluate on the test period (never used in training).
  - Aggregate metrics over all out-of-sample test periods.

- Metrics:
  - Directional accuracy overall and per class (up/down/neutral).
  - Sharpe ratio of a simple strategy:
    - Go long when model predicts "up"; be flat otherwise.
  - Max drawdown.
  - Profit factor (gross profit / gross loss).

- Important constraints:
  - **No lookahead bias**: features and labels must only use information available up to day t.
  - **Time-based splits only**: never shuffle across time.

## API behavior

- `/predict?symbol=XYZ`:
  - Input: symbol string (must be in our universe).
  - Process:
    - Load most recent daily data for `symbol`.
    - Compute current feature vector for `symbol`.
    - Apply the latest trained model.
  - Output (JSON):
    - `symbol`
    - `date`
    - `prediction` in `[-1, 0, 1]`
    - `probabilities` for each class
    - key indicators (RSI, SMA20, SMA200, MACD, BB position)

- `/health`:
  - Returns "ok" and model metadata.

- `/metrics` (later):
  - Returns recent live performance metrics, updated daily.

## Technologies

- Python 3.11+
- pandas, numpy
- yfinance (for data)
- pandas-ta (for indicators / candlestick patterns)
- xgboost or lightgbm
- scikit-learn (for metrics and pipeline helpers)
- FastAPI + Uvicorn
- Poetry or pip + requirements.txt for dependency management
