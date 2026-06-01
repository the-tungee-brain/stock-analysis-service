"""Ranking-based portfolio backtests for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd

from backtest.metrics import UP_PROB_COLUMN, apply_trade_cost, equity_curve, max_drawdown, profit_factor, sharpe_ratio
from backtest.portfolio_weights import capped_equal_weights, weighted_portfolio_return, weights_for_symbols
from models.labels import EXCESS_RETURN_COLUMN, FUTURE_RETURN_COLUMN, LABEL_HORIZON_DAYS

TRADING_DAYS_PER_YEAR = 252
StrategyName = Literal[
    "long_top_quintile",
    "long_short_quintile",
    "long_top_n",
    "threshold_long",
]


class RankingStrategy(str, Enum):
    LONG_TOP_QUINTILE = "long_top_quintile"
    LONG_SHORT_QUINTILE = "long_short_quintile"
    LONG_TOP_N = "long_top_n"
    THRESHOLD_LONG = "threshold_long"


@dataclass(frozen=True)
class RankingPortfolioConfig:
    strategy: RankingStrategy | str = RankingStrategy.LONG_TOP_QUINTILE
    top_n: int = 5
    quintile_fraction: float = 0.2
    rebalance_days: int = LABEL_HORIZON_DAYS
    hold_days: int = LABEL_HORIZON_DAYS
    trade_cost_bps: float = 10.0
    min_up_prob: float | None = None
    max_position_weight: float | None = None
    use_excess_returns: bool = True


@dataclass(frozen=True)
class PortfolioPeriod:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    gross_return: float
    net_return: float
    turnover: float
    n_long: int
    n_short: int
    long_symbols: tuple[str, ...]
    short_symbols: tuple[str, ...]


def quintile_count(n_symbols: int, *, fraction: float = 0.2) -> int:
    """Number of names in one quintile bucket."""
    if n_symbols <= 0:
        return 0
    return max(1, int(np.ceil(n_symbols * fraction)))


def select_long_short_symbols(
    cross_section: pd.DataFrame,
    *,
    strategy: RankingStrategy | str,
    top_n: int,
    quintile_fraction: float,
    min_up_prob: float | None,
    score_col: str = UP_PROB_COLUMN,
) -> tuple[list[str], list[str]]:
    """Return long and short symbol lists for one rebalance date."""
    scheme = RankingStrategy(strategy)
    ranked = cross_section.sort_values(score_col, ascending=False)
    if min_up_prob is not None:
        ranked = ranked[ranked[score_col] >= min_up_prob]

    n = len(ranked)
    if n == 0:
        return [], []

    if scheme == RankingStrategy.LONG_TOP_QUINTILE:
        k = quintile_count(n, fraction=quintile_fraction)
        return ranked.head(k)["symbol"].astype(str).tolist(), []

    if scheme == RankingStrategy.LONG_SHORT_QUINTILE:
        k = quintile_count(n, fraction=quintile_fraction)
        long_symbols = ranked.head(k)["symbol"].astype(str).tolist()
        short_symbols = ranked.tail(k)["symbol"].astype(str).tolist()
        return long_symbols, short_symbols

    if scheme == RankingStrategy.LONG_TOP_N:
        k = min(top_n, n)
        return ranked.head(k)["symbol"].astype(str).tolist(), []

    if scheme == RankingStrategy.THRESHOLD_LONG:
        long_symbols = ranked[ranked[score_col] >= (min_up_prob or 0.65)]["symbol"].astype(str).tolist()
        return long_symbols, []

    raise ValueError(f"Unsupported strategy: {strategy}")


def _equal_weight_return(
    cross_section: pd.DataFrame,
    symbols: list[str],
    *,
    return_col: str,
) -> float:
    if not symbols:
        return 0.0
    subset = cross_section[cross_section["symbol"].isin(symbols)]
    if subset.empty:
        return 0.0
    return float(subset[return_col].astype("float64").mean())


def _compute_turnover(
    prev_weights: dict[str, float],
    new_weights: dict[str, float],
) -> float:
    symbols = set(prev_weights) | set(new_weights)
    if not symbols:
        return 0.0
    turnover = 0.0
    for symbol in symbols:
        turnover += abs(new_weights.get(symbol, 0.0) - prev_weights.get(symbol, 0.0))
    return turnover / 2.0


def _weights_for_symbols(symbols: list[str], *, gross: float = 1.0) -> dict[str, float]:
    return weights_for_symbols(symbols, gross=gross)


def simulate_ranking_portfolio(
    panel: pd.DataFrame,
    config: RankingPortfolioConfig,
    *,
    return_col: str | None = None,
    score_col: str = UP_PROB_COLUMN,
) -> tuple[pd.DataFrame, list[PortfolioPeriod]]:
    """Simulate non-overlapping ranking portfolios on rebalance dates."""
    cfg = _normalize_config(config)
    if panel.empty:
        return _empty_period_frame(), []

    resolved_return_col = return_col or (
        EXCESS_RETURN_COLUMN if cfg.use_excess_returns else FUTURE_RETURN_COLUMN
    )
    required = {"date", "symbol", score_col, resolved_return_col}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel missing required columns: {sorted(missing)}")

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame = frame.dropna(subset=[score_col, resolved_return_col])

    rebalance_dates = _rebalance_dates(frame["date"], cfg.rebalance_days)
    periods: list[PortfolioPeriod] = []
    prev_long_weights: dict[str, float] = {}
    prev_short_weights: dict[str, float] = {}

    for entry_date in rebalance_dates:
        cross_section = frame[frame["date"] == entry_date]
        if len(cross_section) < 2:
            continue

        long_symbols, short_symbols = select_long_short_symbols(
            cross_section,
            strategy=cfg.strategy,
            top_n=cfg.top_n,
            quintile_fraction=cfg.quintile_fraction,
            min_up_prob=cfg.min_up_prob,
            score_col=score_col,
        )
        if cfg.strategy == RankingStrategy.LONG_SHORT_QUINTILE:
            if not long_symbols or not short_symbols:
                continue
        elif not long_symbols:
            continue

        long_ret = weighted_portfolio_return(
            cross_section,
            capped_equal_weights(
                long_symbols,
                gross=1.0,
                max_weight=cfg.max_position_weight,
            ),
            return_col=resolved_return_col,
        )
        if cfg.strategy == RankingStrategy.LONG_SHORT_QUINTILE:
            short_ret = weighted_portfolio_return(
                cross_section,
                capped_equal_weights(
                    short_symbols,
                    gross=1.0,
                    max_weight=cfg.max_position_weight,
                ),
                return_col=resolved_return_col,
            )
            gross_return = long_ret - short_ret
            new_long = capped_equal_weights(
                long_symbols,
                gross=0.5,
                max_weight=cfg.max_position_weight,
            )
            new_short = capped_equal_weights(
                short_symbols,
                gross=0.5,
                max_weight=cfg.max_position_weight,
            )
            turnover = _compute_turnover(prev_long_weights, new_long) + _compute_turnover(
                prev_short_weights,
                {symbol: -weight for symbol, weight in new_short.items()},
            )
            prev_long_weights = new_long
            prev_short_weights = new_short
        else:
            gross_return = long_ret
            new_long = capped_equal_weights(
                long_symbols,
                gross=1.0,
                max_weight=cfg.max_position_weight,
            )
            turnover = _compute_turnover(prev_long_weights, new_long)
            prev_long_weights = new_long
            prev_short_weights = {}

        net_return = apply_trade_cost(gross_return, cfg.trade_cost_bps * turnover)
        exit_date = entry_date + pd.offsets.BDay(cfg.hold_days)
        periods.append(
            PortfolioPeriod(
                entry_date=entry_date,
                exit_date=exit_date,
                gross_return=gross_return,
                net_return=net_return,
                turnover=turnover,
                n_long=len(long_symbols),
                n_short=len(short_symbols),
                long_symbols=tuple(long_symbols),
                short_symbols=tuple(short_symbols),
            )
        )

    return _periods_to_frame(periods), periods


def summarize_portfolio_performance(
    period_frame: pd.DataFrame,
    *,
    hold_days: int,
) -> dict[str, Any]:
    """Compute CAGR, Sharpe, Sortino, PF, drawdown, turnover."""
    if period_frame.empty:
        return _empty_summary()

    rets = period_frame["net_return"].astype("float64")
    gross = period_frame["gross_return"].astype("float64")
    equity = equity_curve(rets)
    periods_per_year = TRADING_DAYS_PER_YEAR / hold_days
    years = len(rets) / periods_per_year if periods_per_year > 0 else float("nan")
    cumulative = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years and years > 0 else float("nan")
    ann_return = float(rets.mean() * periods_per_year)

    return {
        "n_periods": int(len(rets)),
        "cagr": cagr,
        "annualized_return": ann_return,
        "cumulative_return": cumulative,
        "sharpe_ratio": sharpe_ratio(rets, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(rets, periods_per_year=periods_per_year),
        "profit_factor": profit_factor(rets),
        "max_drawdown": max_drawdown(equity),
        "avg_turnover": float(period_frame["turnover"].mean()),
        "total_turnover": float(period_frame["turnover"].sum()),
        "win_rate": float((rets > 0).mean()),
        "avg_gross_return": float(gross.mean()),
        "avg_net_return": float(rets.mean()),
    }


def sortino_ratio(
    strategy_rets: pd.Series,
    *,
    periods_per_year: float,
    mar: float = 0.0,
) -> float:
    if strategy_rets.empty:
        return float("nan")
    rets = strategy_rets.fillna(0.0)
    downside = rets[rets < mar]
    if downside.empty:
        return float("inf") if rets.mean() > mar else float("nan")
    downside_std = float(downside.std(ddof=0))
    if downside_std == 0:
        return float("nan")
    return float((rets.mean() - mar) / downside_std * np.sqrt(periods_per_year))


def build_simulation_panel(
    predictions: pd.DataFrame,
    labeled_by_symbol: dict[str, pd.DataFrame],
    *,
    hold_days: int,
    benchmark_excess: bool = True,
) -> pd.DataFrame:
    """Merge OOS predictions with forward returns for the requested hold horizon."""
    if predictions.empty:
        return pd.DataFrame()

    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"]).dt.normalize()
    return_col = EXCESS_RETURN_COLUMN if benchmark_excess else FUTURE_RETURN_COLUMN

    if hold_days == LABEL_HORIZON_DAYS and return_col in pred.columns:
        return pred

    pieces: list[pd.DataFrame] = []
    for symbol, df in labeled_by_symbol.items():
        piece = df.copy()
        if "date" not in piece.columns:
            piece = piece.reset_index()
            if piece.columns[0] != "date":
                piece = piece.rename(columns={piece.columns[0]: "date"})
        piece["date"] = pd.to_datetime(piece["date"]).dt.normalize()
        piece["symbol"] = symbol.strip().upper()
        pieces.append(piece[["date", "symbol", FUTURE_RETURN_COLUMN, EXCESS_RETURN_COLUMN]])

    labels = pd.concat(pieces, ignore_index=True)
    merged = pred.merge(labels, on=["date", "symbol"], how="left", suffixes=("", "_label"))

    if hold_days == LABEL_HORIZON_DAYS:
        merged[return_col] = merged[EXCESS_RETURN_COLUMN]
        return merged

    # Recompute forward returns for non-5D holds from stored close in raw data.
    from data.loader import load_symbol

    forward_rows: list[pd.DataFrame] = []
    for symbol in merged["symbol"].unique():
        raw = load_symbol(str(symbol))
        close = raw["close"].astype("float64")
        stock_ret = close.shift(-hold_days) / close - 1.0
        out = pd.DataFrame(
            {
                "date": pd.to_datetime(stock_ret.index).normalize(),
                "symbol": symbol,
                FUTURE_RETURN_COLUMN: stock_ret.to_numpy(),
            }
        )
        forward_rows.append(out)

    forward = pd.concat(forward_rows, ignore_index=True)

    if benchmark_excess:
        spy = load_symbol("SPY")["close"].astype("float64")
        spy_ret = spy.shift(-hold_days) / spy - 1.0
        spy_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(spy_ret.index).normalize(),
                "spy_ret": spy_ret.to_numpy(),
            }
        )
        forward = forward.merge(spy_frame, on="date", how="left")
        forward[EXCESS_RETURN_COLUMN] = forward[FUTURE_RETURN_COLUMN] - forward["spy_ret"]
        forward = forward.drop(columns=["spy_ret"])
    else:
        forward[EXCESS_RETURN_COLUMN] = forward[FUTURE_RETURN_COLUMN]

    merged = pred.merge(forward, on=["date", "symbol"], how="left", suffixes=("", "_fwd"))
    merged[return_col] = merged[EXCESS_RETURN_COLUMN if benchmark_excess else FUTURE_RETURN_COLUMN]
    return merged.dropna(subset=[return_col, UP_PROB_COLUMN])


def _rebalance_dates(dates: pd.Series, rebalance_days: int) -> list[pd.Timestamp]:
    unique = sorted(pd.to_datetime(dates.dropna().unique()))
    if not unique:
        return []
    if rebalance_days <= 1:
        return [pd.Timestamp(date).normalize() for date in unique]
    return [pd.Timestamp(unique[idx]).normalize() for idx in range(0, len(unique), rebalance_days)]


def _normalize_config(config: RankingPortfolioConfig) -> RankingPortfolioConfig:
    strategy = RankingStrategy(config.strategy)
    hold_days = int(config.hold_days)
    rebalance_days = int(config.rebalance_days)
    if hold_days <= 0 or rebalance_days <= 0:
        raise ValueError("hold_days and rebalance_days must be positive")
    return RankingPortfolioConfig(
        strategy=strategy,
        top_n=int(config.top_n),
        quintile_fraction=float(config.quintile_fraction),
        rebalance_days=rebalance_days,
        hold_days=hold_days,
        trade_cost_bps=float(config.trade_cost_bps),
        min_up_prob=config.min_up_prob,
        max_position_weight=config.max_position_weight,
        use_excess_returns=bool(config.use_excess_returns),
    )


def _periods_to_frame(periods: list[PortfolioPeriod]) -> pd.DataFrame:
    if not periods:
        return _empty_period_frame()
    return pd.DataFrame(
        {
            "entry_date": [p.entry_date for p in periods],
            "exit_date": [p.exit_date for p in periods],
            "gross_return": [p.gross_return for p in periods],
            "net_return": [p.net_return for p in periods],
            "turnover": [p.turnover for p in periods],
            "n_long": [p.n_long for p in periods],
            "n_short": [p.n_short for p in periods],
            "long_symbols": [",".join(p.long_symbols) for p in periods],
            "short_symbols": [",".join(p.short_symbols) for p in periods],
        }
    )


def _empty_period_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "entry_date",
            "exit_date",
            "gross_return",
            "net_return",
            "turnover",
            "n_long",
            "n_short",
            "long_symbols",
            "short_symbols",
        ]
    )


def _empty_summary() -> dict[str, Any]:
    return {
        "n_periods": 0,
        "cagr": float("nan"),
        "annualized_return": float("nan"),
        "cumulative_return": float("nan"),
        "sharpe_ratio": float("nan"),
        "sortino_ratio": float("nan"),
        "profit_factor": float("nan"),
        "max_drawdown": float("nan"),
        "avg_turnover": float("nan"),
        "total_turnover": float("nan"),
        "win_rate": float("nan"),
        "avg_gross_return": float("nan"),
        "avg_net_return": float("nan"),
    }
