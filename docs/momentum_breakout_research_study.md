# Momentum Breakout — Full Research Study

**Universe:** AAPL, MSFT, NVDA, META, AMZN  
**Requested period:** 2000-01-01 → 2024-12-31  
**Effective sample start:** 2011-06-02 (limited by listing history)  

## Data availability

| Symbol | First bar in dataset |
|--------|----------------------|
| AAPL | 2011-06-02 |
| MSFT | 2011-06-02 |
| NVDA | 2011-06-02 |
| META | 2012-05-18 |
| AMZN | 2011-06-02 |

**Total historical trades:** 136  
**Current production config:** RS ≥ 80.0, volume ratio ≥ 1.5, target R = 2.0, stop = 10-day low  

## 1. Overall performance

| Metric | Value |
|--------|-------|
| Total trades | 136 |
| Win rate | 64.0% |
| Average win | 6.6% |
| Average loss | -5.7% |
| Expectancy | 2.2% |
| Profit factor | 2.05 |
| Sharpe ratio | 1.02 |
| Max drawdown | 63.9% |
| Average holding days | 17.9 |

## 2. Walk-forward (out-of-sample years)

| Test year | Trades | Win rate | Profit factor | Expectancy | Flags |
|-----------|--------|----------|---------------|------------|-------|
| 2019 | 12 | 75.0% | 2.22 | 1.9% | — |
| 2020 | 16 | 50.0% | 0.59 | -2.7% | PF < 1, E < 0 |
| 2021 | 22 | 59.1% | 2.42 | 3.6% | — |
| 2022 | 0 | 0.0% | 0.00 | 0.0% | PF < 1 |
| 2023 | 8 | 87.5% | 33.02 | 6.1% | — |
| 2024 | 3 | 100.0% | ∞ | 9.2% | — |
| **OOS aggregate** | **61** | **65.6%** | **1.76** | **2.2%** | — |

**Flagged OOS years (PF < 1 or negative expectancy):** 2020, 2022

## 3. Regime analysis

| Regime | Trades | Win rate | Profit factor | Expectancy | Avg return |
|--------|--------|----------|---------------|------------|------------|
| RISK_ON | 96 | 52.1% | 1.37 | 1.0% | 1.0% |
| NEUTRAL | 31 | 96.8% | 22.34 | 4.7% | 4.7% |
| RISK_OFF | 9 | 77.8% | 8.41 | 5.5% | 5.5% |

**Regime filter recommendation:** No regime meets the disable threshold (PF < 1 and negative expectancy with ≥10 trades). Prefer sizing down in weaker regimes rather than a hard off switch.

## 4. Feature analysis

### Top 10 single-feature bins (by expectancy)

| Rank | Feature | Range | Trades | Win rate | Expectancy |
|------|---------|-------|--------|----------|------------|
| 1 | close_vs_sma200 | 0.2083 – 0.3312 | 27 | 70.4% | 4.2% |
| 2 | volume_ratio | 1.501 – 1.599 | 28 | 71.4% | 3.9% |
| 3 | volume_ratio | 1.918 – 2.215 | 27 | 66.7% | 3.7% |
| 4 | distance_to_20d_high | 0 – 0.002209 | 28 | 64.3% | 3.4% |
| 5 | rs_percentile | 93.33 – 97.5 | 24 | 66.7% | 3.3% |
| 6 | close_vs_sma50 | 0.06785 – 0.09383 | 27 | 70.4% | 3.2% |
| 7 | volume_ratio | 1.599 – 1.742 | 27 | 70.4% | 3.2% |
| 8 | distance_to_20d_high | 0.004471 – 0.008447 | 27 | 66.7% | 3.1% |
| 9 | close_vs_sma50 | 0.02286 – 0.06785 | 28 | 75.0% | 2.9% |
| 10 | close_vs_sma200 | 0.05658 – 0.1336 | 28 | 82.1% | 2.9% |

### Bottom 10 single-feature bins

| Rank | Feature | Range | Trades | Win rate | Expectancy |
|------|---------|-------|--------|----------|------------|
| 1 | volume_ratio | 1.742 – 1.918 | 27 | 44.4% | -0.4% |
| 2 | distance_to_20d_high | 0.002209 – 0.004471 | 27 | 59.3% | -0.1% |
| 3 | close_vs_sma200 | 0.1634 – 0.2083 | 27 | 33.3% | 0.2% |
| 4 | volume_ratio | 2.215 – 6.009 | 27 | 66.7% | 0.3% |
| 5 | close_vs_sma200 | 0.3312 – 0.9984 | 27 | 63.0% | 0.7% |
| 6 | close_vs_sma50 | 0.09383 – 0.1199 | 27 | 55.6% | 0.9% |
| 7 | close_vs_sma50 | 0.1857 – 0.3767 | 27 | 70.4% | 1.5% |
| 8 | rs_percentile | 87.5 – 93.33 | 30 | 60.0% | 1.6% |
| 9 | rs_percentile | 80 – 87.5 | 28 | 64.3% | 1.7% |
| 10 | distance_to_20d_high | 0.01221 – 0.01936 | 27 | 63.0% | 1.9% |

### Top 10 multi-feature combinations (RS × volume × trend proxies)

| Rank | Condition | Trades | Win rate | PF | Expectancy |
|------|-----------|--------|----------|-----|------------|
| 1 | rs_percentile >= 95, close_vs_sma50 >= 0.08 | 61 | 65.6% | 2.16 | 2.9% |
| 2 | rs_percentile >= 95, close_vs_sma50 >= 8% | 61 | 65.6% | 2.16 | 2.9% |
| 3 | rs_percentile >= 95, close_vs_sma50 >= 0.05 | 68 | 67.7% | 2.23 | 2.8% |
| 4 | rs_percentile >= 95, close_vs_sma50 >= 5% | 68 | 67.7% | 2.23 | 2.8% |
| 5 | rs_percentile >= 95, volume_ratio >= 1.5 | 71 | 67.6% | 2.23 | 2.8% |
| 6 | rs_percentile >= 95, close_vs_sma50 >= 0.0 | 71 | 67.6% | 2.23 | 2.8% |
| 7 | rs_percentile >= 95, close_vs_sma50 >= 0.02 | 71 | 67.6% | 2.23 | 2.8% |
| 8 | rs_percentile >= 95, close_vs_sma50 >= 0% | 71 | 67.6% | 2.23 | 2.8% |
| 9 | rs_percentile >= 95, close_vs_sma50 >= 2% | 71 | 67.6% | 2.23 | 2.8% |
| 10 | rs_percentile >= 90, close_vs_sma50 >= 0.08 | 78 | 61.5% | 2.05 | 2.5% |

### Bottom 10 multi-feature combinations

| Rank | Condition | Trades | Win rate | PF | Expectancy |
|------|-----------|--------|----------|-----|------------|
| 1 | volume_ratio >= 3.0, close_vs_sma50 >= 0.08 | 8 | 62.5% | 0.16 | -3.0% |
| 2 | rs_percentile >= 95, volume_ratio >= 3.0 | 7 | 71.4% | 0.47 | -1.6% |
| 3 | rs_percentile >= 90, volume_ratio >= 3.0 | 7 | 71.4% | 0.47 | -1.6% |
| 4 | volume_ratio >= 3.0, close_vs_sma50 >= 0.05 | 12 | 66.7% | 0.57 | -1.3% |
| 5 | volume_ratio >= 3.0, close_vs_sma50 >= 0.02 | 12 | 66.7% | 0.57 | -1.3% |
| 6 | volume_ratio >= 3.0, close_vs_sma50 >= 0.0 | 12 | 66.7% | 0.57 | -1.3% |
| 7 | volume_ratio >= 3.0, volume_ratio >= 3.0 | 12 | 66.7% | 0.57 | -1.3% |
| 8 | volume_ratio >= 3.0, volume_ratio >= 2.5 | 12 | 66.7% | 0.57 | -1.3% |
| 9 | volume_ratio >= 3.0, volume_ratio >= 2.0 | 12 | 66.7% | 0.57 | -1.3% |
| 10 | volume_ratio >= 3.0, volume_ratio >= 1.5 | 12 | 66.7% | 0.57 | -1.3% |

## 5. Yearly in-sample stability

| Year | Trades | Win rate | PF | Expectancy |
|------|--------|----------|-----|------------|
| 2013 | 9 | 44.4% | 3.35 | 3.0% |
| 2014 | 11 | 54.5% | 2.30 | 1.4% |
| 2015 | 17 | 70.6% | 4.96 | 4.3% |
| 2016 | 14 | 71.4% | 1.70 | 0.8% |
| 2017 | 12 | 75.0% | 4.96 | 2.1% |
| 2018 | 12 | 50.0% | 1.17 | 0.5% |
| 2019 | 12 | 75.0% | 2.22 | 1.9% |
| 2020 | 16 | 50.0% | 0.59 | -2.7% |
| 2021 | 22 | 59.1% | 2.42 | 3.6% |
| 2023 | 8 | 87.5% | 33.02 | 6.1% |
| 2024 | 3 | 100.0% | ∞ | 9.2% |

## 6. Recommendation report

### Does Momentum Breakout have a positive edge?
- **Yes (aggregate):** Expectancy 2.2%, profit factor 2.05, Sharpe 1.02 over 136 trades.

### Is the edge stable across years?
- OOS aggregate (61 trades): expectancy 2.2%, PF 1.76.
- 67% of walk-forward test years (4/6) show PF ≥ 1 and non-negative expectancy.
- **Instability:** Flagged years: 2020, 2022.

### Is the edge regime dependent?
- **Mixed / insufficient separation** between regimes on this sample.

### Feature ranges that improve expectancy
- Best single-feature bin: **close_vs_sma200** in [0.2083 – 0.3312] (E=4.2%, n=27).
- Best combo: **rs_percentile >= 95, close_vs_sma50 >= 0.08** (E=2.9%, n=61).

### Feature ranges to filter out
- Worst single-feature bin: **volume_ratio** in [1.742 – 1.918] (E=-0.4%, n=27).
- Worst combo: **volume_ratio >= 3.0, close_vs_sma50 >= 0.08** (E=-3.0%, n=8).

## 7. Proposed configuration (data-driven, no code changes applied)

| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| RS percentile min | 80.0 | **95** | Lift threshold toward top-expectancy RS bins |
| Volume ratio min | 1.5 | **1.75** | Emphasize expansion days linked to winners |
| Stop | 10-day low | **Keep 10-day low** | Standard rule; no evidence here to widen |
| Target R | 2.0 | **Keep 2.0R** | Fixed plan geometry unless PF < 1 drives retest |
| Regime filter | None | **No hard regime gate; reduce size in NEUTRAL/RISK_OFF** | From regime table |

---
*Generated by `scripts/run_momentum_breakout_study.py` using existing `trade_planner.research` pipeline.*