# Momentum Breakout — Robustness Validation

**Universe:** AAPL, MSFT, NVDA, META, AMZN  
**Period:** 2011-06-02 → 2024-12-31  
**OOS window:** signal years [2019, 2020, 2021, 2022, 2023, 2024]  
**Grid size:** 256 parameter combinations  

## 1. Parameter sensitivity (top 15 by robustness score)

| Rank | Params | Total | OOS | WR | E | PF | MaxDD | OOS E | OOS PF | OOS MaxDD | Year%+ | Score |
|------|--------|-------|-----|----|----|-----|-------|-------|--------|-----------|--------|-------|
| 1 | RS95_V1.3_R1.5_S5 | 133 | 61 | 64% | 2.9% | 2.74 | 50% | 4.2% | 3.16 | 50% | 80% | 71 |
| 2 | RS95_V1.3_R2.0_S5 | 133 | 61 | 63% | 3.0% | 2.74 | 50% | 4.4% | 3.30 | 50% | 80% | 71 |
| 3 | RS95_V1.3_R2.5_S5 | 133 | 61 | 63% | 3.1% | 2.77 | 50% | 4.6% | 3.39 | 50% | 80% | 71 |
| 4 | RS80_V1.3_R2.0_S5 | 239 | 112 | 60% | 2.2% | 2.21 | 54% | 2.9% | 2.41 | 54% | 100% | 60 |
| 5 | RS80_V1.3_R2.5_S5 | 239 | 112 | 60% | 2.2% | 2.25 | 54% | 3.1% | 2.50 | 54% | 100% | 60 |
| 6 | RS80_V1.3_R1.5_S5 | 239 | 112 | 60% | 2.1% | 2.18 | 54% | 2.7% | 2.30 | 54% | 80% | 56 |
| 7 | RS80_V1.3_R1.5_S10 | 239 | 112 | 61% | 2.0% | 2.01 | 64% | 2.9% | 2.23 | 64% | 80% | 56 |
| 8 | RS80_V1.3_R1.5_S15 | 239 | 112 | 63% | 2.2% | 2.16 | 67% | 3.0% | 2.26 | 67% | 80% | 56 |
| 9 | RS80_V1.3_R1.5_S20 | 239 | 112 | 64% | 2.3% | 2.21 | 70% | 3.0% | 2.20 | 70% | 80% | 56 |
| 10 | RS80_V1.3_R2.0_S10 | 239 | 112 | 61% | 2.1% | 2.03 | 64% | 2.9% | 2.23 | 64% | 80% | 56 |
| 11 | RS80_V1.3_R2.0_S15 | 239 | 112 | 63% | 2.2% | 2.15 | 69% | 2.9% | 2.20 | 69% | 80% | 56 |
| 12 | RS80_V1.3_R2.0_S20 | 239 | 112 | 64% | 2.3% | 2.17 | 72% | 2.8% | 2.11 | 72% | 80% | 56 |
| 13 | RS80_V1.3_R2.5_S10 | 239 | 112 | 61% | 2.1% | 2.06 | 64% | 2.9% | 2.24 | 64% | 80% | 56 |
| 14 | RS80_V1.3_R2.5_S15 | 239 | 112 | 63% | 2.2% | 2.13 | 69% | 2.9% | 2.19 | 69% | 80% | 56 |
| 15 | RS80_V1.3_R2.5_S20 | 239 | 112 | 64% | 2.2% | 2.15 | 72% | 2.8% | 2.11 | 72% | 80% | 56 |

### Current production config (baseline)

- **Params:** RS80_V1.5_R2.0_S10  
- Total 136 / OOS 61 | E 2.2% | OOS E 2.2% | PF 2.05 | OOS PF 1.76 | MaxDD 61% | Score **56**

### Best robust parameter set (grid winner)

- **Params:** RS95_V1.3_R1.5_S5  
- Total 133 / OOS 61 | OOS E 4.2% | OOS PF 3.16 | OOS MaxDD 50% | Score **71**

## 2. Robustness scoring methodology

Rewards: positive OOS expectancy (+25), OOS PF ≥ 1.2 (+20), OOS trades ≥ 50 (+20), OOS MaxDD < 35% (+15), year consistency (+0–20).  
Penalties: OOS trades < 30 (−30), OOS PF < 1 (−15), MaxDD > 50% (−25), one-year profit share > 55% (−20), unstable vs RS/vol neighbors (−8).  

## 3. Volume filter investigation (baseline trades)

| Band | Trades | Win rate | Expectancy | Profit factor |
|------|--------|----------|------------|---------------|
| All trades (vol >= setup min) | 136 | 64% | 2.2% | 2.05 |
| volume_ratio >= 1.5 | 136 | 64% | 2.2% | 2.05 |
| 1.5 <= volume < 1.75 | 57 | 68% | 3.2% | 2.87 |
| 1.9 <= volume <= 2.5 | 34 | 59% | 2.0% | 1.80 |
| volume_ratio >= 3.0 | 12 | 67% | -1.3% | 0.57 |

**Volume conclusion:**  
- **Avoid volume_ratio ≥ 3.0** — negative expectancy (−1.3%, PF 0.57, n=12).  
- Band **1.5–1.75** shows higher expectancy (+3.2%, n=57) but that is a *subset* of trades already passing vol ≥ 1.5; **raising volume_min to 1.75 would drop those 57 trades**, not isolate them.  
- **Do not raise volume_min to 1.75** — grid robustness winner uses **1.3** (looser), and post-hoc band analysis does not support a higher floor.  
- Optional research filter: **cap volume_ratio at 3.0** (exclude climax days), not a higher minimum.  

## 4. Drawdown investigation (63.9% sequential trade curve)

- **Sequential compounded max DD:** **60.9%** (trade-by-trade curve, all symbols).  
- **Largest loss streak:** 8 consecutive losing exits (from ~2018-07-26).  
- **2020 OOS year:** PF 0.59, E −2.7% — primary regime stress year.  
- **Max consecutive losses:** 8 (from ~2018-07-26)  
- **Same-symbol overlapping trades:** 94 pairs  
- **Portfolio-wide overlapping trades:** 212 pairs  

**By symbol (drawdown window):**  
- AMZN: 1 trades, sum return -1.4%  
- NVDA: 3 trades, sum return 14.0%  

**By regime (drawdown window):**  
- RISK_OFF: 4 trades, sum return 12.6%  

**Worst trades in window:**  

| Symbol | Signal | Exit | Return | Regime | Vol ratio |
|--------|--------|------|--------|--------|-----------|
| AMZN | 2020-04-30 | 2020-05-29 | -1.4% | RISK_OFF | 1.66 |
| NVDA | 2020-05-18 | 2020-06-16 | 1.6% | RISK_OFF | 1.65 |
| NVDA | 2020-05-22 | 2020-06-22 | 4.6% | RISK_OFF | 2.05 |
| NVDA | 2020-05-15 | 2020-06-15 | 7.8% | RISK_OFF | 2.22 |

**Drawdown drivers:** Sequential compounding across **all symbols** treats each trade as full capital commitment; overlapping positions and 2020-style loss clusters inflate path risk vs per-trade edge.  

**Mandatory risk controls before production alerts:**  
1. **Max open positions** (e.g. 3–5) — limits overlap inflation.  
2. **Max 1 active trade per symbol** — eliminates same-symbol overlap.  
3. **Portfolio risk cap** (e.g. 1–2% risk per trade, ≤6% aggregate open risk).  
4. **Circuit breaker:** pause after 3–4 consecutive losses or −10% rolling 20-trade window.  
5. **Correlation throttle:** reduce size when ≥3 mega-cap tech signals same week.  

## 5. Final recommendations

### Current vs best robust config

| | Current | Best robust |
|---|---------|-------------|
| RS min | 80.0 | 95.0 |
| Volume min | 1.5 | 1.3 |
| Target R | 2.0 | 1.5 |
| Stop lookback | 10 | 5 |
| Robustness score | 56 | 71 |
| OOS trades | 61 | 61 |
| OOS expectancy | 2.2% | 4.2% |

### Should production config change now?
**Keep production config unchanged (RS80 / vol 1.5 / 2R / 10-day stop)** for now.

| Concern | Baseline | Grid winner (RS95_V1.3_R1.5_S5) |
|---------|----------|----------------------------------|
| Robustness score | 56 | 71 |
| OOS MaxDD | 61% | 50% (still **> 35%** target) |
| Parameter stability | Center of RS/vol grid | Edge cell (RS95 + tight 5-day stop) |
| Volume | Matches research default | **1.3** — looser, not 1.75 |

The grid winner improves OOS metrics but still fails drawdown targets and sits on a **narrow** parameter corner (5-day stop, RS 95). That pattern is consistent with **overfitting risk** on n=61 OOS trades. Paper-trade alternative configs; do **not** ship RS95/V1.3/S5 without more history and portfolio-level risk simulation.

### Ready for user-facing alerts?
**Conditional yes** — edge is positive OOS, but **only with mandatory risk controls** (position limits, per-symbol cap, circuit breaker). Not ready for unconstrained alert firing.

---
*Generated by `scripts/run_momentum_robustness_study.py` — research only.*