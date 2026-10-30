# Systematic Quantitative Experiments & Hypothesis Log

## [2026-10-30] Experiment 01: Decoupled Two-Stage Gatekeeper & 20+ Microstructure Signals

### 1. Quantitative Hypothesis
- **Hypothesis**: The competition payoff matrix severely penalizes unselective long bets (losing $-0.30$ to $-1.00$ SOL on failed migrations). A decoupled two-stage architecture:
  1. **Stage 1 (Gatekeeper)**: Calibrated binary classification ($P(\max(X) \ge 3.0)$ via Platt scaling) enforcing ruthless selectivity (top 2-3 high-conviction trades per day, skipping the rest at $0.00$ cost).
  2. **Stage 2 (Quantile / Expected Utility Optimizer)**: Evaluates expected payoff curves over discrete levels $K = [3.0, 4.0, 5.0, 7.0, 10.0, 15.0]$ maximizing $\mathbb{E}[\Pi(k)]$ under monotonic probability constraints.
- **Microstructure Feature Hypotheses**:
  - Gini coefficient of early token holdings (`amm_gini`) and unique buyers ratio (`unique_buyers_ratio`) detect single-actor wash trading loops.
  - Second derivative of volume velocity ($d^2 V / dt^2$) separates real late acceleration ($t \in [45, 60\text{s}]$) from early snipes followed by dead liquidity.
  - Launch block slot sniping share (slots 0-2) captures predatory bot front-running.
  - Causal creator launch history (prior launches, median/max peak multiples, $3\times/5\times$ hit rates under strict $mt - 3600$ embargo) captures serial rug-pullers vs repeat winners.

### 2. Empirical Validation Results (7-Day Embargoed Walk-Forward Split: 2026-10-03 to 2026-10-09)

| Model Configuration | Feature Space | Trades/Day | Total Trades | Precision (%) | Total PnL (SOL) | Daily Sharpe ($\mu - 0.25\sigma$) | Positive Days |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Multi-Level (Uncalibrated)** | 74 features | ~40.0 | 280 | 17.1% | **-7.83 SOL** | -3.07 | 4 / 7 (57%) |
| **Two-Stage + Creation/Reputation Only** | 54 features | 2.0 | 14 | 21.4% | **-2.90 SOL** | -1.09 | 2 / 7 (29%) |
| **Two-Stage Gatekeeper + 20 New Microstructure Signals** | **103 features** | **2.0** | **14** | **28.6%** | **+3.85 SOL** | **-0.10** | **4 / 7 (57%)** |

### 3. Key Findings & Strategic Takeaways
1. **Dramatic PnL Reversal**: Shifting from indiscriminate multi-level trading (~40 trades/day) to the Two-Stage Gatekeeper (2 trades/day) delivered an incremental **`+11.68 SOL` gain** (from `-7.83 SOL` to `+3.85 SOL`).
2. **Break-Even Hurdle Cleared**: Baseline hit rate is $6.07\%$, break-even at $5\times$ requires $19.57\%$ precision. The Two-Stage Gatekeeper achieved **`28.6%` precision**, safely surpassing the friction barrier.
3. **Microstructure Feature Alpha**: Adding the high-frequency tape signals (Gini, $d^2 V / dt^2$, slot 0-2 sniping, wash trading ratio, and causal creator historical metrics) contributed **`+6.75 SOL`** over the creation-only baseline.
4. **Sweet-Spot Targets**: Rather than holding for $50\times$ moonshots that collapse to zero, conservative $4\times - 7\times$ targets provided the vast majority of cumulative realized gains.
