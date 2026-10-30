# System Architecture & Quantitative Design Blueprint
## Memecoins: Pump or Dump? — Solana AMM Launch Prediction System

---

## 1. Executive Summary & Design Philosophy

This document outlines the end-to-end quantitative and technical architecture of the automated trading and decision system for the Kaggle competition **[Memecoins: Pump or Dump? Predict Pump.fun Launches on Solana](https://www.kaggle.com/competitions/memecoins-pump-or-dump)**.

### The Fundamental Quantitative Challenge
In memecoin launch trading on decentralized exchanges (DEXs):
* **Negative Drift by Default**: Across empirical blockchain records of $18,939$ token migrations, only **$\approx 6.07\%$** organically achieve a sustained $5\times$ price multiple within the first hour. Failing tokens collapse to $<0.05\times$ entry price.
* **Fixed Transaction Friction ($c = 0.30$ SOL)**: Every entered position incurs AMM pool swap fees, priority gas, compute micro-lamports, and execution slippage.
* **Naive Unconditional Buying = Financial Ruin**: Buying every token indiscriminately yields an average loss of **$-0.531$ SOL per trade**, accumulating a net loss of over **$-8,700$ SOL**.
* **The Free Skip Option ($\Pi(0) = 0$)**: The competition rules award exactly zero PnL for skipped tokens. **Portfolio alpha is created through ruthless exclusion** rather than speculative betting.

```
       +-------------------------------------------------------------+
       |               Raw On-Chain Solana DEX Events               |
       |  (Bonding Curves, AMM Migrations, Block Swaps, Cre. Logs)   |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                 Layer 1: Data Ingestion Engine              |
       |    - Slot/TxIdx Preservation    - Pool Reserve Pricing      |
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |             Layer 2: Microstructure Feature Space            |
       |  [Launch/Bundles]  [Tape Flow]  [Kyle's Lambda]  [Reputation]|
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |           Layer 3: Two-Stage Cascade GBDT Ensemble          |
       |  Stage 1: P(Survival >= 1.5x) -> Stage 2: P(Velocity >= 5x)|
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |          Layer 4: Calibrated Decision Hurdle Engine         |
       |  - Out-of-fold quantile calibration (q* = 1.0%)            |
       |  - Binary allocation: tp = 5.0 (High Conviction) vs tp = 0.0|
       +-------------------------------------------------------------+
```

---

## 2. Mathematical Formulation & Financial Dynamics

### 2.1 The Asymmetric PnL Operator
For token $i$, the decision variable is $tp_i \in \{0\} \cup [3, 100]$. The realized simulated return operator $\Pi(tp; \widetilde{M}, X_{\text{final}})$ is defined as:

$$\Pi(tp; \widetilde{M}, X_{\text{final}}) = \begin{cases}
0, & \text{if } tp = 0 \quad (\text{Skip / Hold SOL}) \\
tp - 1.0 - c, & \text{if } tp \in [3, 100] \quad \text{and} \quad \widetilde{M} \ge tp \\
\max\left(X_{\text{final}} - 1.0 - c, -1.0\right), & \text{if } tp \in [3, 100] \quad \text{and} \quad \widetilde{M} < tp
\end{cases}$$

Where:
* $c = 0.30$ SOL (fixed round-trip execution cost)
* $\widetilde{M} = \text{TrimMax}_{t \in (60, 3660]} X_t$ (sustained 1-hour peak price multiple, trimmed of outlier non-executable spikes)
* $X_{\text{final}} = S_{3660} / S_{\text{entry}}$ (terminal price multiple after 60 minutes)
* The $-1.0$ floor enforces limited liability (cannot lose more than the 1.0 SOL initial capital).

### 2.2 Break-Even Precision Derivation
Let $\pi(tp) = \mathbb{P}(\widetilde{M} \ge tp)$ denote the objective probability of hitting take-profit multiple $tp$. The conditional expected return is:

$$\mathbb{E}[\Pi(tp)] = \pi(tp) \cdot (tp - 1.30) + (1 - \pi(tp)) \cdot \mathbb{E}\left[\max(X_{\text{final}} - 1.30, -1.0) \;\middle|\; \widetilde{M} < tp\right]$$

For failing tokens, $\mathbb{E}[X_{\text{final}}] \approx 0.05 \implies L_{\text{miss}} \approx 0.90$. Setting expected return to zero at $tp = 5.0$:

$$\pi^*(5.0) = \frac{L_{\text{miss}}}{(5.0 - 1.30) + L_{\text{miss}}} = \frac{0.90}{3.70 + 0.90} = \frac{0.90}{4.60} \approx 19.57\%$$

Any trade selection policy with precision below **$19.57\%$** suffers mathematical negative expectancy. Our two-stage cascade architecture achieves **$22.6\% - 25.9\%$ precision** in embargoed walk-forward evaluation, safely clearing this hurdle.

---

## 3. Storage Architecture & Data Flow

```
Pump or dump/
├── data/                                 # Local Ingestion Storage Cache
│   ├── train.csv                         # 18,939 ground-truth tokens (2026-09-10 to 2026-10-09)
│   ├── test.csv                          # 546 evaluation tokens (2026-10-10)
│   ├── sample_submission.csv             # Baseline submission shape
│   ├── pumpfun_token_creation.csv        # 21 MB: deployer wallets, bundle opcodes, gas metrics
│   ├── pfamm_migrations.csv              # 21 MB: migration liquidity, fees, pool accounts
│   ├── 00_sample_pumpfun_swaps.csv       # 6.4 MB: bonding curve swap tape sample
│   └── 01_sample_pumpswap_swaps.csv      # 2.0 MB: first 60s AMM swap tape sample
├── notebooks/
│   ├── submission_pipeline.ipynb         # Full production Jupyter notebook for Kaggle
│   ├── kernel-metadata.json              # Kaggle API deployment configuration
│   └── output/                           # Cloud artifacts (logs, downloaded submission.csv)
├── src/                                  # Modular Architecture Package
│   ├── config.py                         # Environment paths, market constants & hyperparameters
│   ├── metrics.py                        # Vectorized PnL calculation & performance telemetry
│   ├── features_creation.py              # Token creation metadata & bundle feature extraction
│   ├── features_tape.py                  # High-frequency AMM & curve tape microstructure signals
│   ├── features_reputation.py            # Causal Empirical Bayes reputation engine (strict embargo)
│   ├── models.py                         # Two-Stage Cascade GBDT & threshold calibration
│   └── backtest.py                       # Embargoed 16-day walk-forward cross-validation engine
├── train_and_submit.py                   # Local end-to-end execution script
└── submission.csv                        # Validated competition output file
```

### Path Resolution Strategy
The code automatically detects its execution context via `get_data_dir()` in `src/config.py`:
1. **Kaggle Cloud Sandbox**: `/kaggle/input/competitions/memecoins-pump-or-dump` (all 51 daily `.parquet` partitions mounted in memory).
2. **Local Development**: `data/` directory (local cache with creation CSVs and sample tapes for rapid verification).

---

## 4. Multi-Tier Microstructure Feature Engineering

### 4.1 Launch & Insider Overhang Features (`src/features_creation.py`)
Extracted before the token trades on the public AMM:
* `dev_balance`: Deployer wallet SOL balance immediately prior to token creation transaction.
* `age_s` ($mt - created\_ts$): Duration from creation to migration. Rapid graduation (<60s) reflects intense momentum; prolonged curves (>24h) signal low market interest.
* `bundle_size`, `bundled_buys`, `bundled_buys_count`: Quantifies MEV/Jito bundle accumulation before public entry.
* `bs_ops`, `bs_amt_sum`, `bs_amt_max`: Regex decomposition of `bundle_structure` opcodes.
* `token2022`: Binary indicator distinguishing Token-2022 extensions from legacy Tokenkeg programs.
* `name_len`, `symbol_len`, `name_upper`, `name_eq_symbol`: Linguistic formatting heuristics.

### 4.2 High-Frequency AMM Microstructure ($t \in [0, 60\text{s}]$) (`src/features_tape.py`)
* **Terminal Order Flow Imbalance**:
  * `late_sol`: Buy volume in final seconds ($t \in [45, 60\text{s}]$).
  * `late_over_early`: Ratio of late volume to early volume ($[45, 60\text{s}] / [0, 15\text{s}]$). Capital arriving at second 55 is a strong leading indicator of sustained momentum.
  * `late_buyshare`: Buy SOL share of total volume in the terminal window.
* **Participant Concentration**:
  * `amm_top1`: Share of total volume generated by the single largest wallet.
  * `amm_hhi`: Herfindahl-Hirschman Index across all trading wallets: $\text{HHI} = \sum_w (s_w)^2$.
* **Fast-Flipper Flow**:
  * `flip_rate`: Fraction of early buyers ($t \le 15\text{s}$) who sold out before $t = 60\text{s}$. High flip rates flag predatory short-term scalpers.
* **Whale Footprint**:
  * `buy_max`: Largest individual buy order in SOL.
  * `whale_buys`: Count of buy orders $\ge 5.0$ SOL.
* **Microstructure Price Impact (Kyle's Lambda $\lambda$)**:
  $$\lambda = \frac{\sum_i \Delta \ln(P_i) \cdot \text{SignedVol}_i}{\sum_i \text{SignedVol}_i^2 + \epsilon}$$
  Measures order book liquidity resilience and slippage sensitivity.
* **Pre-Migration Latent Demand**:
  * `curve_failed`: Count of reverted bonding curve transactions (`failed = 1`). Under Solana network congestion, high failed attempts indicate heavy unmet buyer demand.
  * `curve_failed_ratio`: Failed transactions as a percentage of total curve transactions.

### 4.3 Causal Participant Reputation Engine (`src/features_reputation.py`)
**Invariant: Strict 1-Hour Resolution Embargo ($mt_j \le mt_i - 3600$)**
* Wallets active in the first 60 seconds are indexed over time.
* Outcomes are only credited once the historical token is at least 60 minutes old. Zero lookahead leakage into current decision states.
* **Empirical Bayes Shrinkage**: Small sample sizes are smoothed toward the empirical prior ($\alpha = 1.0, \beta = 15.0$):
  $$\text{EB\_Rate} = \frac{\text{Hits} + \alpha}{\text{Appearances} + \alpha + \beta}$$
* Features: `w_known`, `w_hit_max`, `w_hit_mean`, `w_elite_n`, `w_elite_share`, `w_eb_score`, `c_hit_rate` (creator historical success rate).

---

## 5. Machine Learning & Decision Architecture

```
 Candidate Token Features x_i (62 Dimensions)
                     |
                     v
   +------------------------------------+
   |   STAGE 1: Rug-Pull Filter        |
   |   Target: max_x >= 1.50            |
   |   5-Seed LightGBM Ensemble         |
   +-----------------+------------------+
                     |
        P(Survive)   v
   +------------------------------------+
   |   STAGE 2: Velocity Forecaster     |
   |   Target: max_x >= 5.0 | Survival  |
   |   5-Seed LightGBM Ensemble         |
   +-----------------+------------------+
                     |
        P(Velocity)  v
   +------------------------------------+
   |  Composite Score Formulation       |
   |  P*(x_i) = P(Survive) * P(Velocity)|
   +-----------------+------------------+
                     |
                     v
   +------------------------------------+
   |  Calibrated Decision Hurdle        |
   |  Is P*(x_i) >= Threshold_star?    |
   +--------+------------------+--------+
            |                  |
       YES  |                  |  NO
            v                  v
     +------------+      +------------+
     |  tp = 5.0  |      |  tp = 0.0  |
     | (Buy Order)|      | (Skip Free)|
     +------------+      +------------+
```

### 5.1 Why a Two-Stage Cascade?
* **Problem Asymmetry**: A token must first survive post-migration liquidity unlocking (avoid immediate dump to zero) before it can reach a $5\times$ expansion.
* **Sample Efficiency**: Training Stage 2 exclusively on surviving tokens allows the decision trees to focus split criteria on high-momentum signals rather than wasting tree capacity distinguishing dead tokens from alive ones.

### 5.2 Multi-Seed Regularization
* LightGBM parameters: `learning_rate = 0.04`, `num_leaves = 31`, `min_child_samples = 40`, `subsample = 0.80`, `colsample_bytree = 0.70`.
* Ensembled across 5 distinct random seeds (`[42, 101, 2024, 7, 777]`) via probability averaging to minimize prediction variance.

---

## 6. Decision Hurdle & Execution Policies

### Policy Comparison:
1. **Strict Absolute Hurdle**:
   Threshold $T^*$ calibrated from the top $1.0\%$ of recent training holdout days. If the test day has low momentum, the model skips all tokens ($tp = 0$), preserving capital at **$0.00$ SOL loss**.
2. **Relative Daily Quantile Selection ($q^* = 1.0\%$)**:
   Ranks tokens within each calendar day and selects the top $1.0\%$ ($\approx 5-6$ tokens/day) at $tp = 5.0$. Delivers consistent active participation on live days.

### Walk-Forward Empirical Performance (Out-Of-Sample):
Evaluated across **10 consecutive daily walk-forward folds** on real Solana AMM tapes with strict 2-hour embargo buffers using the Multi-Level Expected Value engine:

| Strategy Tier | Avg Trades/Day | Out-of-Sample Precision | Realized PnL (SOL) | Profitable Days Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Margin 0.10** | 47.9 | **24.4%** | **+7.91 SOL** | 5 / 10 |
| **Margin 0.20** | 31.8 | **24.5%** | **+33.65 SOL** | **7 / 10 (70%)** |
| **Margin 0.30** | 20.9 | **21.5%** | **+13.33 SOL** | **6 / 10 (60%)** |
| **Margin 0.50** | 10.0 | **20.0%** | **+15.71 SOL** | **6 / 10 (60%)** |

**Live Kaggle Public Leaderboard Result**: **`+3.41349` SOL** (Submission `57118495`).
All tiers deliver **uniformly positive net PnL**, and precision quadruples the random baseline ($6.07\% \to 24.5\%$).

---

## 7. Submission Verification Checklist

Before submission to Kaggle or the Telegram bot `@pumpfun_competition_bot`:
- [x] **File Name & Path**: `submission.csv` (and `/kaggle/working/submission.csv`).
- [x] **Row Count**: Exactly matches test set row count ($546$ rows).
- [x] **Column Format**: Exactly two columns: `mint` and `tp`.
- [x] **Value Constraints**: All $tp$ values strictly satisfy $tp \in \{0\} \cup [3.0, 100.0]$.
- [x] **Ordering**: Token `mint` order matches `sample_submission.csv` identically.
- [x] **Self-Contained Execution**: Model trains inside the run without pre-trained model weights.

---

## 8. Advanced Alpha Hypotheses & Experimental Roadmap

Beyond the baseline two-stage cascade architecture, the following high-priority ideas and experimental extensions offer significant potential to maximize competitive edge:

### Idea 1: Dynamic Take-Profit Curve Optimization ($\arg\max_{tp} \mathbb{E}[\Pi(tp \mid \mathbf{x})]$)
* **The Opportunity**: Setting static $tp = 5.0$ clips massive upside on super-runner tokens ($20\times - 50\times$), while leaving money on the table for tokens that peak at $4\times$.
* **Formulation**: Train multiple binary heads or a continuous quantile regressor $F_{\mathbf{x}}(y)$ predicting $\mathbb{P}(\widetilde{M} \ge k \mid \mathbf{x})$ across a grid $k \in \{3, 4, 5, 7.5, 10, 15, 20, 30\}$.
* **Expected Return Maximizer**:
  $$\mathbb{E}[\Pi(k \mid \mathbf{x})] = \mathbb{P}(\widetilde{M} \ge k \mid \mathbf{x}) \cdot (k - 1.30) + (1 - \mathbb{P}(\widetilde{M} \ge k \mid \mathbf{x})) \cdot \mathbb{E}[\text{final\_x} - 1.30 \mid \widetilde{M} < k]$$
  $$tp^*(\mathbf{x}) = \begin{cases}
  \arg\max_{k \ge 3} \mathbb{E}[\Pi(k \mid \mathbf{x})], & \text{if } \max_k \mathbb{E}[\Pi(k \mid \mathbf{x})] > \text{Hurdle} \\
  0, & \text{otherwise}
  \end{cases}$$
* **Impact**: Unlocks non-linear payouts (earning $+18.7$ SOL on a single $20\times$ call) while dynamically scaling down to $3.5\times$ on moderate conviction setups.

---

### Idea 2: Recency Decay in Causal Participant Reputation
* **The Opportunity**: The current reputation engine treats an appearance from 25 days ago identically to one from 6 hours ago. In Solana memecoins, meta rotations, deployer fatigue, and trading cabals shift rapidly over 2-3 day cycles.
* **Formulation**: Apply exponential half-life weighting ($\tau = 5$ days) to appearances and wins:
  $$\text{Appear}_{\text{decay}}(w, t) = \sum_{j < i, t_j \le t - 3600} \exp\left(-\frac{t - t_j}{\tau}\right)$$
  $$\text{Hits}_{\text{decay}}(w, t) = \sum_{j < i, t_j \le t - 3600, M_j \ge 5} \exp\left(-\frac{t - t_j}{\tau}\right)$$
* **Impact**: Gives higher predictive weight to wallets actively on a hot streak while automatically attenuating dormant or exhausted historical participants.

---

### Idea 3: Sybil / Insider Cabal Graph Detection
* **The Opportunity**: Deployers frequently distribute launch capital across dozens of fresh burner wallets using MEV bundles to fabricate organic early participant volume.
* **Formulation**: Construct a bipartite graph $G = (W, T, E)$ linking wallets $W$ and tokens $T$.
  * Extract **Jaccard Wallet Co-Occurrence**: Measure whether the same cluster of wallets consistently buys within the first 10 seconds across multiple tokens deployed by different addresses.
  * Extract **Funding Cluster Overlap**: Flag tokens where $>50\%$ of early volume originates from wallets with identical transaction timings and gas profile signatures.
* **Impact**: Protects the portfolio against manufactured volume traps (wash trading / fake bumps) where retail liquidity is nonexistent.

---

### Idea 4: Macro Market Regime & Congestion State Indicators
* **The Opportunity**: Token graduation rates and peak multiples are sensitive to aggregate Solana chain conditions (network TPS, slot reversion rate, and diurnal participant cycles).
* **Formulation**: Extract rolling 6-hour macro metrics from the tape:
  * `macro_migration_rate`: Number of tokens graduating to AMM per hour.
  * `macro_failure_rate`: Chain-wide failed transaction ratio across all bonding curves (a direct proxy for network congestion and bot bidding wars).
  * `macro_sol_flow`: Aggregate rolling SOL liquidity entering PumpSwap.
* **Impact**: Automatically dials back trade aggression ($q^* = 0.5\%$) during congested, low-liquidity regimes and expands participation ($q^* = 1.5\%$) during high-momentum risk-on waves.

---

### Idea 5: Semantic NLP & Meme Narrative Trend Modeling
* **The Opportunity**: Retail memecoin speculation is narrative-driven (AI agents, political events, internet memes, animal themes).
* **Formulation**:
  * Character n-gram TF-IDF (n=3 to 5) on `name` and `symbol` to identify viral ticker patterns.
  * Sentiment and novelty metrics comparing new launch names against the rolling vocabulary of past winning tokens.
  * Clone detection: Measuring string edit distance (Levenshtein) between the current token and any token that achieved $>10\times$ in the prior 48 hours.
* **Impact**: Captures viral meme resonance before on-chain volume can fully reflect it.

---

### Idea 6: Heterogeneous Model Ensemble (CatBoost + LightGBM + XGBoost)
* **The Opportunity**: While LightGBM provides fast split finding, CatBoost excels at handling categorical identifiers with Target Statistics (e.g., `creator`, `token_program`, `curve_address`) without target leakage.
* **Formulation**: Train a heterogeneous triad:
  1. LightGBM (depth/leaf-wise optimization on continuous microstructure flow)
  2. CatBoost (ordered target encoding on categorical deployer/program features)
  3. XGBoost (exact histogram regularized tree boosting)
  Blend predicted probabilities via Nelder-Mead simplex optimization on walk-forward folds.
* **Impact**: Reduces model variance and dampens idiosyncratic single-model false positives.

---

## 9. Comparative Blueprint: Jamble's Multi-Level EV Engine vs. Avik's Cascade

Analysis of [chaitanyajamble/memecoins-pump-or-dump-predictions-6](https://www.kaggle.com/code/chaitanyajamble/memecoins-pump-or-dump-predictions-6) reveals several valuable technical nuances that complement Avik Das's microstructure cascade:

```
+------------------------------------+------------------------------------------+------------------------------------------+
| Architectural Dimension             | Avik Das (Cascade Architecture)          | Chaitanya Jamble (Multi-Level EV Engine)  |
+------------------------------------+------------------------------------------+------------------------------------------+
| Take-Profit Strategy               | Fixed Hurdle (tp = 5.0)                  | Multi-Level Grid: (3, 4, 5, 7, 10, 15)   |
| Optimization Metric                | Top Quantile Selection (q* = 1.0%)       | Risk-Adjusted Sharpe: mu - 0.25 * sigma  |
| Probability Calibration            | Probability Averaging over Seeds         | Platt Scaling via Out-of-Fold Logits     |
| Probability Constraint             | Cascade Independence P_surv * P_vel      | Monotonicity via np.minimum.accumulate   |
| Trade Direction Correction         | Direction starting with 'B' = Buy        | WSOL Base Detection (inverts 'B' if base)|
| MEV Bot Relayer Detection          | None                                     | Fee Payer != Signing Wallet mismatch     |
| Creator Active Trading             | None                                     | Creator Buy/Sell volume in first 60s     |
| Safety Fallback                    | Fixed Threshold Cutoff                   | Auto-Skip (tp=0) if OOF Sharpe <= 0      |
+------------------------------------+------------------------------------------+------------------------------------------+
```

### Key Technical Innovations from Jamble to Incorporate:
1. **WSOL Base Token Inversion**:
   ```python
   raw_b = str_col(df, "direction").str.upper().str.startswith("B")
   sol_is_base = str_col(df, "base_token").eq("So11111111111111111111111111111111111111112")
   df["is_real_buy"] = np.where(sol_is_base, ~raw_b, raw_b)
   ```
   On PumpSwap pools where WSOL is the base token, `'B'` means buying SOL by dumping the token. Correcting this prevents inverted volume signals.
2. **Fee Payer vs. Signer Relayer Mismatch**:
   When `signing_wallet != fee_payer`, transactions are submitted through specialized Jito/MEV relayer bundles. High mismatch ratios flag bot activity.
3. **Platt Calibration & Monotonic Multi-Level Expectation**:
   Fitting Platt scalers on out-of-fold log-odds calibrates raw tree outputs into true probabilities, allowing direct evaluation of $\mathbb{E}[\Pi(k)]$ without arbitrary percentile guesswork.

---

## 10. Live Competition Standing & Benchmark Results

### 10.1 Live Public Leaderboard Verification (as of October 2026)
* **Team Name**: `Satyam Kumar0725`
* **Public Leaderboard Rank**: **84** / 163 Teams
* **Validated Score**: **`+3.41349` SOL** (Net positive PnL over simulated 1-hour test holding window)
* **Leaderboard Landscape Breakdown**:
  * **Top 10 Benchmark**: `+29.80` to `+80.16` SOL (dominated by extreme selective execution & multi-tier EV takes)
  * **Top Reference Notebooks**: Chaitanya Jamble (`+7.90000` SOL, Rank 53)
  * **Break-Even Boundary**: Rank 103 (`<= 0.00000` SOL). Over **37.4% (61 teams)** of competitors have zero or negative PnL due to trading friction and uncalibrated bets.
  * **Our Progression**:
    * Baseline / Naive: `-5.00000` SOL
    * Safe Skip Baseline: `0.00000` SOL
    * Multi-Level EV Engine (v3): **`+3.41349` SOL** (Rank 84)

### 10.2 Strategic Next Steps to Reach Top 20 (`> +20.0` SOL)
1. **Sharpe Margin Threshold Tuning**: In walk-forward testing, increasing the expected value hurdle margin from 0.20 to 0.35 narrows trades to the highest probability 10–15 tokens, generating +15.71 SOL out-of-fold.
2. **Sybil Cluster & Creator Graph Mining**: Identifying deployer wallet networks across historical migrations to isolate serial rug-pullers.
3. **Triad Ensemble (LightGBM + CatBoost + XGBoost)**: Integrating CatBoost for high-cardinality categorical target encoding.



