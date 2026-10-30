---
name: pumpfun-eval
description: Evaluates model predictions against the exact competition PnL metric with -0.3 penalty.
---

# Pump.fun Memecoin Evaluation & Guardrails Protocol

When testing any model, feature addition, or take-profit threshold:
1. **NEVER evaluate on accuracy, ROC-AUC, or MSE**. Always evaluate using the custom PnL function:
   - Hit (`max_multiple >= tp`): `PnL = tp - 1.0 - 0.3`
   - Miss (`max_multiple < tp`): `PnL = max(end_price - 1.0 - 0.3, -1.0)`
   - Skip (`tp == 0`): `PnL = 0.0`
2. **Enforce strict 1-hour embargoes** on wallet and creator reputation features to prevent lookahead leakage (`target_mt <= mt_train_max - 3600`).
3. **Validate only using rolling daily time-series splits** (no randomized k-folds).
4. **Never allow editing `evaluate_cv.py`** (it is read-only).
5. **Decouple the decision architecture into two stages**:
   - Stage 1: Precision Gatekeeper ($P(\max(X) \ge 3.0)$) selecting 2-3 high-conviction trades/day and skipping the rest.
   - Stage 2: Expected Utility Optimizer maximizing $\mathbb{E}[\text{PnL}(k)]$ across discrete levels $K = [3, 4, 5, 7, 10, 15]$.
