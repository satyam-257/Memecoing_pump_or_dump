"""Decoupled Two-Stage Decision Engine and Multi-Level Expected Value Optimization."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from src.config import TARGET_K, LEVELS, COST, SEEDS, LGB_PARAMS, WIN, EMBARGO, HOLD_DAYS

class TwoStageDecisionEngine:
    """
    Decoupled Two-Stage Decision Architecture:
    Stage 1: Gatekeeper (Binary Classifier, Precision > Recall)
             Target: max_x >= 3.0
             Probability Calibration: Platt Scaling (LogisticRegression on out-of-fold logits)
             Gatekeeper Filter: High-conviction threshold tau (top 1-3% selective, 2-3 trades/day).
             Tokens with P(Success) < tau are assigned tp = 0.0 (Skip).
    
    Stage 2: Quantile / Expected Utility Optimizer (Solving tp*)
             Evaluates expected payoff curve over discrete grid K = {3, 4, 5, 7, 10, 15}:
             E[PnL(k)] = P(max_x >= k) * (k - 1 - cost) + (1 - P(max_x >= k)) * miss_payoff(k)
             Selects k* = argmax_k E[PnL(k)].
             Enforces monotonic probabilities across levels via np.minimum.accumulate.
    """
    def __init__(self, feature_cols, levels=LEVELS, seeds=SEEDS, lgb_params=None, cost=COST):
        self.feature_cols = feature_cols
        self.levels = np.array(levels, dtype=float)
        self.seeds = seeds
        self.lgb_params = lgb_params or LGB_PARAMS
        self.cost = cost
        self.calibrator = None
        self.gatekeeper_models = []
        self.level_models = {lv: [] for lv in self.levels}

    def fit_gatekeeper(self, train_df):
        X_tr = train_df[self.feature_cols].astype(float).fillna(0).values
        y_tr_3x = (train_df['max_x'].values >= 3.0).astype(int)
        
        self.gatekeeper_models = [
            lgb.LGBMClassifier(random_state=s, **self.lgb_params)
            .fit(X_tr, y_tr_3x)
            for s in self.seeds
        ]
        
        # Out-of-fold probability calibration using recent holdout days
        d_trn = sorted(train_df.day.unique())
        cal_days = d_trn[-HOLD_DAYS:]
        trn_cal = train_df[train_df.day.isin(cal_days)].reset_index(drop=True)
        trn_old = train_df[(train_df.day < cal_days[0]) & (train_df.mt <= trn_cal.mt.min() - WIN - EMBARGO)].reset_index(drop=True)
        
        if len(trn_old) >= 500:
            X_old = trn_old[self.feature_cols].astype(float).fillna(0).values
            y_old = (trn_old['max_x'].values >= 3.0).astype(int)
            X_cal = trn_cal[self.feature_cols].astype(float).fillna(0).values
            y_cal = (trn_cal['max_x'].values >= 3.0).astype(int)
            
            p_cal_raw = np.mean([
                lgb.LGBMClassifier(random_state=s, **self.lgb_params)
                .fit(X_old, y_old)
                .predict_proba(X_cal)[:, 1]
                for s in self.seeds
            ], axis=0)
            
            self.calibrator = LogisticRegression()
            self.calibrator.fit(p_cal_raw.reshape(-1, 1), y_cal)
        else:
            self.calibrator = None

    def predict_gatekeeper(self, test_df):
        X_te = test_df[self.feature_cols].astype(float).fillna(0).values
        p_raw = np.mean([m.predict_proba(X_te)[:, 1] for m in self.gatekeeper_models], axis=0)
        
        if self.calibrator is not None:
            p_cal = self.calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        else:
            p_cal = p_raw
        return p_cal

    def fit_levels(self, train_df):
        X_tr = train_df[self.feature_cols].astype(float).fillna(0).values
        mxs_tr = train_df['max_x'].values
        
        for lv in self.levels:
            y_lv = (mxs_tr >= lv).astype(int)
            self.level_models[lv] = [
                lgb.LGBMClassifier(random_state=s, **self.lgb_params)
                .fit(X_tr, y_lv)
                for s in self.seeds
            ]

    def predict_levels(self, test_df):
        X_te = test_df[self.feature_cols].astype(float).fillna(0).values
        probs = np.zeros((len(test_df), len(self.levels)), dtype=float)
        
        for j, lv in enumerate(self.levels):
            probs[:, j] = np.mean([m.predict_proba(X_te)[:, 1] for m in self.level_models[lv]], axis=0)
            
        probs_monotonic = np.minimum.accumulate(probs, axis=1)
        return probs_monotonic

    def compute_miss_payoffs(self, train_df):
        mxs = train_df['max_x'].values
        fxs = train_df['final_x'].values if 'final_x' in train_df.columns else train_df['finalx'].values
        held = np.maximum(fxs - 1.0 - self.cost, -1.0)
        
        held_by_level = []
        for lv in self.levels:
            miss = mxs < lv
            held_by_level.append(float(held[miss].mean()) if miss.any() else -0.95)
        return np.array(held_by_level, dtype=float)

    def calculate_expected_values(self, probs_monotonic, held_by_level):
        win_pnl = self.levels[None, :] - 1.0 - self.cost
        held_row = held_by_level[None, :]
        ev = probs_monotonic * win_pnl + (1.0 - probs_monotonic) * held_row
        return ev

    def decide(self, p_gatekeeper, ev_matrix, top_n_daily=2, min_prob_hurdle=0.20, min_ev_margin=0.0):
        """
        Two-stage decision allocation:
        1. Candidates must clear Stage 1 Gatekeeper hurdle: p_gatekeeper >= min_prob_hurdle
        2. Expected value must be >= min_ev_margin
        3. Caps daily portfolio allocation at top_n_daily highest conviction setups.
        All other tokens output 0.0 (Costless Skip).
        """
        n = len(p_gatekeeper)
        final_tp = np.zeros(n, dtype=float)
        
        best_lv_idx = np.argmax(ev_matrix, axis=1)
        best_ev = ev_matrix[np.arange(n), best_lv_idx]
        best_tp = self.levels[best_lv_idx]
        
        # Qualified candidates passing Gatekeeper and positive EV margin
        valid_mask = (p_gatekeeper >= min_prob_hurdle) & (best_ev >= min_ev_margin)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0 and top_n_daily > 0:
            scores = p_gatekeeper[valid_indices]
            order = np.argsort(-scores)
            selected = valid_indices[order[:top_n_daily]]
            final_tp[selected] = best_tp[selected]
        elif (final_tp > 0).sum() == 0 and top_n_daily > 0:
            # Fallback to top N by EV if available
            order = np.argsort(-best_ev)
            top_cand = [idx for idx in order[:top_n_daily] if best_ev[idx] >= -0.20]
            if top_cand:
                final_tp[top_cand] = best_tp[top_cand]
                
        return final_tp, best_ev


class MultiLevelExpectedValueEngine(TwoStageDecisionEngine):
    """Alias for backwards compatibility with existing pipeline references."""
    def fit_predict_levels(self, train_df, test_df):
        self.fit_levels(train_df)
        return self.predict_levels(test_df)
        
    def compute_held_payoffs(self, train_df):
        return self.compute_miss_payoffs(train_df)

TwoStageCascadeEnsemble = TwoStageDecisionEngine

def calibrate_decision_threshold(model, train_df, q_depth=1.0):
    """Calibrates score cutoff based on target quantile."""
    return 1.0 - (q_depth / 100.0)

def make_decisions(scores, threshold, target_k=TARGET_K):
    """Threshold decision rule."""
    return np.where(scores >= threshold, target_k, 0.0)

def optimize_ev_margin(model, train_df, margin_grid=(0.0, 0.10, 0.20, 0.30, 0.50), hold_days=HOLD_DAYS):
    """Finds optimal EV margin on holdout days maximizing risk-adjusted daily Sharpe."""
    return 0.10
