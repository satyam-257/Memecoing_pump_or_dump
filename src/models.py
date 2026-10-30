"""Decoupled Two-Stage Decision Engine with Feature Subsets and Causal Per-Token Execution."""
import re
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from src.config import TARGET_K, LEVELS, COST, SEEDS, LGB_PARAMS, WIN, EMBARGO, HOLD_DAYS

# Regex patterns for feature subset decoupling
STAGE1_PATTERN = r'bs_|bundle|dev_balance|is_micro|token2022|gas|instruction|lookup|creation|is_ipfs|is_bot|is_deployer|custom_domain|name_|symbol_|amm_gini|amm_hhi|amm_top1|unique_buyers|trades_per|wallet_max|fee_mismatch|creator_sell|creator_buy|early_dump|flip_rate|slot_snipe|c_launches|c_hit_3x|c_hit_5x|curve_failed'
STAGE2_PATTERN = r'w_shrunk|w_decay|w_elite|w_known|d2v|rel_flow|flow_accel|bp_|late_|fresh_wallets|amm_uw|amm_sol|buysell|net_sol|buy_max|whale_buys|ret60|runup|at_high|drawdown|vol|depth|impact|curve_dur|curve_sol|migration_pop|c_max_peak|c_median_peak|c_last_peak|c_shrunk_rate'

class TwoStageDecisionEngine:
    """
    Decoupled Two-Stage Decision Architecture:
    
    1. Stage 1 (Gatekeeper / Rug Filter):
       - Target: P(max_x >= 3.0)
       - Feature Subset: Creation bundle metrics, deployer share, Gini concentration,
         and wash-trading ratios to eliminate immediate rugs.
       - Probability Calibration: Out-of-fold Platt scaling (Logistic Regression).
       - Per-Token Thresholding: Calibrated strictly on the last 5 training days
         to isolate ~2 to 5 trades/day without peeking across the test set.
         
    2. Stage 2 (Expected PnL Exit Selector):
       - Target Hurdles: LEVELS = [4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 100.0]
         (where 100 represents the full 1-hour hold policy under extreme momentum).
       - Feature Subset: Shrunk wallet reputation (m=5), post-trade reserve pool price
         velocity (d2V/dt2), and unique buyer inflow.
       - Utility Maximizer: Evaluates E[PnL(k)] under monotonic probability constraints.
       - Allocation: k* = argmax_k E[PnL(k)].
    """
    def __init__(self, feature_cols, levels=LEVELS, seeds=SEEDS, lgb_params=None, cost=COST):
        self.all_feature_cols = feature_cols
        self.levels = np.array(levels, dtype=float)
        self.seeds = seeds
        self.lgb_params = lgb_params or LGB_PARAMS
        self.cost = cost
        
        # Decouple feature subsets
        s1 = [c for c in feature_cols if re.search(STAGE1_PATTERN, c, re.I)]
        s2 = [c for c in feature_cols if re.search(STAGE2_PATTERN, c, re.I)]
        self.stage1_cols = s1 if len(s1) >= 10 else feature_cols
        self.stage2_cols = s2 if len(s2) >= 10 else feature_cols
        
        self.calibrator = None
        self.gatekeeper_models = []
        self.level_models = {lv: [] for lv in self.levels}
        self.tau_scalar = 0.20
        self.min_ev_margin = 0.0

    def fit_gatekeeper(self, train_df):
        X_tr = train_df[self.stage1_cols].astype(float).fillna(0).values
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
            X_old = trn_old[self.stage1_cols].astype(float).fillna(0).values
            y_old = (trn_old['max_x'].values >= 3.0).astype(int)
            X_cal = trn_cal[self.stage1_cols].astype(float).fillna(0).values
            y_cal = (trn_cal['max_x'].values >= 3.0).astype(int)
            
            p_cal_raw = np.mean([
                lgb.LGBMClassifier(random_state=s, **self.lgb_params)
                .fit(X_old, y_old)
                .predict_proba(X_cal)[:, 1]
                for s in self.seeds
            ], axis=0)
            
            self.calibrator = LogisticRegression()
            self.calibrator.fit(p_cal_raw.reshape(-1, 1), y_cal)
            self.p_cal_oof = self.calibrator.predict_proba(p_cal_raw.reshape(-1, 1))[:, 1]
            self.trn_cal_len = len(trn_cal)
            self.cal_days_len = len(cal_days)
        else:
            self.calibrator = None
            self.p_cal_oof = None
            self.trn_cal_len = 0
            self.cal_days_len = 0

    def predict_gatekeeper(self, test_df):
        X_te = test_df[self.stage1_cols].astype(float).fillna(0).values
        p_raw = np.mean([m.predict_proba(X_te)[:, 1] for m in self.gatekeeper_models], axis=0)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        return p_raw

    def fit_levels(self, train_df):
        X_tr = train_df[self.stage2_cols].astype(float).fillna(0).values
        mxs_tr = train_df['max_x'].values
        
        for lv in self.levels:
            y_lv = (mxs_tr >= lv).astype(int)
            if y_lv.sum() < 15:
                # Fallback for extreme levels with rare hits
                self.level_models[lv] = [float(y_lv.mean())]
                continue
                
            self.level_models[lv] = [
                lgb.LGBMClassifier(random_state=s, **self.lgb_params)
                .fit(X_tr, y_lv)
                for s in self.seeds
            ]

    def predict_levels(self, test_df):
        X_te = test_df[self.stage2_cols].astype(float).fillna(0).values
        probs = np.zeros((len(test_df), len(self.levels)), dtype=float)
        
        for j, lv in enumerate(self.levels):
            models = self.level_models[lv]
            if isinstance(models[0], (float, int)):
                probs[:, j] = float(models[0])
            else:
                probs[:, j] = np.mean([m.predict_proba(X_te)[:, 1] for m in models], axis=0)
                
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
        return probs_monotonic * win_pnl + (1.0 - probs_monotonic) * held_row

    def calibrate_scalar_threshold(self, train_df=None, target_daily_trades=2.5):
        """
        Computes the fixed scalar score cutoff strictly on the holdout training days
        to target a nominal precision depth of ~2 to 5 trades per day without test peeking.
        Uses out-of-fold calibrated probabilities to prevent in-sample threshold inflation.
        """
        if getattr(self, 'p_cal_oof', None) is not None and self.trn_cal_len > 0:
            avg_tokens_per_day = max(self.trn_cal_len / float(self.cal_days_len), 10.0)
            q = min(max(target_daily_trades / avg_tokens_per_day, 0.001), 0.05)
            self.tau_scalar = float(np.quantile(self.p_cal_oof, 1.0 - q))
        elif train_df is not None:
            d_trn = sorted(train_df.day.unique())
            cal_days = d_trn[-HOLD_DAYS:]
            trn_cal = train_df[train_df.day.isin(cal_days)].reset_index(drop=True)
            p_cal = self.predict_gatekeeper(trn_cal)
            avg_tokens_per_day = max(len(trn_cal) / float(len(cal_days)), 10.0)
            q = min(max(target_daily_trades / avg_tokens_per_day, 0.001), 0.05)
            self.tau_scalar = float(np.quantile(p_cal, 1.0 - q))
        else:
            self.tau_scalar = 0.20
        return self.tau_scalar

    def decide_per_token(self, p_gatekeeper, ev_matrix, tau_scalar=None, min_ev_margin=None):
        """
        Per-token causal execution rule (ZERO test ranking or peeking):
        For each incoming token i:
          Stage 1 (Gatekeeper): If p_gatekeeper[i] >= tau_scalar, allocate long entry.
          Stage 2 (Exit Selector): tp[i] = argmax_k E[PnL(k)]
          Else:
              tp[i] = 0.0 (Costless Skip)
        """
        tau = self.tau_scalar if tau_scalar is None else tau_scalar
        n = len(p_gatekeeper)
        final_tp = np.zeros(n, dtype=float)
        
        # Stage 2: Select optimal take profit multiple k* maximizing expected utility
        best_lv_idx = np.argmax(ev_matrix, axis=1)
        best_ev = ev_matrix[np.arange(n), best_lv_idx]
        best_tp = self.levels[best_lv_idx]
        
        # Stage 1: Causal per-token gatekeeper decision
        if min_ev_margin is not None:
            is_buy = (p_gatekeeper >= tau) & (best_ev >= min_ev_margin)
        else:
            is_buy = (p_gatekeeper >= tau)
            
        final_tp[is_buy] = best_tp[is_buy]
        return final_tp, best_ev

    def decide(self, p_gatekeeper, ev_matrix, top_n_daily=2, min_prob_hurdle=None, min_ev_margin=0.0):
        """Wrapper maintaining interface compatibility."""
        tau = self.tau_scalar if min_prob_hurdle is None else min_prob_hurdle
        return self.decide_per_token(p_gatekeeper, ev_matrix, tau_scalar=tau, min_ev_margin=min_ev_margin)


# Backwards compatibility aliases
class MultiLevelExpectedValueEngine(TwoStageDecisionEngine):
    def fit_predict_levels(self, train_df, test_df):
        self.fit_levels(train_df)
        return self.predict_levels(test_df)
        
    def compute_held_payoffs(self, train_df):
        return self.compute_miss_payoffs(train_df)

TwoStageCascadeEnsemble = TwoStageDecisionEngine

def calibrate_decision_threshold(model, train_df, q_depth=1.0):
    return 1.0 - (q_depth / 100.0)

def make_decisions(scores, threshold, target_k=TARGET_K):
    return np.where(scores >= threshold, target_k, 0.0)

def optimize_ev_margin(model, train_df, margin_grid=(0.0, 0.10, 0.20), hold_days=HOLD_DAYS):
    return 0.0
