"""Unified Entry Ensemble (3x LGBM + 3x CatBoost), Terminal Floor Regressor, and Continuous Monotonic TP Grid Engine."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import LogisticRegression
from src.config import TARGET_K, COST, LGB_PARAMS, WIN, EMBARGO, HOLD_DAYS

TP_GRID = [3.8, 4.5, 5.0, 6.0, 7.5, 10.0]
SEEDS_3 = [42, 101, 777]

CB_PARAMS = {
    'iterations': 200,
    'learning_rate': 0.06,
    'depth': 5,
    'thread_count': -1,
    'verbose': 0
}

LGB_PARAMS_OPT = dict(
    n_estimators=300, learning_rate=0.04, num_leaves=31, min_child_samples=40,
    subsample=0.80, subsample_freq=1, colsample_bytree=0.70, verbose=-1, n_jobs=-1
)

class UnifiedSurvivalEngine:
    """
    Advanced Quantitative Decision Engine featuring:
    1. 6-Model Seed Ensemble for Primary Entry:
       - 3x LightGBM + 3x CatBoost on P(max_x >= 5.0) using distinct seeds (42, 101, 777).
       - Platt scaling out-of-fold calibration on recent training days to eliminate seed variance.
    2. Expected Terminal Floor Regressor (hat_hold):
       - Predicts expected terminal floor hat_hold = max(final_x - 1.3, -1.0) on tokens failing to reach target.
    3. Monotonic Survival Curve & Continuous TP Grid:
       - Evaluates P(max_x >= k) across k in {3.8, 4.5, 5.0, 6.0, 7.5, 10.0} enforcing:
         P(max_x >= k1) >= P(max_x >= k2) for k1 < k2, anchored on calibrated 5x ensemble.
    4. Exact Token-Level Expected Utility Maximizer:
       - E[PnL(k)] = P(max_x >= k) * (k - 1.3) + (1 - P(max_x >= k)) * hat_hold
       - Assigns tp* = argmax_k E[PnL(k)], defaulting to 5.0 unless another level beats 5.0 by >= 0.15 SOL.
    5. Rug-Risk Veto:
       - Vetoes entry (tp = 0.0) if hat_hold < -0.92 and dev_sell_ratio > 0.5.
    """
    def __init__(self, feature_cols, seeds=SEEDS_3, lgb_params=None, cb_params=None, cost=COST, tp_grid=None):
        self.feature_cols = list(feature_cols)
        self.seeds = seeds if len(seeds) > 0 else [42]
        self.lgb_params = lgb_params or LGB_PARAMS_OPT
        self.cb_params = cb_params or CB_PARAMS
        self.cost = cost
        self.tp_grid = tp_grid or TP_GRID
        self.idx_5 = self.tp_grid.index(5.0) if 5.0 in self.tp_grid else 2
        
        # Primary 5x ensemble models (3 LGBM + 3 CatBoost)
        self.lgb_entry_models = []
        self.cat_entry_models = []
        
        # Survival models for other levels
        self.level_models = {k: [] for k in self.tp_grid if k != 5.0}
        self.level_calibrators = {}
        
        # Terminal floor regressor
        self.hold_regressor = None
        self.calibrator_5x = None
        self.p_cal_oof = None
        self.best_ev_cal_oof = None
        self.trn_cal_len = 0
        self.cal_days_len = 0
        self.tau_scalar = 0.0

    def fit_entry(self, train_df):
        """Fits the primary 6-model ensemble, terminal floor regressor, and survival grid."""
        X_tr = train_df[self.feature_cols].astype(float).fillna(0).values
        mxs_tr = train_df['max_x'].values
        fxs_tr = train_df['final_x'].values if 'final_x' in train_df.columns else train_df['finalx'].values
        y_tr_5x = (mxs_tr >= 5.0).astype(int)
        
        # Terminal floor target: max(final_x - 1.3, -1.0)
        y_hold_tr = np.clip(fxs_tr - 1.0 - self.cost, -1.0, 0.0)
        
        # Out-of-fold split for calibration
        d_trn = sorted(train_df.day.unique())
        cal_days = d_trn[-HOLD_DAYS:]
        trn_cal = train_df[train_df.day.isin(cal_days)].reset_index(drop=True)
        trn_old = train_df[(train_df.day < cal_days[0]) & (train_df.mt <= trn_cal.mt.min() - WIN - EMBARGO)].reset_index(drop=True)
        
        if len(trn_old) >= 500 and len(trn_cal) >= 10:
            X_old = trn_old[self.feature_cols].astype(float).fillna(0).values
            mxs_old = trn_old['max_x'].values
            fxs_old = trn_old['final_x'].values if 'final_x' in trn_old.columns else trn_old['finalx'].values
            y_old_5x = (mxs_old >= 5.0).astype(int)
            y_old_hold = np.clip(fxs_old - 1.0 - self.cost, -1.0, 0.0)
            
            X_cal = trn_cal[self.feature_cols].astype(float).fillna(0).values
            mxs_cal = trn_cal['max_x'].values
            y_cal_5x = (mxs_cal >= 5.0).astype(int)
            
            # OOF 5x models
            oof_5_preds = []
            for s in self.seeds:
                m_lgb = lgb.LGBMClassifier(random_state=s, **self.lgb_params).fit(X_old, y_old_5x)
                m_cb = cb.CatBoostClassifier(random_seed=s, **self.cb_params).fit(X_old, y_old_5x)
                oof_5_preds.append(m_lgb.predict_proba(X_cal)[:, 1])
                oof_5_preds.append(m_cb.predict_proba(X_cal)[:, 1])
                
            p_oof_5x_raw = np.mean(oof_5_preds, axis=0)
            self.calibrator_5x = LogisticRegression().fit(p_oof_5x_raw.reshape(-1, 1), y_cal_5x)
            p_cal_oof_5x = self.calibrator_5x.predict_proba(p_oof_5x_raw.reshape(-1, 1))[:, 1]
            self.p_cal_oof = p_cal_oof_5x
            
            # OOF hold regressor
            mask_old_miss = (mxs_old < 5.0)
            oof_hold_reg = lgb.LGBMRegressor(
                n_estimators=100, learning_rate=0.05, max_depth=4, num_leaves=15,
                subsample=0.8, colsample_bytree=0.7, random_state=42, verbose=-1, n_jobs=-1
            ).fit(X_old[mask_old_miss], y_old_hold[mask_old_miss])
            hat_hold_cal = np.clip(oof_hold_reg.predict(X_cal), -1.0, 0.0)
            
            # Compute OOF Expected Utility
            ev_cal_5x = p_cal_oof_5x * (5.0 - 1.0 - self.cost) + (1.0 - p_cal_oof_5x) * hat_hold_cal
            self.best_ev_cal_oof = ev_cal_5x
            self.trn_cal_len = len(trn_cal)
            self.cal_days_len = len(cal_days)
        else:
            self.calibrator_5x = None
            self.p_cal_oof = None
            self.best_ev_cal_oof = None
            self.trn_cal_len = 0
            self.cal_days_len = 0
            
        # Fit Full Production 6-Model 5x Ensemble
        self.lgb_entry_models = [
            lgb.LGBMClassifier(random_state=s, **self.lgb_params).fit(X_tr, y_tr_5x)
            for s in self.seeds
        ]
        self.cat_entry_models = [
            cb.CatBoostClassifier(random_seed=s, **self.cb_params).fit(X_tr, y_tr_5x)
            for s in self.seeds
        ]
        
        # Fit Terminal Floor Regressor on misses
        mask_miss = (mxs_tr < 5.0)
        self.hold_regressor = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=5, num_leaves=20,
            subsample=0.8, colsample_bytree=0.7, random_state=42, verbose=-1, n_jobs=-1
        ).fit(X_tr[mask_miss], y_hold_tr[mask_miss])
        
        # Fit Survival Models for other levels in TP_GRID
        for k in self.tp_grid:
            if k == 5.0:
                continue
            y_k = (mxs_tr >= k).astype(int)
            if y_k.sum() >= 15:
                self.level_models[k] = [
                    lgb.LGBMClassifier(random_state=s, **self.lgb_params).fit(X_tr, y_k)
                    for s in self.seeds[:2]
                ]
            else:
                self.level_models[k] = [float(y_k.mean())]

    def fit_gatekeeper(self, train_df):
        return self.fit_entry(train_df)

    def fit_levels(self, train_df):
        pass

    def predict_entry(self, test_df):
        """Predicts calibrated P(max_x >= 5.0) using the 6-model ensemble."""
        X_te = test_df[self.feature_cols].astype(float).fillna(0).values
        preds = []
        for m in self.lgb_entry_models:
            preds.append(m.predict_proba(X_te)[:, 1])
        for m in self.cat_entry_models:
            preds.append(m.predict_proba(X_te)[:, 1])
        p_raw = np.mean(preds, axis=0)
        if self.calibrator_5x is not None:
            return self.calibrator_5x.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        return p_raw

    def predict_gatekeeper(self, test_df):
        return self.predict_entry(test_df)

    def predict_survival(self, test_df):
        """
        Generates calibrated, strictly monotonic survival probabilities across TP_GRID:
        P(max_x >= k1) >= P(max_x >= k2) for k1 < k2, anchored on calibrated P(max_x >= 5.0).
        """
        X_te = test_df[self.feature_cols].astype(float).fillna(0).values
        n = len(test_df)
        p_5x = self.predict_entry(test_df)
        
        raw_probs = np.zeros((n, len(self.tp_grid)), dtype=float)
        raw_probs[:, self.idx_5] = p_5x
        
        for j, k in enumerate(self.tp_grid):
            if k == 5.0:
                continue
            models = self.level_models[k]
            if isinstance(models[0], (float, int)):
                raw_probs[:, j] = float(models[0])
            else:
                raw_probs[:, j] = np.mean([m.predict_proba(X_te)[:, 1] for m in models], axis=0)
                
        # Enforce strict isotonic/monotonic constraints anchored on 5x:
        # For k < 5.0 (left of anchor): P(>= 4.5) >= P(>= 5.0); P(>= 3.8) >= P(>= 4.5)
        # For k > 5.0 (right of anchor): P(>= 6.0) <= P(>= 5.0); P(>= 7.5) <= P(>= 6.0); P(>= 10.0) <= P(>= 7.5)
        probs_mono = raw_probs.copy()
        
        # Left sweep from anchor to index 0
        for j in range(self.idx_5 - 1, -1, -1):
            probs_mono[:, j] = np.maximum(probs_mono[:, j], probs_mono[:, j + 1])
            
        # Right sweep from anchor to end
        for j in range(self.idx_5 + 1, len(self.tp_grid)):
            probs_mono[:, j] = np.minimum(probs_mono[:, j], probs_mono[:, j - 1])
            
        return probs_mono

    def predict_hold(self, test_df):
        """Predicts expected terminal floor hat_hold for failed tokens."""
        X_te = test_df[self.feature_cols].astype(float).fillna(0).values
        if self.hold_regressor is not None:
            return np.clip(self.hold_regressor.predict(X_te), -1.0, 0.0)
        return np.full(len(test_df), -0.95)

    def calculate_expected_values(self, test_df=None, probs=None, hat_hold=None):
        """
        Computes exact token-level expected utility matrix:
        E[PnL(k)] = P(max_x >= k) * (k - 1.3) + (1 - P(max_x >= k)) * hat_hold
        """
        if probs is None and test_df is not None:
            probs = self.predict_survival(test_df)
        if hat_hold is None and test_df is not None:
            hat_hold = self.predict_hold(test_df)
            
        n = len(probs)
        ev_matrix = np.zeros((n, len(self.tp_grid)), dtype=float)
        
        for j, k in enumerate(self.tp_grid):
            win_pnl = k - 1.0 - self.cost
            ev_matrix[:, j] = probs[:, j] * win_pnl + (1.0 - probs[:, j]) * hat_hold
            
        return ev_matrix

    def calibrate_scalar_threshold(self, train_df=None, target_daily_trades=2.5):
        """
        Calibrates the fixed scalar score cutoff strictly on the holdout training days (OOF)
        targeting ~2 to 5 trades per day without test peeking.
        """
        if getattr(self, 'p_cal_oof', None) is not None and self.trn_cal_len > 0:
            avg_tokens_per_day = max(self.trn_cal_len / float(self.cal_days_len), 10.0)
            q = min(max(target_daily_trades / avg_tokens_per_day, 0.001), 0.05)
            self.tau_scalar = float(np.quantile(self.p_cal_oof, 1.0 - q))
        elif train_df is not None:
            d_trn = sorted(train_df.day.unique())
            cal_days = d_trn[-HOLD_DAYS:]
            trn_cal = train_df[train_df.day.isin(cal_days)].reset_index(drop=True)
            p_cal = self.predict_entry(trn_cal)
            avg_tokens_per_day = max(len(trn_cal) / float(len(cal_days)), 10.0)
            q = min(max(target_daily_trades / avg_tokens_per_day, 0.001), 0.05)
            self.tau_scalar = float(np.quantile(p_cal, 1.0 - q))
        else:
            self.tau_scalar = 0.05
        return self.tau_scalar

    def decide_per_token(self, test_df, ev_matrix=None, p_entry=None, tau_scalar=None, safety_margin=0.15, mode='continuous'):
        """
        Causal per-token execution rule:
        1. Checks entry clearance: p_entry >= tau_scalar
        2. Enforces rug_risk_veto: vetoes entry if hat_hold < -0.92 and dev_sell_ratio > 0.5
        3. If mode == 'fixed5': assigns tp = 5.0
        4. If mode == 'continuous': assigns tp* = argmax_k E[PnL(k)], defaulting to 5.0
           unless another level in {3.8, 4.5, 6.0, 7.5, 10.0} beats E[PnL(5.0)] by >= 0.15 SOL.
        5. Else tp = 0.0 (Costless Skip).
        """
        if p_entry is None:
            p_entry = self.predict_entry(test_df)
        if ev_matrix is None:
            ev_matrix = self.calculate_expected_values(test_df)
            
        tau = self.tau_scalar if tau_scalar is None else tau_scalar
        n = len(test_df)
        final_tp = np.zeros(n, dtype=float)
        
        hat_hold = self.predict_hold(test_df)
        dev_sell = test_df['dev_sell_ratio'].values if 'dev_sell_ratio' in test_df.columns else np.zeros(n)
        
        # Candidate clearance via primary gatekeeper
        is_candidate = (p_entry >= tau)
        
        # Rug-risk veto
        is_vetoed = (hat_hold < -0.92) & (dev_sell > 0.5)
        
        ev_5 = ev_matrix[:, self.idx_5]
        
        for i in range(n):
            if is_candidate[i] and not is_vetoed[i]:
                if mode == 'fixed5':
                    final_tp[i] = 5.0
                else:
                    # Continuous monotonic utility grid
                    chosen_k = 5.0
                    # Find candidate level with maximum expected utility outside 5.0
                    cand_evs = ev_matrix[i].copy()
                    cand_evs[self.idx_5] = -999.0
                    best_other_idx = int(np.argmax(cand_evs))
                    
                    if cand_evs[best_other_idx] >= ev_5[i] + safety_margin:
                        chosen_k = self.tp_grid[best_other_idx]
                    final_tp[i] = chosen_k
                    
        return final_tp, ev_5

    def predict_levels(self, test_df):
        return self.predict_survival(test_df)

    def compute_miss_payoffs(self, train_df):
        mxs = train_df['max_x'].values
        fxs = train_df['final_x'].values if 'final_x' in train_df.columns else train_df['finalx'].values
        held = np.maximum(fxs - 1.0 - self.cost, -1.0)
        return {'held_mean': float(held[mxs < 5.0].mean()) if (mxs < 5.0).any() else -0.95}

    def decide(self, test_df, ev_matrix=None, top_n_daily=2, min_prob_hurdle=None, min_ev_margin=0.0):
        tau = self.tau_scalar if min_prob_hurdle is None else min_prob_hurdle
        return self.decide_per_token(test_df, ev_matrix=ev_matrix, tau_scalar=tau, mode='continuous')


# Backwards compatibility aliases
UnifiedTwoTierEngine = UnifiedSurvivalEngine
TwoStageDecisionEngine = UnifiedSurvivalEngine
MultiLevelExpectedValueEngine = UnifiedSurvivalEngine
TwoStageCascadeEnsemble = UnifiedSurvivalEngine
