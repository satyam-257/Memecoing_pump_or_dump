"""Unified LightGBM + CatBoost Entry Classifier & Constrained 2-Tier Exit Utility Engine."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import LogisticRegression
from src.config import TARGET_K, COST, SEEDS, LGB_PARAMS, WIN, EMBARGO, HOLD_DAYS

CB_PARAMS = {
    'iterations': 250,
    'learning_rate': 0.05,
    'depth': 6,
    'thread_count': -1,
    'verbose': 0
}

class UnifiedTwoTierEngine:
    """
    Unified Full-Feature Entry Gatekeeper & Constrained 2-Tier Exit Policy:
    
    1. Unified Full-Feature Entry Classifier (target: max_x >= 5.0):
       - Uses ALL features simultaneously (Creation Bundles + Tape Velocity + Shrunk Wallet Reputation).
       - Never starves the entry gatekeeper of wallet reputation.
       - Blends LightGBM + CatBoost ensemble probabilities for P(max_x >= 5.0).
       - Calibrated via Out-of-Fold Logistic Regression on the last 5 training days.
       - Per-Token Thresholding: Evaluates against fixed scalar cutoff tau* (target ~2 to 5 trades/day)
         strictly without test peeking.
         
    2. Constrained 2-Tier Exit Policy (locks in baseline 5x edge, upgrades only on high conviction):
       - Default Exit: tp = 5.0 (+3.70 SOL net payout).
       - High-Conviction Runner Upgrade: Independently trains calibrated model for P(max_x >= 8.0).
         Upgrades to tp = 8.0 (+6.70 SOL net payout) iff E[PnL(8.0)] > E[PnL(5.0)] + 0.35 SOL safety margin.
       - Skips non-qualifying tokens at zero cost (tp = 0.0).
    """
    def __init__(self, feature_cols, seeds=SEEDS[:2], lgb_params=None, cb_params=None, cost=COST):
        self.feature_cols = list(feature_cols)
        self.seeds = seeds if len(seeds) > 0 else [42]
        self.lgb_params = lgb_params or LGB_PARAMS
        self.cb_params = cb_params or CB_PARAMS
        self.cost = cost
        
        # Models
        self.lgb_entry_models = []
        self.cat_entry_models = []
        self.lgb_runner_models = []
        self.cat_runner_models = []
        
        # Calibrators
        self.calibrator_5x = None
        self.calibrator_8x = None
        self.p_cal_oof = None
        self.p_cal_8x_oof = None
        self.trn_cal_len = 0
        self.cal_days_len = 0
        self.tau_scalar = 0.20
        self.miss_payoffs = {'held_5': -0.95, 'held_8': -0.95}

    def fit_entry(self, train_df):
        """Fits the unified entry classifier ensemble (LGBM + CatBoost) on P(max_x >= 5.0)."""
        X_tr = train_df[self.feature_cols].astype(float).fillna(0).values
        y_tr_5x = (train_df['max_x'].values >= 5.0).astype(int)
        y_tr_8x = (train_df['max_x'].values >= 8.0).astype(int)
        
        # Out-of-fold probability calibration using recent holdout days
        d_trn = sorted(train_df.day.unique())
        cal_days = d_trn[-HOLD_DAYS:]
        trn_cal = train_df[train_df.day.isin(cal_days)].reset_index(drop=True)
        trn_old = train_df[(train_df.day < cal_days[0]) & (train_df.mt <= trn_cal.mt.min() - WIN - EMBARGO)].reset_index(drop=True)
        
        if len(trn_old) >= 500 and len(trn_cal) >= 10:
            X_old = trn_old[self.feature_cols].astype(float).fillna(0).values
            y_old_5x = (trn_old['max_x'].values >= 5.0).astype(int)
            y_old_8x = (trn_old['max_x'].values >= 8.0).astype(int)
            X_cal = trn_cal[self.feature_cols].astype(float).fillna(0).values
            y_cal_5x = (trn_cal['max_x'].values >= 5.0).astype(int)
            y_cal_8x = (trn_cal['max_x'].values >= 8.0).astype(int)
            
            oof_5_preds = []
            oof_8_preds = []
            for s in self.seeds:
                # OOF 5x models
                m_lgb5 = lgb.LGBMClassifier(random_state=s, **self.lgb_params).fit(X_old, y_old_5x)
                m_cb5 = cb.CatBoostClassifier(random_seed=s, **self.cb_params).fit(X_old, y_old_5x)
                oof_5_preds.append(0.5 * m_lgb5.predict_proba(X_cal)[:, 1] + 0.5 * m_cb5.predict_proba(X_cal)[:, 1])
                
                # OOF 8x models
                if y_old_8x.sum() >= 10:
                    m_lgb8 = lgb.LGBMClassifier(random_state=s, **self.lgb_params).fit(X_old, y_old_8x)
                    m_cb8 = cb.CatBoostClassifier(random_seed=s, **self.cb_params).fit(X_old, y_old_8x)
                    oof_8_preds.append(0.5 * m_lgb8.predict_proba(X_cal)[:, 1] + 0.5 * m_cb8.predict_proba(X_cal)[:, 1])
                else:
                    oof_8_preds.append(np.full(len(X_cal), float(y_old_8x.mean())))
                    
            p_oof_5x_raw = np.mean(oof_5_preds, axis=0)
            p_oof_8x_raw = np.mean(oof_8_preds, axis=0)
            
            self.calibrator_5x = LogisticRegression()
            self.calibrator_5x.fit(p_oof_5x_raw.reshape(-1, 1), y_cal_5x)
            self.p_cal_oof = self.calibrator_5x.predict_proba(p_oof_5x_raw.reshape(-1, 1))[:, 1]
            
            if y_cal_8x.sum() >= 5:
                self.calibrator_8x = LogisticRegression()
                self.calibrator_8x.fit(p_oof_8x_raw.reshape(-1, 1), y_cal_8x)
                self.p_cal_8x_oof = self.calibrator_8x.predict_proba(p_oof_8x_raw.reshape(-1, 1))[:, 1]
            else:
                self.calibrator_8x = None
                self.p_cal_8x_oof = p_oof_8x_raw
                
            self.trn_cal_len = len(trn_cal)
            self.cal_days_len = len(cal_days)
        else:
            self.calibrator_5x = None
            self.calibrator_8x = None
            self.p_cal_oof = None
            self.trn_cal_len = 0
            self.cal_days_len = 0
            
        # Fit full production models on entire train_df
        self.lgb_entry_models = [
            lgb.LGBMClassifier(random_state=s, **self.lgb_params).fit(X_tr, y_tr_5x)
            for s in self.seeds
        ]
        self.cat_entry_models = [
            cb.CatBoostClassifier(random_seed=s, **self.cb_params).fit(X_tr, y_tr_5x)
            for s in self.seeds
        ]
        
        # Fit full runner models on entire train_df
        if y_tr_8x.sum() >= 15:
            self.lgb_runner_models = [
                lgb.LGBMClassifier(random_state=s, **self.lgb_params).fit(X_tr, y_tr_8x)
                for s in self.seeds
            ]
            self.cat_runner_models = [
                cb.CatBoostClassifier(random_seed=s, **self.cb_params).fit(X_tr, y_tr_8x)
                for s in self.seeds
            ]
        else:
            self.lgb_runner_models = []
            self.cat_runner_models = []
            
        # Compute empirical miss payoffs
        self.compute_miss_payoffs(train_df)

    def fit_gatekeeper(self, train_df):
        """Interface compatibility alias."""
        return self.fit_entry(train_df)

    def fit_levels(self, train_df):
        """Interface compatibility alias."""
        pass

    def predict_entry(self, test_df):
        """Predicts calibrated P(max_x >= 5.0) probability blend."""
        X_te = test_df[self.feature_cols].astype(float).fillna(0).values
        p_lgb = np.mean([m.predict_proba(X_te)[:, 1] for m in self.lgb_entry_models], axis=0)
        p_cat = np.mean([m.predict_proba(X_te)[:, 1] for m in self.cat_entry_models], axis=0)
        p_raw = 0.5 * p_lgb + 0.5 * p_cat
        
        if self.calibrator_5x is not None:
            return self.calibrator_5x.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        return p_raw

    def predict_gatekeeper(self, test_df):
        """Interface compatibility alias."""
        return self.predict_entry(test_df)

    def predict_runner(self, test_df):
        """Predicts calibrated P(max_x >= 8.0) probability blend."""
        X_te = test_df[self.feature_cols].astype(float).fillna(0).values
        if len(self.lgb_runner_models) > 0 and len(self.cat_runner_models) > 0:
            p_lgb = np.mean([m.predict_proba(X_te)[:, 1] for m in self.lgb_runner_models], axis=0)
            p_cat = np.mean([m.predict_proba(X_te)[:, 1] for m in self.cat_runner_models], axis=0)
            p_raw = 0.5 * p_lgb + 0.5 * p_cat
        else:
            p_raw = np.full(len(test_df), 0.02)
            
        if self.calibrator_8x is not None:
            return self.calibrator_8x.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        return p_raw

    def predict_levels(self, test_df):
        """Interface compatibility returning [p_5x, p_8x]."""
        p5 = self.predict_entry(test_df)
        p8 = self.predict_runner(test_df)
        p8_mono = np.minimum(p8, p5)
        return np.column_stack([p5, p8_mono])

    def compute_miss_payoffs(self, train_df):
        """Computes empirical average miss return for 5x and 8x thresholds."""
        mxs = train_df['max_x'].values
        fxs = train_df['final_x'].values if 'final_x' in train_df.columns else train_df['finalx'].values
        held = np.maximum(fxs - 1.0 - self.cost, -1.0)
        
        miss_5 = mxs < 5.0
        miss_8 = mxs < 8.0
        held_5 = float(held[miss_5].mean()) if miss_5.any() else -0.95
        held_8 = float(held[miss_8].mean()) if miss_8.any() else -0.95
        self.miss_payoffs = {'held_5': held_5, 'held_8': held_8}
        return self.miss_payoffs

    def calculate_expected_values(self, test_df=None, p_levels=None, miss_payoffs=None):
        """Computes expected PnL for 5x vs 8x exits."""
        if p_levels is None and test_df is not None:
            p_levels = self.predict_levels(test_df)
            
        p5 = p_levels[:, 0]
        p8 = p_levels[:, 1]
        
        mp = miss_payoffs or self.miss_payoffs
        h5 = mp.get('held_5', -0.95) if isinstance(mp, dict) else mp[0]
        h8 = mp.get('held_8', -0.95) if isinstance(mp, dict) else mp[1]
        
        # Net win payoffs: 5.0 - 1.0 - 0.3 = +3.70; 8.0 - 1.0 - 0.3 = +6.70
        ev_5 = p5 * 3.70 + (1.0 - p5) * h5
        ev_8 = p8 * 6.70 + (1.0 - p8) * h8
        return np.column_stack([ev_5, ev_8])

    def calibrate_scalar_threshold(self, train_df=None, target_daily_trades=2.5):
        """
        Computes the fixed scalar cutoff strictly on the holdout training days (OOF)
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
            self.tau_scalar = 0.20
        return self.tau_scalar

    def decide_per_token(self, p_entry, ev_matrix, tau_scalar=None, safety_margin=0.35, mode='2tier'):
        """
        Per-token causal execution rule (zero test ranking or peeking):
        For each token i:
          If p_entry[i] >= tau_scalar:
             If mode == 'fixed5':
                 tp[i] = 5.0
             Else (mode == '2tier'):
                 Upgrade to tp[i] = 8.0 iff E[PnL(8.0)] > E[PnL(5.0)] + 0.35 SOL safety margin
                 Else tp[i] = 5.0
          Else:
             tp[i] = 0.0 (Costless Skip)
        """
        tau = self.tau_scalar if tau_scalar is None else tau_scalar
        n = len(p_entry)
        final_tp = np.zeros(n, dtype=float)
        
        ev_5 = ev_matrix[:, 0]
        ev_8 = ev_matrix[:, 1]
        
        is_buy = (p_entry >= tau)
        
        if mode == 'fixed5':
            final_tp[is_buy] = 5.0
        else:
            # Constrained 2-tier exit upgrade
            is_upgrade = is_buy & (ev_8 > (ev_5 + safety_margin))
            final_tp[is_buy] = 5.0
            final_tp[is_upgrade] = 8.0
            
        return final_tp, ev_5

    def decide(self, p_entry, ev_matrix, top_n_daily=2, min_prob_hurdle=None, min_ev_margin=0.0):
        """Wrapper maintaining interface compatibility."""
        tau = self.tau_scalar if min_prob_hurdle is None else min_prob_hurdle
        return self.decide_per_token(p_entry, ev_matrix, tau_scalar=tau, mode='2tier')


# Interface compatibility aliases
TwoStageDecisionEngine = UnifiedTwoTierEngine
MultiLevelExpectedValueEngine = UnifiedTwoTierEngine
TwoStageCascadeEnsemble = UnifiedTwoTierEngine
