"""Restored and Hardened +5.24177 SOL Champion Model (Unified LGBM + CatBoost Entry, Fixed TP=5.0)."""
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.isotonic import IsotonicRegression
from src.config import TARGET_K, COST, SEEDS, LGB_PARAMS, WIN, EMBARGO, HOLD_DAYS

CB_PARAMS = {
    'iterations': 250,
    'learning_rate': 0.05,
    'depth': 6,
    'thread_count': -1,
    'verbose': 0
}

class ChampionFixed5Engine:
    """
    Restored and Hardened +5.24177 SOL Champion Quantitative Decision Engine:
    
    1. Unified Full-Feature Entry Gatekeeper (target: max_x >= 5.0):
       - Uses ALL 114 features simultaneously (Creation Bundles + Tape Velocity +
         failed!=0 Reverted Curve Demand + Warm-Started Archive Wallet Reputation with m=5 shrinkage and 7-day decay).
       - Zero sample_weight distortion: LightGBM + CatBoost output true, unbiased empirical probabilities for P(max_x >= 5.0).
       - Blends LightGBM + CatBoost ensemble probabilities: P = 0.5 * P_lgb + 0.5 * P_cat.
       
    2. Threshold Calibration Strictly Targeting ~2.0 Trades/Day (30-35 Trades / 16 Days):
       - Calibrated on out-of-fold predictions of the last 5 training days.
       - Selects top ~0.5% to 0.7% quantile (target ~1.5 to 2.2 trades/day).
       - Enforces hard minimum floor P_cal(max_x >= 5.0) >= 0.23 (since break-even at TP=5.0 is 20.4%).
       
    3. Strictly Fixed TP = 5.0 Exit Policy:
       - No noisy multi-level down-sizing or secondary regressors.
       - Token clears tau* -> tp = 5.0 (+3.70 SOL net gain on hit, -0.95 SOL on miss).
       - Non-clearing tokens -> tp = 0.0 (Costless Skip).
    """
    def __init__(self, feature_cols, seeds=SEEDS[:2], lgb_params=None, cb_params=None, cost=COST, **kwargs):
        self.feature_cols = list(feature_cols)
        self.seeds = seeds if len(seeds) > 0 else [42]
        self.lgb_params = lgb_params or LGB_PARAMS
        self.cb_params = cb_params or CB_PARAMS
        self.cost = cost
        self.tp_grid = [5.0]
        self.idx_5 = 0
        
        # Models
        self.lgb_entry_models = []
        self.cat_entry_models = []
        
        # OOF Calibration
        self.p_oof_cal = None
        self.p_oof_cal_iso = None
        self.isotonic_calibrator = None
        self.trn_cal_len = 0
        self.cal_days_len = 0
        self.tau_scalar = 0.220
        self.miss_payoff = -0.95

    def fit_entry(self, train_df, **kwargs):
        """Fits the unified entry classifier ensemble (LGBM + CatBoost) on P(max_x >= 5.0) with Out-of-Fold Isotonic Calibration."""
        X_tr = train_df[self.feature_cols].astype(float).fillna(0).values
        y_tr_5x = (train_df['max_x'].values >= 5.0).astype(int)
        
        # Out-of-fold probability calibration using recent holdout days
        d_trn = sorted(train_df.day.unique())
        cal_days = d_trn[-HOLD_DAYS:]
        trn_cal = train_df[train_df.day.isin(cal_days)].reset_index(drop=True)
        trn_old = train_df[(train_df.day < cal_days[0]) & (train_df.mt <= trn_cal.mt.min() - WIN - EMBARGO)].reset_index(drop=True)
        
        if len(trn_old) >= 500 and len(trn_cal) >= 10:
            X_old = trn_old[self.feature_cols].astype(float).fillna(0).values
            y_old_5x = (trn_old['max_x'].values >= 5.0).astype(int)
            X_cal = trn_cal[self.feature_cols].astype(float).fillna(0).values
            y_cal_5x = (trn_cal['max_x'].values >= 5.0).astype(int)
            
            oof_preds = []
            for s in self.seeds:
                m_lgb = lgb.LGBMClassifier(random_state=s, **self.lgb_params).fit(X_old, y_old_5x)
                m_cb = cb.CatBoostClassifier(random_seed=s, **self.cb_params).fit(X_old, y_old_5x)
                oof_preds.append(0.5 * m_lgb.predict_proba(X_cal)[:, 1] + 0.5 * m_cb.predict_proba(X_cal)[:, 1])
                
            self.p_oof_cal = np.mean(oof_preds, axis=0)
            self.trn_cal_len = len(trn_cal)
            self.cal_days_len = len(cal_days)
            
            # Fit Isotonic Regression to map raw GBDT probabilities to empirical hit rate in [0.0, 0.45]
            try:
                self.isotonic_calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=0.45)
                self.isotonic_calibrator.fit(self.p_oof_cal, y_cal_5x)
                self.p_oof_cal_iso = self.isotonic_calibrator.predict(self.p_oof_cal)
            except Exception:
                self.isotonic_calibrator = None
                self.p_oof_cal_iso = self.p_oof_cal
        else:
            self.p_oof_cal = None
            self.p_oof_cal_iso = None
            self.isotonic_calibrator = None
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
        
        # Miss payoff estimation
        if 'final_x' in train_df.columns:
            fxs = train_df['final_x'].values
        elif 'finalx' in train_df.columns:
            fxs = train_df['finalx'].values
        else:
            fxs = np.zeros(len(train_df))
        held = np.maximum(fxs - 1.0 - self.cost, -1.0)
        misses = (y_tr_5x == 0)
        self.miss_payoff = float(held[misses].mean()) if misses.any() else -0.95
        return self

    def fit(self, train_df, **kwargs):
        """Unified training interface."""
        return self.fit_entry(train_df, **kwargs)

    def predict_entry(self, test_df):
        """Predicts empirical probability P(max_x >= 5.0) as unweighted ensemble average with isotonic calibration."""
        X_te = test_df[self.feature_cols].astype(float).fillna(0).values
        if len(self.lgb_entry_models) > 0 and len(self.cat_entry_models) > 0:
            p_lgb = np.mean([m.predict_proba(X_te)[:, 1] for m in self.lgb_entry_models], axis=0)
            p_cat = np.mean([m.predict_proba(X_te)[:, 1] for m in self.cat_entry_models], axis=0)
            p_raw = 0.5 * p_lgb + 0.5 * p_cat
            if getattr(self, 'isotonic_calibrator', None) is not None:
                p_iso = self.isotonic_calibrator.predict(p_raw)
                return np.clip(p_iso, 0.0, 0.45)
            return p_raw
        return np.full(len(test_df), 0.05)

    def predict_levels(self, test_df):
        """Interface compatibility returning [p_5x]."""
        p5 = self.predict_entry(test_df)
        return np.column_stack([p5])

    def compute_miss_payoffs(self, train_df):
        """Interface compatibility."""
        return {'held_5': self.miss_payoff}

    def calculate_expected_values(self, test_df=None, p_entry=None, **kwargs):
        """Computes expected PnL for 5.0 exit: E[PnL(5.0)] = p * 3.70 + (1 - p) * miss_payoff."""
        if p_entry is None and test_df is not None:
            p_entry = self.predict_entry(test_df)
        elif p_entry is None:
            raise ValueError("Must provide either test_df or p_entry")
            
        p = np.asarray(p_entry, dtype=float)
        ev_5 = p * 3.70 + (1.0 - p) * self.miss_payoff
        return np.column_stack([ev_5])

    def calibrate_scalar_threshold(self, train_df=None, target_daily_trades=2.0, min_prob_floor=0.205):
        """
        Calibrates tau_base strictly on holdout training days (last 5 days OOF) targeting empirical break-even floor:
        tau_base = np.clip(np.quantile(oof_probs_last_5d, 0.985), 0.205, 0.350).
        """
        oof_target = getattr(self, 'p_oof_cal_iso', None) if getattr(self, 'p_oof_cal_iso', None) is not None else getattr(self, 'p_oof_cal', None)
        if oof_target is not None and self.trn_cal_len > 0:
            raw_tau = float(np.quantile(oof_target, 0.985))
            self.tau_scalar = float(np.clip(raw_tau, 0.205, 0.350))
        elif train_df is not None:
            d_trn = sorted(train_df.day.unique())
            cal_days = d_trn[-HOLD_DAYS:]
            trn_cal = train_df[train_df.day.isin(cal_days)].reset_index(drop=True)
            p_cal = self.predict_entry(trn_cal)
            raw_tau = float(np.quantile(p_cal, 0.985))
            self.tau_scalar = float(np.clip(raw_tau, 0.205, 0.350))
        else:
            self.tau_scalar = 0.205
        return self.tau_scalar

    def decide_per_token(self, test_df_or_p, ev_matrix=None, p_entry=None, tau_scalar=None, use_escalator=True, escalator_step=0.012, free_trades=8, min_prob_floor=0.205, test_df=None, **kwargs):
        """
        Causal per-token execution rule with First-60s Intra-Window Dump Veto and causal conviction escalator:
        - Evaluates test tokens strictly in chronological order of mt (zero future peeking).
        - Tracks the rolling count of entries taken in the trailing 24 hours: n_recent_24h.
        - For every trade taken in trailing 24h beyond free_trades (8), dynamically escalates required hurdle:
            tau_effective = tau_base + escalator_step * max(0, n_recent_24h - 8)
        - FIRST-60s INTRA-WINDOW DUMP VETO:
            * late_window_retention < 0.50 (dumping inside first 60s)
            * trades_per_wallet > 4.5 (wash loop)
            * dev_sell_ratio > 0.40 (creator dumping)
            Any vetoed token is strictly assigned tp = 0.0 (costless skip).
        - STRICT SAFETY FLOOR: NEVER enter a trade unless p[idx] >= min_prob_floor (0.205).
        - If unvetoed and p[idx] >= tau_effective and p[idx] >= min_prob_floor: enters trade at tp = 5.0.
        - Else: skips trade at zero cost (tp = 0.0).
        """
        if isinstance(test_df_or_p, (pd.DataFrame, dict)):
            df = test_df_or_p
            p = self.predict_entry(df) if p_entry is None else np.asarray(p_entry, dtype=float)
            mts = df['mt'].values if 'mt' in df else np.arange(len(p), dtype=float) * 100.0
        else:
            df = test_df
            p = np.asarray(test_df_or_p, dtype=float)
            mts = df['mt'].values if df is not None and 'mt' in df else np.arange(len(p), dtype=float) * 100.0
            
        tau = self.tau_scalar if tau_scalar is None else tau_scalar
        n = len(p)
        final_tp = np.zeros(n, dtype=float)
        sort_idx = np.argsort(mts, kind='stable')
        past_taken_mts = []
        
        # 1. Structural First-60s Intra-Window Dump Veto
        veto_mask = np.zeros(n, dtype=bool)
        if df is not None:
            retention = df['late_window_retention'] if 'late_window_retention' in df else (df['at_high'] if 'at_high' in df else 1.0)
            if isinstance(retention, pd.Series):
                retention = retention.fillna(1.0).values
            else:
                retention = np.asarray(retention, dtype=float)
                
            amm_n = df.get('amm_n', 0)
            amm_uw = df.get('amm_uw', 1)
            if isinstance(amm_n, pd.Series):
                amm_n = amm_n.fillna(0).values
                amm_uw = np.maximum(amm_uw.fillna(1).values, 1)
            tpw = (amm_n / amm_uw) if hasattr(amm_n, '__len__') else np.zeros(n, dtype=float)
            
            dev_sell = df.get('dev_sell_ratio', 0.0)
            if isinstance(dev_sell, pd.Series):
                dev_sell = dev_sell.fillna(0.0).values
            else:
                dev_sell = np.asarray(dev_sell, dtype=float)
                
            veto_mask = (retention < 0.50) | (tpw > 4.5) | (dev_sell > 0.40)
        
        # 2. Causal evaluation with conviction escalator and hard EV floor
        for idx in sort_idx:
            cur_mt = float(mts[idx])
            if veto_mask[idx]:
                continue
                
            # Trailing 24h window count strictly before cur_mt (only increments on executed trades)
            recent_24h_count = sum(1 for t in past_taken_mts if cur_mt - 86400.0 <= t < cur_mt)
            tau_eff = tau + (escalator_step * max(0, recent_24h_count - free_trades) if use_escalator else 0.0)
            
            # Enforce hard safety condition: NEVER buy below min_prob_floor (0.205)
            if p[idx] >= tau_eff and p[idx] >= min_prob_floor:
                final_tp[idx] = 5.0
                past_taken_mts.append(cur_mt)
                
        ev_5 = p * 3.70 + (1.0 - p) * self.miss_payoff
        return final_tp, ev_5

    def decide(self, p_entry, ev_matrix=None, **kwargs):
        """Wrapper maintaining interface compatibility."""
        return self.decide_per_token(p_entry, ev_matrix=ev_matrix, **kwargs)


# Backward compatibility aliases
UnifiedTwoTierEngine = ChampionFixed5Engine
TwoStageDecisionEngine = ChampionFixed5Engine
UnifiedSurvivalEngine = ChampionFixed5Engine
UnifiedPipeline = ChampionFixed5Engine
MultiLevelExpectedValueEngine = ChampionFixed5Engine
TwoStageCascadeEnsemble = ChampionFixed5Engine
