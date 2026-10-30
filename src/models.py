"""Multi-level expected value prediction engine and risk-averse policy optimization."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.config import TARGET_K, LEVELS, COST, SEEDS, LGB_PARAMS, WIN, EMBARGO, HOLD_DAYS

class MultiLevelExpectedValueEngine:
    """
    Multi-Level Expected Value Modeling:
    Trains classifiers across discrete target hurdles: LEVELS = (3, 4, 5, 7, 10, 15).
    Enforces monotonic probability structure: P(>= 15) <= P(>= 10) <= ... <= P(>= 3).
    Evaluates expected PnL per token across levels and assigns optimal take-profit target.
    """
    def __init__(self, feature_cols, levels=LEVELS, seeds=SEEDS, lgb_params=None, cost=COST):
        self.feature_cols = feature_cols
        self.levels = np.array(levels, dtype=float)
        self.seeds = seeds
        self.lgb_params = lgb_params or LGB_PARAMS
        self.cost = cost

    def fit_predict_levels(self, train_df, test_df):
        X_tr = train_df[self.feature_cols].astype(float).fillna(0).values
        X_te = test_df[self.feature_cols].astype(float).fillna(0).values
        
        mxs_tr = train_df['max_x'].values
        probs = np.zeros((len(test_df), len(self.levels)), dtype=float)
        
        for j, lv in enumerate(self.levels):
            y_lv = (mxs_tr >= lv).astype(int)
            if y_lv.sum() < 20:
                probs[:, j] = float(y_lv.mean())
                continue
                
            p_seed = np.mean([
                lgb.LGBMClassifier(random_state=s, **self.lgb_params)
                .fit(X_tr, y_lv)
                .predict_proba(X_te)[:, 1]
                for s in self.seeds
            ], axis=0)
            probs[:, j] = p_seed
            
        # Enforce strict probability monotonicity
        probs_monotonic = np.minimum.accumulate(probs, axis=1)
        return probs_monotonic

    def compute_held_payoffs(self, train_df):
        mxs = train_df['max_x'].values
        fxs = train_df['final_x'].values if 'final_x' in train_df.columns else train_df['finalx'].values
        held = np.maximum(fxs - 1.0 - self.cost, -1.0)
        
        held_by_level = []
        for lv in self.levels:
            miss = mxs < lv
            held_by_level.append(float(held[miss].mean()) if miss.any() else -0.95)
        return np.array(held_by_level, dtype=float)

    def calculate_expected_values(self, probs, held_by_level):
        """
        EV(k | x) = p_k * (k - 1 - cost) + (1 - p_k) * held_miss(k)
        """
        levels_row = self.levels[None, :]
        win_pnl = levels_row - 1.0 - self.cost
        held_row = held_by_level[None, :]
        
        ev = probs * win_pnl + (1.0 - probs) * held_row
        return ev

    def decide(self, ev_matrix, min_ev_margin=0.20, fallback_q=1.0):
        """
        Assigns optimal level argmax_k EV(k) if EV >= margin, else 0.0.
        If no token clears margin, applies fallback top quantile.
        """
        best_idx = np.argmax(ev_matrix, axis=1)
        best_ev = ev_matrix[np.arange(len(ev_matrix)), best_idx]
        best_level = self.levels[best_idx]
        
        tp = np.where(best_ev >= min_ev_margin, best_level, 0.0)
        
        # If absolute margin selects 0 tokens, select strictly the top fallback_q percent by rank
        if (tp > 0).sum() == 0 and fallback_q > 0:
            top_k = max(1, int(len(ev_matrix) * fallback_q / 100.0))
            tp = np.zeros(len(ev_matrix), dtype=float)
            if best_ev.max() > best_ev.min():
                order = np.argsort(-best_ev)
                top_indices = order[:top_k]
                tp[top_indices] = best_level[top_indices]
            
        return tp, best_ev

def optimize_ev_margin(model, train_df, margin_grid=(0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0), hold_days=HOLD_DAYS):
    """
    Finds optimal EV margin on holdout days maximizing risk-adjusted daily Sharpe.
    """
    d = sorted(train_df.day.unique())
    if len(d) <= hold_days:
        hold_days = max(1, len(d) // 3)
        
    cal = train_df[train_df.day >= d[-hold_days]].reset_index(drop=True)
    old = train_df[(train_df.day < d[-hold_days]) & (train_df.mt <= cal.mt.min() - WIN - EMBARGO)].reset_index(drop=True)
    if len(old) < 500:
        old = train_df[train_df.day < d[-hold_days]].reset_index(drop=True)
        
    probs = model.fit_predict_levels(old, cal)
    held = model.compute_held_payoffs(old)
    ev = model.calculate_expected_values(probs, held)
    
    best_margin = 0.20
    best_score = -9999.0
    
    for margin in margin_grid:
        tp, _ = model.decide(ev, min_ev_margin=margin, fallback_q=0.0)
        hit = (cal.max_x.values >= tp) & (tp > 0)
        held_loss = np.maximum(cal.final_x.values - 1.0 - COST, -1.0)
        pnl = np.where(tp <= 0, 0.0, np.where(hit, tp - 1.0 - COST, held_loss))
        
        daily = pd.Series(pnl).groupby(cal.day).sum()
        trades = (tp > 0).sum()
        if trades < 3:
            continue
            
        mean_pnl = float(daily.mean())
        std_pnl = float(daily.std(ddof=0))
        robust_sharpe = mean_pnl - 0.25 * std_pnl
        
        if robust_sharpe > best_score:
            best_score = robust_sharpe
            best_margin = margin
            
    return best_margin
