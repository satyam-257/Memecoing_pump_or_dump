"""Automated Optuna hyperparameter optimization targeting competition PnL and Sharpe."""
import os
import sys
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

from src.config import DATA_DIR, LEVELS, COST, WIN, EMBARGO, SEEDS
from src.features_creation import extract_creation_features
from src.features_tape import extract_tape_features, CURVE_COLS, AMM_COLS
from src.features_reputation import build_causal_reputation
from src.backtest import evaluate_pnl

# Suppress Optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)

def build_feature_dataset():
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    if 'finalx' in train_df.columns:
        train_df = train_df.rename(columns={'finalx': 'final_x'})
    train_df['day'] = pd.to_datetime(train_df.mt, unit='s').dt.strftime('%Y-%m-%d')
    
    cre_df = extract_creation_features(DATA_DIR)
    c_sample = os.path.join(DATA_DIR, '00_sample_pumpfun_swaps.csv')
    a_sample = os.path.join(DATA_DIR, '01_sample_pumpswap_swaps.csv')
    
    tape_features_df = None
    amm_edges_df = None
    if os.path.exists(c_sample) and os.path.exists(a_sample):
        c_cols = [c for c in CURVE_COLS if c not in ('slot', 'tx_idx')] + ['mint', 'failed', 'new_price', 'sol', 'ts_rel']
        a_cols = [c for c in AMM_COLS if c not in ('slot', 'tx_idx')] + ['mint', 'direction', 'base_token', 'new_price', 'sol', 'ts_rel', 'signing_wallet', 'fee_payer']
        c_df = pd.read_csv(c_sample, usecols=[c for c in c_cols if c in pd.read_csv(c_sample, nrows=1).columns])
        a_df = pd.read_csv(a_sample, usecols=[c for c in a_cols if c in pd.read_csv(a_sample, nrows=1).columns])
        tape_features_df = extract_tape_features(c_df, a_df)
        a_buys = a_df[(a_df.ts_rel <= 60)].copy().rename(columns={'signing_wallet': 'wallet'})
        amm_edges_df = a_buys.groupby(['mint', 'wallet'], as_index=False)['sol'].sum()

    creator_map = dict(zip(cre_df.mint, cre_df.creator))
    all_tokens = train_df[['mint', 'mt', 'max_x']].copy()
    rep_df = build_causal_reputation(all_tokens, amm_edges_df, creator_map=creator_map)

    m = train_df.merge(cre_df, on='mint', how='left')
    if tape_features_df is not None:
        m = m.merge(tape_features_df, on='mint', how='left')
    m = m.merge(rep_df, on='mint', how='left')
    m['age_s'] = m.mt - m.created_ts
    df = m.sort_values('mt', kind='stable').reset_index(drop=True)

    exclude_cols = {'mint', 'creator', 'mt', 'day', 'max_x', 'final_x', 'created_ts', 'dt'}
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]
    return df, feature_cols

def run_optuna_study(n_trials=20, n_eval_days=5):
    print(f"Building dataset for Optuna study across {n_eval_days} walk-forward evaluation days...")
    df, feature_cols = build_feature_dataset()
    days = sorted(df.day.unique())
    eval_days = days[-n_eval_days:]
    
    def objective(trial):
        # Hyperparameters for Stage 1 & Stage 2
        learning_rate = trial.suggest_float('learning_rate', 0.02, 0.08, log=True)
        num_leaves = trial.suggest_int('num_leaves', 15, 63)
        min_child_samples = trial.suggest_int('min_child_samples', 20, 80)
        subsample = trial.suggest_float('subsample', 0.60, 0.95)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.50, 0.90)
        
        top_n_daily = trial.suggest_int('top_n_daily', 1, 3)
        min_prob_hurdle = trial.suggest_float('min_prob_hurdle', 0.15, 0.30)
        min_ev_margin = trial.suggest_float('min_ev_margin', -0.10, 0.20)
        
        lgb_params = {
            'n_estimators': 250,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'subsample_freq': 1,
            'verbose': -1,
            'n_jobs': -1
        }
        
        daily_pnls = []
        for d in eval_days:
            val = df[df.day == d].reset_index(drop=True)
            trn = df[(df.day < d) & (df.mt <= val.mt.min() - WIN - EMBARGO)].reset_index(drop=True)
            if len(trn) < 500 or len(val) < 10:
                continue
                
            X_tr = trn[feature_cols].astype(float).fillna(0).values
            y_tr_3x = (trn['max_x'].values >= 3.0).astype(int)
            X_val = val[feature_cols].astype(float).fillna(0).values
            
            # Stage 1: Gatekeeper
            p1_val = np.mean([
                lgb.LGBMClassifier(random_state=s, **lgb_params)
                .fit(X_tr, y_tr_3x)
                .predict_proba(X_val)[:, 1]
                for s in SEEDS[:3]
            ], axis=0)
            
            # Stage 2: Utility Maximization
            probs_levels = np.zeros((len(val), len(LEVELS)), dtype=float)
            mxs_tr = trn['max_x'].values
            for j, lv in enumerate(LEVELS):
                y_lv = (mxs_tr >= lv).astype(int)
                probs_levels[:, j] = np.mean([
                    lgb.LGBMClassifier(random_state=s, **lgb_params)
                    .fit(X_tr, y_lv)
                    .predict_proba(X_val)[:, 1]
                    for s in SEEDS[:3]
                ], axis=0)
            probs_monotonic = np.minimum.accumulate(probs_levels, axis=1)
            
            # Miss Payoffs
            fxs_tr = trn['final_x'].values
            held = np.maximum(fxs_tr - 1.0 - COST, -1.0)
            held_by_level = np.array([
                float(held[mxs_tr < lv].mean()) if (mxs_tr < lv).any() else -0.95
                for lv in LEVELS
            ], dtype=float)
            
            win_pnl = np.array(LEVELS)[None, :] - 1.0 - COST
            ev_matrix = probs_monotonic * win_pnl + (1.0 - probs_monotonic) * held_by_level[None, :]
            
            best_idx = np.argmax(ev_matrix, axis=1)
            best_ev = ev_matrix[np.arange(len(ev_matrix)), best_idx]
            best_tp = np.array(LEVELS)[best_idx]
            
            # Allocation rule
            valid_mask = (p1_val >= min_prob_hurdle) & (best_ev >= min_ev_margin)
            valid_indices = np.where(valid_mask)[0]
            
            tp = np.zeros(len(val), dtype=float)
            if len(valid_indices) > 0 and top_n_daily > 0:
                scores = p1_val[valid_indices]
                order = np.argsort(-scores)
                selected = valid_indices[order[:top_n_daily]]
                tp[selected] = best_tp[selected]
                
            day_pnl = evaluate_pnl(val['max_x'].values, val['final_x'].values, tp)
            daily_pnls.append(day_pnl)
            
        tot_pnl = sum(daily_pnls)
        mean_pnl = np.mean(daily_pnls) if daily_pnls else 0.0
        std_pnl = np.std(daily_pnls, ddof=0) if daily_pnls else 1.0
        robust_score = mean_pnl - 0.25 * std_pnl
        return robust_score

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    print(f"Launching Optuna optimization with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print("\n" + "=" * 60)
    print("OPTUNA STUDY RESULTS")
    print("=" * 60)
    print(f"Best Trial Score (Robust Daily Sharpe): {study.best_value:.3f}")
    print("Best Hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    return study.best_params

if __name__ == '__main__':
    run_optuna_study(n_trials=10, n_eval_days=5)
