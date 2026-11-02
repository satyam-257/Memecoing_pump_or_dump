"""End-to-end Two-Stage Decoupled Gatekeeper training, validation, and submission generation."""
import os
import sys
import time
import numpy as np
import pandas as pd

from src.config import DATA_DIR, LEVELS, COST, WIN, EMBARGO
from src.features_creation import extract_creation_features
from src.features_tape import extract_tape_features, CURVE_COLS, AMM_COLS
from src.features_reputation import build_causal_reputation
from src.models import TwoStageDecisionEngine
from src.metrics import compute_pnl

def load_dataset():
    print(f"Loading dataset from: {DATA_DIR}")
    train_path = os.path.join(DATA_DIR, 'train.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    if 'finalx' in train.columns:
        train = train.rename(columns={'finalx': 'final_x'})
        
    for df in (train, test):
        df['day'] = pd.to_datetime(df.mt, unit='s').dt.strftime('%Y-%m-%d')
        
    print(f"Train universe: {len(train):,} tokens over {train.day.nunique()} days")
    print(f"Test universe:  {len(test):,} tokens on {test.day.iloc[0]}")
    return train, test

def main():
    t0 = time.time()
    train_df, test_df = load_dataset()
    
    # 1. Creation and launch metadata
    print("\n[1/5] Extracting token launch and creation metadata features...")
    cre_df = extract_creation_features(DATA_DIR)
    
    # 2. Tape features with WSOL correction and window acceleration
    print("[2/5] Ingesting high-frequency microstructure tapes...")
    c_sample = os.path.join(DATA_DIR, '00_sample_pumpfun_swaps.csv')
    a_sample = os.path.join(DATA_DIR, '01_sample_pumpswap_swaps.csv')
    
    tape_features_df = None
    amm_edges_df = None
    
    if os.path.exists(c_sample) and os.path.exists(a_sample):
        print("  Processing local sample tapes with direction disambiguation & Gini...")
        c_avail = pd.read_csv(c_sample, nrows=1).columns
        a_avail = pd.read_csv(a_sample, nrows=1).columns
        c_cols = [c for c in CURVE_COLS if c in c_avail] + [c for c in ['slot', 'tx_idx', 'signature'] if c in c_avail and c not in CURVE_COLS]
        a_cols = [c for c in AMM_COLS if c in a_avail] + [c for c in ['slot', 'tx_idx', 'signature'] if c in a_avail and c not in AMM_COLS]
        
        c_df = pd.read_csv(c_sample, usecols=list(set(c_cols)))
        a_df = pd.read_csv(a_sample, usecols=list(set(a_cols)))
        
        tape_features_df = extract_tape_features(c_df, a_df)
        
        # AMM buyers for causal reputation
        a_buys = a_df[(a_df.ts_rel <= 60)].copy()
        if 'signing_wallet' in a_buys.columns:
            a_buys = a_buys.rename(columns={'signing_wallet': 'wallet'})
        amm_edges_df = a_buys.groupby(['mint', 'wallet'], as_index=False)['sol'].sum()
        
    # 3. Causal reputation with recency decay
    print("[3/5] Computing causal participant and creator reputation (recency decay)...")
    creator_map = dict(zip(cre_df.mint, cre_df.creator))
    all_tokens = pd.concat([
        train_df[['mint', 'mt', 'max_x']],
        test_df[['mint', 'mt']].assign(max_x=np.nan)
    ], ignore_index=True)
    
    rep_df = build_causal_reputation(all_tokens, amm_edges_df, creator_map=creator_map)
    
    # Merge all feature tables
    def merge_features(df):
        m = df.merge(cre_df, on='mint', how='left')
        if tape_features_df is not None:
            m = m.merge(tape_features_df, on='mint', how='left')
        m = m.merge(rep_df, on='mint', how='left')
        m['age_s'] = m.mt - m.created_ts
        return m.sort_values('mt', kind='stable').reset_index(drop=True)
        
    TR = merge_features(train_df)
    TE = merge_features(test_df)
    
    exclude_cols = {'mint', 'creator', 'mt', 'day', 'max_x', 'final_x', 'created_ts', 'dt'}
    feature_cols = [c for c in TR.columns if c not in exclude_cols and np.issubdtype(TR[c].dtype, np.number)]
    print(f"Total features assembled: {len(feature_cols)}")
    
    # 4. Restored Champion Full-Feature Entry Gatekeeper & Fixed TP=5.0
    print("\n[4/5] Training Champion Entry Classifier (LGBM + CatBoost on P(>=5x), zero sample_weight distortion)...")
    engine = TwoStageDecisionEngine(feature_cols=feature_cols)
    
    engine.fit_entry(TR)
    
    # 5. Predict on test set
    print("\n[5/5] Generating unbiased test predictions and Fixed TP=5.0 allocation...")
    test_p = engine.predict_entry(TE)
    
    # Causal per-token execution rule (zero test ranking or peeking)
    tau_scalar = engine.calibrate_scalar_threshold(TR, target_daily_trades=2.0, min_prob_floor=0.23)
    print(f"Calibrated causal Stage 1 scalar cutoff on last 5 training days: tau = {tau_scalar:.4f} (floor >= 0.23)")
    test_tp, best_ev = engine.decide_per_token(TE, p_entry=test_p, tau_scalar=tau_scalar)
    
    # Build verified submission dataframe
    sub = pd.DataFrame({
        'mint': TE['mint'],
        'tp': test_tp
    })
    
    sample_sub_path = os.path.join(DATA_DIR, 'sample_submission.csv')
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        sub = sample_sub[['mint']].merge(sub, on='mint', how='left')
        sub['tp'] = sub['tp'].fillna(0.0).astype(float)
        assert len(sub) == len(sample_sub), f"Row count mismatch: {len(sub)} vs {len(sample_sub)}"
        assert sub['mint'].equals(sample_sub['mint']), "Mint alignment mismatch"
        
    valid_tp = (sub['tp'] == 0.0) | ((sub['tp'] >= 3.0) & (sub['tp'] <= 100.0))
    assert valid_tp.all(), "Invalid take-profit values found outside allowed competition bounds!"
    
    sub_path = 'submission.csv'
    sub.to_csv(sub_path, index=False)
    
    buys = int((sub['tp'] > 0).sum())
    print(f"\nSuccessfully generated {sub_path}:")
    print(f"Total tokens: {len(sub)}")
    print(f"Tokens bought (tp > 0): {buys} ({buys / len(sub) * 100:.2f}%)")
    print("Take-profit level allocation counts:")
    print(sub[sub.tp > 0]['tp'].value_counts().sort_index())
    print(f"Pipeline executed in {time.time() - t0:.1f}s")
    
if __name__ == '__main__':
    main()
