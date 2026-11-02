"""Executes the full 16-day embargoed walk-forward evaluation protocol and prints side-by-side comparison table."""
import os
import sys
import time
import numpy as np
import pandas as pd

from src.config import DATA_DIR, LEVELS, COST, WIN, EMBARGO
from src.features_creation import extract_creation_features
from src.features_tape import extract_tape_features, CURVE_COLS, AMM_COLS
from src.features_reputation import build_causal_reputation
from src.backtest import run_16day_comparison

def main():
    t0 = time.time()
    print("=" * 90)
    print("INITIALIZING 16-DAY EMBARGOED WALK-FORWARD EVALUATION")
    print("=" * 90)
    
    train_path = os.path.join(DATA_DIR, 'train.csv')
    test_path = os.path.join(DATA_DIR, 'test.csv')
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    if 'finalx' in train.columns:
        train = train.rename(columns={'finalx': 'final_x'})
        
    for df in (train, test):
        df['day'] = pd.to_datetime(df.mt, unit='s').dt.strftime('%Y-%m-%d')
        
    print(f"Loaded train universe: {len(train):,} tokens over {train.day.nunique()} days ({train.day.min()} to {train.day.max()})")
    
    # 1. Creation features
    print("Extracting creation features...")
    cre_df = extract_creation_features(DATA_DIR)
    
    # 2. Tape features
    print("Extracting tape features...")
    c_sample = os.path.join(DATA_DIR, '00_sample_pumpfun_swaps.csv')
    a_sample = os.path.join(DATA_DIR, '01_sample_pumpswap_swaps.csv')
    
    tape_features_df = None
    amm_edges_df = None
    
    if os.path.exists(c_sample) and os.path.exists(a_sample):
        c_avail = pd.read_csv(c_sample, nrows=1).columns
        a_avail = pd.read_csv(a_sample, nrows=1).columns
        
        c_cols = [c for c in CURVE_COLS if c in c_avail] + [c for c in ['slot', 'tx_idx', 'signature'] if c in c_avail and c not in CURVE_COLS]
        a_cols = [c for c in AMM_COLS if c in a_avail] + [c for c in ['slot', 'tx_idx', 'signature'] if c in a_avail and c not in AMM_COLS]
        
        c_df = pd.read_csv(c_sample, usecols=list(set(c_cols)))
        a_df = pd.read_csv(a_sample, usecols=list(set(a_cols)))
        
        tape_features_df = extract_tape_features(c_df, a_df)
        a_buys = a_df[a_df.ts_rel <= 60].copy()
        if 'signing_wallet' in a_buys.columns:
            a_buys = a_buys.rename(columns={'signing_wallet': 'wallet'})
        amm_edges_df = a_buys.groupby(['mint', 'wallet'], as_index=False)['sol'].sum()
        
    # 3. Causal reputation features with lazy 7-day half-life decay and m=5 shrinkage
    print("Computing causal participant and creator reputation...")
    creator_map = dict(zip(cre_df.mint, cre_df.creator))
    all_tokens = pd.concat([
        train[['mint', 'mt', 'max_x']],
        test[['mint', 'mt']].assign(max_x=np.nan)
    ], ignore_index=True)
    
    rep_df = build_causal_reputation(all_tokens, amm_edges_df, creator_map=creator_map)
    
    # Merge
    print("Assembling feature matrix...")
    TR = train.merge(cre_df, on='mint', how='left')
    if tape_features_df is not None:
        TR = TR.merge(tape_features_df, on='mint', how='left')
    TR = TR.merge(rep_df, on='mint', how='left')
    TR['age_s'] = TR.mt - TR.created_ts
    TR = TR.sort_values('mt', kind='stable').reset_index(drop=True)
    
    exclude_cols = {'mint', 'creator', 'mt', 'day', 'max_x', 'final_x', 'created_ts', 'dt'}
    feature_cols = [c for c in TR.columns if c not in exclude_cols and np.issubdtype(TR[c].dtype, np.number)]
    print(f"Total features: {len(feature_cols)}")
    
    # 4. Run 16-day walk-forward evaluation
    b_df, c_df = run_16day_comparison(TR, feature_cols)
    print(f"\nCompleted 16-day walk-forward evaluation in {time.time()-t0:.1f}s.")

if __name__ == '__main__':
    main()

