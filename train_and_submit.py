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
    print("[2/5] Ingesting high-frequency microstructure tapes from daily parquets & sample archives...")
    import glob
    c_files = sorted(glob.glob(os.path.join(DATA_DIR, 'day_*_pumpfun_all_swaps.parquet')))
    a_files = sorted(glob.glob(os.path.join(DATA_DIR, 'day_*_pumpswap_all_swaps.parquet')))
    c_sample = os.path.join(DATA_DIR, '00_sample_pumpfun_swaps.csv')
    a_sample = os.path.join(DATA_DIR, '01_sample_pumpswap_swaps.csv')
    
    # Mints requiring tape extraction
    needed_mints = set(train_df.mint).union(set(test_df.mint))
    check_new_path = os.path.join(DATA_DIR, 'check_new', 'test.csv')
    if os.path.exists(check_new_path):
        t409_df = pd.read_csv(check_new_path)
        needed_mints = needed_mints.union(set(t409_df.mint))
        
    print(f"  Target tokens for tape ingestion: {len(needed_mints):,}")
    print(f"  Found {len(c_files)} curve parquet partitions and {len(a_files)} AMM parquet partitions.")
    
    c_dfs = []
    a_dfs = []
    
    # Fast partition reader with mint pre-filtering
    c_cols = [c for c in CURVE_COLS if c != 'failed'] + ['failed']
    for f in c_files:
        avail = pd.read_parquet(f, columns=['mint'])
        mask = avail.mint.isin(needed_mints)
        if mask.any():
            c_dfs.append(pd.read_parquet(f, columns=c_cols)[mask])
            
    for f in a_files:
        avail = pd.read_parquet(f, columns=['mint'])
        mask = avail.mint.isin(needed_mints)
        if mask.any():
            a_dfs.append(pd.read_parquet(f, columns=AMM_COLS)[mask])
            
    if os.path.exists(c_sample) and os.path.exists(a_sample):
        c_avail = pd.read_csv(c_sample, nrows=1).columns
        a_avail = pd.read_csv(a_sample, nrows=1).columns
        c_c = [c for c in CURVE_COLS if c in c_avail]
        a_c = [c for c in AMM_COLS if c in a_avail]
        c_csv = pd.read_csv(c_sample, usecols=c_c)
        a_csv = pd.read_csv(a_sample, usecols=a_c)
        if len(c_csv):
            c_dfs.append(c_csv[c_csv.mint.isin(needed_mints)])
        if len(a_csv):
            a_dfs.append(a_csv[a_csv.mint.isin(needed_mints)])
            
    c_all = pd.concat(c_dfs, ignore_index=True) if c_dfs else pd.DataFrame(columns=CURVE_COLS)
    a_all = pd.concat(a_dfs, ignore_index=True) if a_dfs else pd.DataFrame(columns=AMM_COLS)
    print(f"  Ingested {len(c_all):,} curve swaps and {len(a_all):,} AMM swaps across all partitions.")
    
    tape_features_df = extract_tape_features(c_all, a_all)
    print(f"  Extracted 57 tape features for {len(tape_features_df):,} tokens.")
    
    # AMM buyers for causal reputation
    a_buys = a_all[(a_all.ts_rel <= 60)].copy()
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
    
    # HARD ASSERTION FOR TEST SET TAPE COVERAGE
    missing_tape_rate = (TE['amm_n'].isna() | (TE['amm_n'] == 0)).mean()
    print(f"\n[Test Tape Audit] Test token AMM tape missing rate: {missing_tape_rate*100:.2f}% ({int(missing_tape_rate * len(TE))}/{len(TE)})")
    if missing_tape_rate > 0.10:
        raise RuntimeError(f"Missing parquet swap partitions for test.csv! {missing_tape_rate * 100:.1f}% missing")
    
    exclude_cols = {'mint', 'creator', 'mt', 'day', 'max_x', 'final_x', 'created_ts', 'dt'}
    feature_cols = [c for c in TR.columns if c not in exclude_cols and np.issubdtype(TR[c].dtype, np.number)]
    print(f"Total features assembled: {len(feature_cols)}")
    non_null_rate = 1.0 - TE[feature_cols].isna().mean().mean()
    print(f"Test token feature non-null rate: {non_null_rate*100:.2f}%")
    
    # 4. Restored Champion Full-Feature Entry Gatekeeper & Fixed TP=5.0
    print("\n[4/5] Training Champion Entry Classifier (LGBM + CatBoost on P(>=5x), zero sample_weight distortion)...")
    engine = TwoStageDecisionEngine(feature_cols=feature_cols)
    
    engine.fit_entry(TR)
    
    # 5. Predict on test set
    print("\n[5/5] Generating unbiased test predictions and Fixed TP=5.0 allocation...")
    test_p = engine.predict_entry(TE)
    
    # Causal per-token execution rule (zero test ranking or peeking)
    tau_scalar = engine.calibrate_scalar_threshold(TR, min_prob_floor=0.205)
    print(f"Calibrated causal Stage 1 scalar cutoff on holdout OOF (last 5 training days): tau_base = {tau_scalar:.4f} (break-even floor >= 0.205)")
    test_tp, best_ev = engine.decide_per_token(
        TE, 
        p_entry=test_p, 
        tau_scalar=tau_scalar, 
        min_prob_floor=0.205, 
        escalator_step=0.012, 
        free_trades=8
    )
    
    # Diagnostic: Top 15 predicted tokens in test universe
    sort_te_idx = np.argsort(test_p)[::-1]
    print("\n--- TOP 15 PREDICTED TOKENS IN TEST UNIVERSE (data/test.csv) ---")
    for r, idx in enumerate(sort_te_idx[:15], 1):
        m_name = TE['mint'].iloc[idx]
        print(f"Rank {r:2d}: mint={m_name[:20]}... | P_cal={test_p[idx]:.4f} | tau={tau_scalar:.4f} | tp={test_tp[idx]}")

    active_test_buys = int((test_tp > 0).sum())
    print(f"\nActive test trades taken: {active_test_buys} / {len(TE)} ({active_test_buys / len(TE) * 100:.2f}%)")
    max_p_cal = float(np.max(test_p))
    print(f"Max P_cal on test set: {max_p_cal:.4f}")

    # Also evaluate on 409-token benchmark set if present
    check_new_path = os.path.join(DATA_DIR, 'check_new', 'test.csv')
    if os.path.exists(check_new_path):
        t409_df = pd.read_csv(check_new_path)
        t409_df['day'] = pd.to_datetime(t409_df.mt, unit='s').dt.strftime('%Y-%m-%d')
        TE409 = merge_features(t409_df)
        p409 = engine.predict_entry(TE409)
        tp409, _ = engine.decide_per_token(
            TE409, 
            p_entry=p409, 
            tau_scalar=tau_scalar, 
            min_prob_floor=0.205, 
            escalator_step=0.012, 
            free_trades=8
        )
        buys_409 = int((tp409 > 0).sum())
        print(f"\n--- BENCHMARK VERIFICATION ON 409 TEST TOKENS (data/check_new/test.csv) ---")
        print(f"Tokens bought (tp > 0) on 409 set: {buys_409} / {len(t409_df)} (Target: 4 to 15 trades)")
        s409_idx = np.argsort(p409)[::-1]
        for r, idx in enumerate(s409_idx[:15], 1):
            m_name = TE409['mint'].iloc[idx]
            print(f"Rank {r:2d}: mint={m_name[:20]}... | P_cal={p409[idx]:.4f} | tp={tp409[idx]}")
        # Log benchmark verification counts without blocking live test submission
        print(f"Benchmark verification note: {buys_409} tokens passed all hurdle checks on 409-token set.")

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
