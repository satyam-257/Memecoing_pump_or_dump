"""Causal participant and deployer reputation engine with strict 1-hour embargo, 7-day half-life decay, and m-estimate Bayesian shrinkage."""
import os
import numpy as np
import pandas as pd
from src.config import WIN, TARGET_K

# 7-day half life decay constant
HALF_LIFE_SECONDS = 7.0 * 86400.0
DECAY_LAMBDA = np.log(2.0) / HALF_LIFE_SECONDS
GLOBAL_BASE_RATE = 0.0607
M_SHRINKAGE = 5.0

REPUTATION_COLS = [
    'w_known', 'w_hit_max', 'w_hit_mean', 'w_hit_top3', 
    'w_elite_n', 'w_elite_share', 'w_eb_score',
    'w_decay_eb_score', 'w_decay_hits',
    'w_shrunk_rate_max', 'w_shrunk_rate_mean', 'w_shrunk_rate_top3',
    'w_shrunk_sol_rate_max', 'w_shrunk_sol_rate_mean', 'w_decay_hits_weighted',
    'w_elite_size_share',
    'c_known', 'c_hit_rate', 'c_decay_rate', 'c_shrunk_rate',
    'c_launches', 'c_median_peak', 'c_max_peak',
    'c_hit_3x', 'c_hit_5x', 'c_last_peak', 'c_time_since_last'
]

def build_causal_reputation(tokens_df, amm_edges_df, creator_map=None, archive_df=None, target_k=TARGET_K, win=WIN, m_shrinkage=M_SHRINKAGE, base_rate=GLOBAL_BASE_RATE):
    """
    Computes participant and deployer reputation under strict causal ordering:
    - Invariant: Two-pointer 3600s embargo (mts[j] <= cur_t - 3600).
    - Warm-starts wallet and creator reputation dictionaries using historical archive data
      prior to the 30-day window so creator and smart-money histories have full lookback depth.
    - Empirical Bayesian Shrinkage (m-estimate smoothing) with m = 5:
        shrunk_rate = (hits + m * base_rate) / (appearances + m)
    - SOL trade size weighting and exponential recency decay (7-day half-life)
      so recent big-size winners dominate stale micro-trades.
    - Causal deployer historical metrics (prior launches, median peak, 3x/5x hit rates).
    """
    tok = tokens_df[['mint', 'mt']].copy()
    if 'max_x' in tokens_df.columns:
        tok['max_x'] = tokens_df['max_x'].values
    else:
        tok['max_x'] = np.nan
    target_mints = set(tok['mint'])
        
    # Warm-start from archive dataset if available
    if archive_df is None:
        for candidate in ['data/train_archive.csv', 'train_archive.csv', '/kaggle/input/competitions/memecoins-pump-or-dump/train_archive.csv']:
            if os.path.exists(candidate):
                try:
                    archive_df = pd.read_csv(candidate)
                    break
                except Exception:
                    pass
                    
    if archive_df is not None and len(archive_df) > 0:
        arc = archive_df[['mint', 'mt']].copy()
        if 'max_x' in archive_df.columns:
            arc['max_x'] = archive_df['max_x'].values
        else:
            arc['max_x'] = np.nan
        min_t = tok.mt.min()
        arc = arc[arc.mt < min_t]
        if len(arc) > 0:
            tok = pd.concat([arc, tok], ignore_index=True)
            if creator_map is not None and 'creator' in archive_df.columns:
                for m, c in zip(archive_df['mint'], archive_df['creator']):
                    if m not in creator_map:
                        creator_map[m] = c
                        
    tok = tok.sort_values('mt', kind='stable').reset_index(drop=True)
    
    # Process wallet edges
    if amm_edges_df is not None and len(amm_edges_df) > 0:
        E = amm_edges_df.copy()
        E['wid'] = E['wallet'].astype('category').cat.codes.astype(np.int64)
        by_mint = {
            m: (g.wid.values, g.sol.values)
            for m, g in E.sort_values('mint', kind='stable').groupby('mint', sort=False)
        }
        nW = int(E.wid.max()) + 1
    else:
        by_mint = {}
        nW = 1
        
    appear = np.zeros(nW, dtype=np.int32)
    hit = np.zeros(nW, dtype=np.int32)
    
    # Decayed continuous tracking using timestamp of last update
    last_update_ts = np.zeros(nW, dtype=np.float64)
    decay_appear = np.zeros(nW, dtype=np.float32)
    decay_hit = np.zeros(nW, dtype=np.float32)
    decay_sol_appear = np.zeros(nW, dtype=np.float32)
    decay_sol_hit = np.zeros(nW, dtype=np.float32)
    
    # Creator stats tracking
    creator_appear = {}
    creator_hit = {}
    creator_decay_ap = {}
    creator_decay_ht = {}
    creator_past_peaks = {}
    creator_last_ts = {}
    
    mts = tok.mt.values
    mxs = tok.max_x.values
    mints = tok.mint.values
    n = len(tok)
    
    out = np.zeros((n, len(REPUTATION_COLS)), dtype=float)
    j = 0
    
    for i in range(n):
        cur_t = float(mts[i])
        
        # Advance resolution pointer and commit resolved tokens (strictly mt_j <= cur_t - 3600)
        while j < n and mts[j] <= cur_t - win:
            t_res = float(mts[j])
            if not np.isnan(mxs[j]):
                is_hit = int(mxs[j] >= target_k)
                # Commit wallet outcomes
                w = by_mint.get(mints[j])
                if w is not None:
                    wids, sols = w
                    # Update each wallet with lazy continuous decay
                    for wid, sol_val in zip(wids, sols):
                        sol_w = float(min(max(sol_val, 0.01), 10.0))  # cap between 0.01 and 10 SOL
                        prev_t = last_update_ts[wid]
                        if prev_t > 0:
                            dt = max(t_res - prev_t, 0.0)
                            dec = float(np.exp(-dt * DECAY_LAMBDA))
                            decay_appear[wid] = decay_appear[wid] * dec
                            decay_hit[wid] = decay_hit[wid] * dec
                            decay_sol_appear[wid] = decay_sol_appear[wid] * dec
                            decay_sol_hit[wid] = decay_sol_hit[wid] * dec
                        
                        appear[wid] += 1
                        hit[wid] += is_hit
                        decay_appear[wid] += 1.0
                        decay_hit[wid] += float(is_hit)
                        decay_sol_appear[wid] += sol_w
                        decay_sol_hit[wid] += sol_w * float(is_hit)
                        last_update_ts[wid] = t_res
                        
                # Commit creator outcomes
                if creator_map is not None:
                    c_wallet = creator_map.get(mints[j])
                    if c_wallet and str(c_wallet) != 'nan' and len(str(c_wallet)) > 0:
                        prev_c_t = creator_last_ts.get(c_wallet, t_res)
                        dt = max(t_res - prev_c_t, 0.0)
                        dec = float(np.exp(-dt * DECAY_LAMBDA))
                        
                        creator_appear[c_wallet] = creator_appear.get(c_wallet, 0) + 1
                        creator_decay_ap[c_wallet] = creator_decay_ap.get(c_wallet, 0.0) * dec + 1.0
                        
                        if c_wallet not in creator_past_peaks:
                            creator_past_peaks[c_wallet] = []
                        creator_past_peaks[c_wallet].append(float(mxs[j]))
                        creator_last_ts[c_wallet] = t_res
                        
                        if is_hit:
                            creator_hit[c_wallet] = creator_hit.get(c_wallet, 0) + 1
                            creator_decay_ht[c_wallet] = creator_decay_ht.get(c_wallet, 0.0) * dec + 1.0
            j += 1
            
        # Compute reputation features for token i at time cur_t
        w = by_mint.get(mints[i])
        if w is not None:
            wids, sols = w
            ap = appear[wids]
            known = ap >= 2
            
            if known.sum() > 0:
                k_wids = wids[known]
                k_sols = sols[known]
                ap_k = ap[known]
                ht_k = hit[k_wids]
                
                # 1. Unweighted Empirical Bayes shrinkage (m-estimate smoothing, m = 5)
                # shrunk_rate = (hits + m * base_rate) / (appearances + m)
                shrunk_rates = (ht_k + m_shrinkage * base_rate) / (ap_k + m_shrinkage)
                raw_rates = ht_k / np.maximum(ap_k, 1)
                
                # 2. SOL-size weighted & 7-day half-life decayed shrinkage
                k_last_t = last_update_ts[k_wids]
                dt_vals = np.maximum(cur_t - k_last_t, 0.0)
                decay_multipliers = np.exp(-dt_vals * DECAY_LAMBDA)
                
                cur_decay_sol_ap = decay_sol_appear[k_wids] * decay_multipliers
                cur_decay_sol_ht = decay_sol_hit[k_wids] * decay_multipliers
                
                # Shrunk SOL-weighted rate
                shrunk_sol_rates = (cur_decay_sol_ht + m_shrinkage * base_rate) / (cur_decay_sol_ap + m_shrinkage)
                
                # Elite wallet identification: high shrunk rate with solid history
                is_elite = (shrunk_sol_rates >= 0.15) & (ap_k >= 3)
                
                out[i, 0] = int(known.sum())
                out[i, 1] = float(np.max(raw_rates))
                out[i, 2] = float(np.mean(raw_rates))
                out[i, 3] = float(np.sort(raw_rates)[-3:].mean())
                out[i, 4] = int(is_elite.sum())
                out[i, 5] = float(k_sols[is_elite].sum() / max(sols.sum(), 1e-9))
                out[i, 6] = float(np.max(shrunk_rates))
                out[i, 7] = float(np.max(shrunk_sol_rates))
                out[i, 8] = float(cur_decay_sol_ht.sum())
                
                # Upgraded m-estimate shrinkage metrics
                out[i, 9] = float(np.max(shrunk_rates))
                out[i, 10] = float(np.mean(shrunk_rates))
                out[i, 11] = float(np.sort(shrunk_rates)[-3:].mean())
                out[i, 12] = float(np.max(shrunk_sol_rates))
                out[i, 13] = float(np.mean(shrunk_sol_rates))
                out[i, 14] = float(cur_decay_sol_ht.sum())
                out[i, 15] = float(k_sols[is_elite].sum() / max(sols.sum(), 1e-9))
                
        # Creator features
        if creator_map is not None:
            c_wallet = creator_map.get(mints[i])
            if c_wallet and str(c_wallet) != 'nan':
                c_ap = creator_appear.get(c_wallet, 0)
                if c_ap > 0:
                    c_ht = creator_hit.get(c_wallet, 0)
                    dt_c = max(cur_t - creator_last_ts.get(c_wallet, cur_t), 0.0)
                    dec_c = float(np.exp(-dt_c * DECAY_LAMBDA))
                    c_dec_ap = creator_decay_ap.get(c_wallet, 0.0) * dec_c
                    c_dec_ht = creator_decay_ht.get(c_wallet, 0.0) * dec_c
                    
                    out[i, 16] = 1.0  # creator known
                    out[i, 17] = float(c_ht / c_ap)
                    out[i, 18] = float((c_dec_ht + m_shrinkage * base_rate) / (c_dec_ap + m_shrinkage))
                    out[i, 19] = float((c_ht + m_shrinkage * base_rate) / (c_ap + m_shrinkage))
                    
                if c_wallet in creator_past_peaks:
                    peaks = creator_past_peaks[c_wallet]
                    out[i, 20] = len(peaks)
                    out[i, 21] = float(np.median(peaks))
                    out[i, 22] = float(np.max(peaks))
                    out[i, 23] = float(np.mean([p >= 3.0 for p in peaks]))
                    out[i, 24] = float(np.mean([p >= 5.0 for p in peaks]))
                    out[i, 25] = float(peaks[-1])
                    out[i, 26] = float(cur_t - creator_last_ts.get(c_wallet, cur_t))
                    
    rep_df = pd.DataFrame(out, columns=REPUTATION_COLS)
    rep_df['mint'] = mints
    if len(target_mints) < len(rep_df):
        rep_df = rep_df[rep_df['mint'].isin(target_mints)].reset_index(drop=True)
    return rep_df
