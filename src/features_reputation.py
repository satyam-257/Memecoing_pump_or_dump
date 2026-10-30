"""Causal participant and deployer reputation engine with strict 1-hour embargo and recency weighting."""
import numpy as np
import pandas as pd
from src.config import WIN, TARGET_K, TAU_DECAY

REPUTATION_COLS = [
    'w_known', 'w_hit_max', 'w_hit_mean', 'w_hit_top3', 
    'w_elite_n', 'w_elite_share', 'w_eb_score',
    'w_decay_eb_score', 'w_decay_hits',
    'c_known', 'c_hit_rate', 'c_decay_rate'
]

def build_causal_reputation(tokens_df, amm_edges_df, creator_map=None, target_k=TARGET_K, win=WIN, tau_decay=TAU_DECAY):
    """
    Computes participant reputation under strict causal ordering with exponential recency decay.
    Only tokens resolved at least WIN (3600s) seconds before migration are committed.
    """
    tok = tokens_df[['mint', 'mt']].copy()
    if 'max_x' in tokens_df.columns:
        tok['max_x'] = tokens_df['max_x'].values
    else:
        tok['max_x'] = np.nan
        
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
    
    # Recency decayed counts
    decay_appear = np.zeros(nW, dtype=np.float32)
    decay_hit = np.zeros(nW, dtype=np.float32)
    last_update_ts = np.zeros(nW, dtype=np.int64)
    
    # Creator stats tracking
    creator_appear = {}
    creator_hit = {}
    creator_decay_ap = {}
    creator_decay_ht = {}
    creator_last_ts = {}
    
    mts = tok.mt.values
    mxs = tok.max_x.values
    mints = tok.mint.values
    
    out = np.zeros((len(tok), len(REPUTATION_COLS)), dtype=float)
    j = 0
    
    # Global prior for empirical Bayes shrinkage (baseline hit rate ~6%)
    alpha_prior = 1.0
    beta_prior = 15.0
    
    for i in range(len(tok)):
        cur_t = mts[i]
        # Advance resolution pointer and commit resolved tokens
        while j < len(tok) and mts[j] <= cur_t - win:
            t_res = mts[j]
            if not np.isnan(mxs[j]):
                # Commit wallet outcomes
                w = by_mint.get(mints[j])
                if w is not None:
                    wids = w[0]
                    np.add.at(appear, wids, 1)
                    
                    # Apply exponential decay to updated wallets
                    decay_factor = np.exp(-(cur_t - t_res) / max(tau_decay, 1.0))
                    np.add.at(decay_appear, wids, decay_factor)
                    
                    if mxs[j] >= target_k:
                        np.add.at(hit, wids, 1)
                        np.add.at(decay_hit, wids, decay_factor)
                        
                # Commit creator outcomes
                if creator_map is not None:
                    c_wallet = creator_map.get(mints[j])
                    if c_wallet and str(c_wallet) != 'nan' and len(str(c_wallet)) > 0:
                        decay_factor = np.exp(-(cur_t - t_res) / max(tau_decay, 1.0))
                        creator_appear[c_wallet] = creator_appear.get(c_wallet, 0) + 1
                        creator_decay_ap[c_wallet] = creator_decay_ap.get(c_wallet, 0.0) + decay_factor
                        if mxs[j] >= target_k:
                            creator_hit[c_wallet] = creator_hit.get(c_wallet, 0) + 1
                            creator_decay_ht[c_wallet] = creator_decay_ht.get(c_wallet, 0.0) + decay_factor
            j += 1
            
        # Compute reputation features for token i
        w = by_mint.get(mints[i])
        if w is not None:
            wids, sols = w
            ap = appear[wids]
            known = ap >= 3
            
            if known.sum() > 0:
                ht = hit[wids][known]
                ap_k = ap[known]
                rate = ht / ap_k
                
                # Empirical Bayes shrunk rate
                eb_rates = (ht + alpha_prior) / (ap_k + alpha_prior + beta_prior)
                
                # Decayed EB rate
                d_ap = decay_appear[wids][known]
                d_ht = decay_hit[wids][known]
                decay_eb = (d_ht + alpha_prior) / (d_ap + alpha_prior + beta_prior)
                
                elite = (ap >= 5) & (hit[wids] >= 1)
                
                out[i, 0] = int(known.sum())
                out[i, 1] = float(rate.max())
                out[i, 2] = float(rate.mean())
                out[i, 3] = float(np.sort(rate)[-3:].mean())
                out[i, 4] = int(elite.sum())
                out[i, 5] = float(sols[elite].sum() / max(sols.sum(), 1e-9))
                out[i, 6] = float(eb_rates.max())
                out[i, 7] = float(decay_eb.max())
                out[i, 8] = float(d_ht.sum())
                
        # Creator features
        if creator_map is not None:
            c_wallet = creator_map.get(mints[i])
            if c_wallet and str(c_wallet) != 'nan':
                c_ap = creator_appear.get(c_wallet, 0)
                if c_ap > 0:
                    c_ht = creator_hit.get(c_wallet, 0)
                    out[i, 9] = 1.0  # creator known
                    out[i, 10] = float(c_ht / c_ap)
                    out[i, 11] = float((creator_decay_ht.get(c_wallet, 0.0) + alpha_prior) / 
                                       (creator_decay_ap.get(c_wallet, 0.0) + alpha_prior + beta_prior))
                    
    rep_df = pd.DataFrame(out, columns=REPUTATION_COLS)
    rep_df['mint'] = mints
    return rep_df
