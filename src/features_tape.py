"""Microstructure feature engineering from bonding curve and AMM transaction tapes with WSOL direction correction."""
import numpy as np
import pandas as pd
from src.config import WSOL

CURVE_COLS = ['failed', 'mint', 'ts_rel', 'slot', 'tx_idx', 'direction', 'sol', 'new_price', 'wallet', 'provided_gas_fee']
AMM_COLS   = [
    'mint', 'ts_rel', 'slot', 'tx_idx', 'direction', 'base_token', 'sol', 'new_price', 
    'signing_wallet', 'fee_payer', 'provided_gas_fee', 'coin_creator',
    'pool_base_token_reserves_after', 'pool_quote_token_reserves_after'
]

def extract_tape_features(c_df, a_df):
    """
    Vectorized high-frequency microstructure signals with:
    - Direction correction when WSOL is base token
    - Fee payer relayer mismatch (MEV bot proxy)
    - Creator trading activity in the first minute
    - Window flow acceleration (0-15s, 15-30s, 30-60s)
    - Reserve pool depth metrics
    """
    if 'signing_wallet' in a_df.columns:
        a_df = a_df.rename(columns={'signing_wallet': 'wallet'})
        
    # Maintain strict blockchain transaction ordering: slot, tx_idx
    sort_cols = [c for c in ['mint', 'slot', 'tx_idx'] if c in a_df.columns]
    if sort_cols:
        a_df = a_df.sort_values(sort_cols, kind='mergesort')
        
    # Price initialization
    for df in (c_df, a_df):
        if 'new_price' in df.columns:
            df['price'] = df.new_price.replace(0, np.nan)
        else:
            df['price'] = np.nan

    # Direction correction for AMM swaps
    if 'direction' in a_df.columns:
        raw_b = a_df.direction.astype(str).str.upper().str.startswith('B')
        if 'base_token' in a_df.columns:
            sol_is_base = a_df.base_token.astype(str) == WSOL
            a_df['is_buy'] = np.where(sol_is_base, ~raw_b, raw_b).astype(np.int8)
        else:
            a_df['is_buy'] = raw_b.astype(np.int8)
    else:
        a_df['is_buy'] = 1
        
    if 'direction' in c_df.columns:
        c_df['is_buy'] = c_df.direction.astype(str).str.upper().str.startswith('B').astype(np.int8)
    else:
        c_df['is_buy'] = 1

    # Buy and sell SOL amounts
    a_df['buy_sol'] = a_df['sol'] * a_df['is_buy']
    a_df['sell_sol'] = a_df['sol'] * (1 - a_df['is_buy'])
    
    # Fee payer vs signing wallet mismatch (MEV bot / relayer detector)
    if 'fee_payer' in a_df.columns and 'wallet' in a_df.columns:
        a_df['mismatch'] = (a_df['wallet'].astype(str) != a_df['fee_payer'].astype(str)).astype(np.int8)
    else:
        a_df['mismatch'] = 0
        
    # Creator trading in first minute
    if 'coin_creator' in a_df.columns and 'wallet' in a_df.columns:
        a_df['is_creator'] = (a_df['wallet'].astype(str) == a_df['coin_creator'].astype(str)).astype(np.int8)
        a_df['creator_sell_sol'] = a_df['sol'] * a_df['is_creator'] * (1 - a_df['is_buy'])
        a_df['creator_buy_sol'] = a_df['sol'] * a_df['is_creator'] * a_df['is_buy']
    else:
        a_df['is_creator'] = 0
        a_df['creator_sell_sol'] = 0.0
        a_df['creator_buy_sol'] = 0.0

    # Reserve pool liquidity depth in SOL units (9 decimals)
    if 'pool_base_token_reserves_after' in a_df.columns and 'pool_quote_token_reserves_after' in a_df.columns and 'base_token' in a_df.columns:
        sol_is_base = a_df.base_token.astype(str) == WSOL
        a_df['depth_sol'] = np.where(
            sol_is_base, 
            a_df['pool_base_token_reserves_after'], 
            a_df['pool_quote_token_reserves_after']
        ) / 1e9
    else:
        a_df['depth_sol'] = np.nan

    g_amm = a_df.groupby('mint')
    F = pd.DataFrame(index=g_amm.size().index)
    
    # 1. AMM Participation & Volume
    F['amm_n'] = g_amm.size()
    F['amm_sol'] = g_amm['sol'].sum()
    F['amm_uw'] = g_amm['wallet'].nunique()
    F['trades_per_wallet'] = F['amm_n'] / F['amm_uw'].replace(0, np.nan)
    
    # Volume concentration: Top-1 wallet share and HHI
    ws = a_df.groupby(['mint', 'wallet'])['sol'].sum()
    ws_tot = ws.groupby('mint').sum().replace(0, np.nan)
    F['amm_top1'] = ws.groupby('mint').max() / ws_tot
    ws_norm = ws / ws.groupby('mint').transform('sum').replace(0, np.nan)
    F['amm_hhi'] = (ws_norm ** 2).groupby('mint').sum()
    
    # 2. Three Intra-Minute Window Slices: W0 (0-15s), W1 (15-30s), W2 (30-60s)
    w0 = a_df[a_df.ts_rel < 15]
    w1 = a_df[(a_df.ts_rel >= 15) & (a_df.ts_rel < 30)]
    w2 = a_df[a_df.ts_rel >= 30]
    
    w0_sol = w0.groupby('mint')['sol'].sum().reindex(F.index).fillna(0.0)
    w0_buy = w0.groupby('mint')['buy_sol'].sum().reindex(F.index).fillna(0.0)
    w2_sol = w2.groupby('mint')['sol'].sum().reindex(F.index).fillna(0.0)
    w2_buy = w2.groupby('mint')['buy_sol'].sum().reindex(F.index).fillna(0.0)
    
    F['flow_accel'] = (w2_sol / (w0_sol + 1e-4)) - 1.0
    F['bp_0_15'] = w0_buy / (w0_sol + 1e-4)
    F['bp_30_60'] = w2_buy / (w2_sol + 1e-4)
    F['bp_delta'] = F['bp_30_60'] - F['bp_0_15']
    
    # Terminal order flow (45-60s)
    late = a_df[a_df.ts_rel >= 45]
    F['late_sol'] = late.groupby('mint')['sol'].sum().reindex(F.index).fillna(0.0)
    F['late_over_early'] = F['late_sol'] / (w0_sol + 1e-4)
    late_buys = late.groupby('mint')['buy_sol'].sum().reindex(F.index).fillna(0.0)
    F['late_buyshare'] = late_buys / (F['late_sol'] + 1e-4)
    
    # Fresh late wallets
    w0_wallets = w0.groupby('mint')['wallet'].apply(set)
    late_wallets = late.groupby('mint')['wallet'].apply(set)
    F['fresh_wallets'] = pd.Series({
        m: len(s - w0_wallets.get(m, set())) for m, s in late_wallets.items()
    }).reindex(F.index).fillna(0)
    
    # 3. Predatory Flow & MEV Detection
    buyers_15 = a_df[(a_df.ts_rel <= 15) & (a_df.is_buy == 1)].groupby('mint')['wallet'].apply(set)
    sellers_60 = a_df[a_df.is_buy == 0].groupby('mint')['wallet'].apply(set)
    F['flip_rate'] = pd.Series({
        m: len(s & sellers_60.get(m, set())) / max(len(s), 1) for m, s in buyers_15.items()
    }).reindex(F.index).fillna(0.0)
    
    F['fee_mismatch_rate'] = a_df.groupby('mint')['mismatch'].mean().reindex(F.index).fillna(0.0)
    F['creator_sell_sol'] = a_df.groupby('mint')['creator_sell_sol'].sum().reindex(F.index).fillna(0.0)
    F['creator_buy_sol'] = a_df.groupby('mint')['creator_buy_sol'].sum().reindex(F.index).fillna(0.0)
    
    # Whale order metrics
    buys = a_df[a_df.is_buy == 1]
    F['buy_max'] = buys.groupby('mint')['sol'].max().reindex(F.index).fillna(0.0)
    F['whale_buys'] = buys[buys.sol >= 5.0].groupby('mint').size().reindex(F.index).fillna(0)
    
    # 4. Price Trajectory & Impact
    p_first = g_amm['price'].first()
    p_last = g_amm['price'].last()
    p_max = g_amm['price'].max()
    p_min = g_amm['price'].min()
    
    F['ret60'] = p_last / p_first.replace(0, np.nan)
    F['runup'] = p_max / p_first.replace(0, np.nan)
    F['at_high'] = p_last / p_max.replace(0, np.nan)
    F['drawdown'] = 1.0 - F['at_high']
    F['price_range'] = (p_max / p_min.replace(0, np.nan)) - 1.0
    
    F['vol'] = a_df.assign(lp=np.log(a_df.price.replace(0, np.nan))).groupby('mint')['lp'].std().reindex(F.index).fillna(0.0)
    
    # Liquidity Depth Metrics
    if 'depth_sol' in a_df.columns:
        d_first = g_amm['depth_sol'].first()
        d_last = g_amm['depth_sol'].last()
        F['depth_sol'] = d_last
        F['depth_ret'] = (d_last / (d_first + 1e-4)) - 1.0
    else:
        F['depth_sol'] = 0.0
        F['depth_ret'] = 0.0
        
    # Kyle's Lambda price impact
    aa = a_df[a_df.price > 0].copy()
    aa['dlp'] = aa.groupby('mint')['price'].transform(lambda s: np.log(s).diff())
    aa['signed'] = np.where(aa.is_buy == 1, aa.sol, -aa.sol)
    aa = aa.dropna(subset=['dlp'])
    if len(aa) > 0:
        num = (aa.signed * aa.dlp).groupby(aa.mint).sum()
        den = (aa.signed ** 2).groupby(aa.mint).sum().replace(0, np.nan)
        F['impact_per_sol'] = (num / den).reindex(F.index).fillna(0.0)
    else:
        F['impact_per_sol'] = 0.0
        
    # 5. Pre-migration Bonding Curve Dynamics
    if len(c_df) > 0:
        c_ok = c_df[c_df.failed == 0]
        c_failed = c_df[c_df.failed != 0]
        
        F['curve_n'] = c_ok.groupby('mint').size().reindex(F.index).fillna(0)
        F['curve_sol'] = c_ok.groupby('mint')['sol'].sum().reindex(F.index).fillna(0.0)
        F['curve_uw'] = c_ok.groupby('mint')['wallet'].nunique().reindex(F.index).fillna(0)
        F['curve_dur'] = -c_ok.groupby('mint')['ts_rel'].first().reindex(F.index).fillna(0.0)
        F['curve_failed'] = c_failed.groupby('mint').size().reindex(F.index).fillna(0)
        F['curve_failed_ratio'] = F['curve_failed'] / (F['curve_n'] + F['curve_failed']).replace(0, np.nan)
        
        p_curve_last = c_ok.groupby('mint')['price'].last()
        F['migration_pop'] = p_first / p_curve_last.replace(0, np.nan)
    else:
        for col in ['curve_n', 'curve_sol', 'curve_uw', 'curve_dur', 'curve_failed', 'curve_failed_ratio', 'migration_pop']:
            F[col] = 0.0
            
    return F.reset_index().rename(columns={'index': 'mint'})
