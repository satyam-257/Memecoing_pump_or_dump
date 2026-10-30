"""Generates the upgraded Two-Stage Decoupled Gatekeeper competition notebook for Kaggle execution."""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.10'
    }
}

cells = []

# Title & Abstract
cells.append(nbf.v4.new_markdown_cell("""# Decoupled Two-Stage Gatekeeper & Multi-Level Expected Utility Engine
### Competition: [Memecoins: Pump or Dump? Predict Pump.fun Launches on Solana](https://www.kaggle.com/competitions/memecoins-pump-or-dump)

---

## Research Abstract & Quantitative Architecture

This notebook implements a decoupled two-stage quantitative decision system for newly migrated Solana memecoin pools (Pump.fun to PumpSwap AMM). At exactly $t = 60\\text{s}$ post-migration, the system evaluates high-frequency microstructure and participant reputation to determine whether to enter a long position and dynamically select an optimal take-profit target $tp \\in [3, 100]$ over a 1-hour holding window, or skip at zero cost ($tp = 0$).

### Core Quantitative Pillars:
1. **Stage 1: Precision Gatekeeper (Rug-Pull & Failure Filter)**:
   - Target: $P(\\max(X) \\ge 3.0)$
   - Out-of-fold probability calibration via Platt scaling (logistic regression on holdout logits).
   - High conviction hurdle $\\tau$ isolating only 2-3 highest-conviction daily opportunities, bypassing the $-0.30$ SOL transaction drain.
2. **Stage 2: Quantile / Expected Utility Optimizer**:
   - Rather than predicting raw peak price, evaluates the expected payoff curve directly across discrete grid $K = (3, 4, 5, 7, 10, 15)$:
     $$\\mathbb{E}[\\text{PnL}(k)] = P(\\max(X) \\ge k) \\cdot (k - 1.30) + (1 - P(\\max(X) \\ge k)) \\cdot \\text{miss\\_payoff}(k)$$
   - Enforces strict probability monotonicity via `np.minimum.accumulate`.
   - Allocates $k^* = \\arg\\max_{k \\in K} \\mathbb{E}[\\text{PnL}(k)]$.
3. **High-Frequency Microstructure & On-Chain Graph Signals**:
   - Gini concentration of early buy volumes & wash trading indicators (`unique_buyers_ratio`).
   - Launch block slot sniping share (slots 0-2).
   - Second-derivative volume acceleration ($d^2 V / dt^2$).
   - Buy/Sell volume and transaction ratios with fast-flipper dump tracking.
4. **Causal Creator History & Participant Reputation**:
   - Strict 1-hour resolution embargo ($mt_j \\le mt_i - 3600$) with zero future leakage.
   - Deployer historical launch counts, median/max peak multiples, and $3\\times/5\\times$ hit rates.
"""))

# Cell 1: Environment Setup & Constants
cells.append(nbf.v4.new_code_cell("""import os, sys, glob, re, time, gc, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

def find_data_dir():
    candidates = [
        '/kaggle/input/competitions/memecoins-pump-or-dump',
        '/kaggle/input/memecoins-pump-or-dump',
        'data',
        '../data'
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return 'data'

DATA = find_data_dir()
print(f"Dataset root resolved to: {DATA}")

WSOL = "So11111111111111111111111111111111111111112"
COST = 0.30
LEVELS = [3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
WIN = 3600
EMBARGO = 3600
HOLD_DAYS = 5
SEEDS = [42, 101, 2024, 7, 777]

def pnl(tp, max_x, final_x, cost=COST):
    tp = np.asarray(tp, float)
    hit = (max_x >= tp) & (tp > 0)
    held = np.maximum(final_x - 1.0 - cost, -1.0)
    return np.where(tp <= 0, 0.0, np.where(hit, tp - 1.0 - cost, held))
"""))

# Cell 2: Creation Metadata Features
cells.append(nbf.v4.new_markdown_cell("""## 1. Creation Metadata & Deployer Funding Profile"""))

cells.append(nbf.v4.new_code_cell("""train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')

if 'finalx' in train.columns:
    train = train.rename(columns={'finalx': 'final_x'})
    
for _d in (train, test):
    _d['day'] = pd.to_datetime(_d.mt, unit='s').dt.strftime('%Y-%m-%d')
    
print(f"Train universe: {len(train):,} tokens over {train.day.nunique()} days")
print(f"Test universe:  {len(test):,} tokens on {test.day.iloc[0]}")

cre_path = f'{DATA}/pumpfun_token_creation.csv'
if not os.path.exists(cre_path):
    cre_path = f'{DATA}/03_sample_pumpfun_token_creation.csv'

cre = pd.read_csv(cre_path).drop_duplicates('mint').copy()
name = cre.name.fillna('').astype(str)
sym = cre.symbol.fillna('').astype(str)
bs = cre.bundle_structure.fillna('').astype(str)
url = cre.url.fillna('').astype(str)

M = pd.DataFrame({'mint': cre.mint, 'creator': cre.creator.fillna('').astype(str)})
M['name_len'] = name.str.len()
M['symbol_len'] = sym.str.len()
M['name_upper'] = name.apply(lambda s: sum(ch.isupper() for ch in s) / max(len(s), 1))
M['name_eq_symbol'] = (name.str.lower().str[:4] == sym.str.lower().str[:4]).astype(np.int8)
M['symbol_is_short'] = (M['symbol_len'] <= 4).astype(np.int8)

# Deployer platforms & custom URL domains
is_ipfs = url.str.contains(r'ipfs\\.io|pinata', regex=True).astype(np.int8)
is_bot = url.str.contains(r'j7tracker|vortexdeployer|rapidlaunch|uxento|sdfgsdfsdf|usepaid|agencypad', regex=True).astype(np.int8)
M['is_ipfs_url'] = is_ipfs
M['is_deployer_bot_url'] = is_bot
M['has_custom_domain'] = ((url != '') & (is_ipfs == 0) & (is_bot == 0)).astype(np.int8)

# Bundle structure analysis
M['bs_ops'] = bs.str.split().str.len().fillna(0)
M['bs_len'] = bs.str.len().fillna(0)
M['bs_buys'] = bs.apply(lambda s: len(re.findall(r'\\dB\\d', s))).astype(np.int16)

amt = bs.apply(lambda s: [int(x) for x in re.findall(r':(\\d{6,})', s)])
M['bs_amt_sum'] = amt.apply(lambda v: float(np.sum(v)) if v else 0.0)
M['bs_amt_max'] = amt.apply(lambda v: float(np.max(v)) if v else 0.0)

for c_ in ['gas_used', 'amount_of_instructions', 'amount_of_lookup_reads', 'amount_of_lookup_writes',
           'creation_ix_index', 'pf_program_index', 'direct_pf_invocation', 'is_cashback_enabled',
           'bundle_size', 'bundled_buys', 'bundled_buys_count', 'dev_balance']:
    if c_ in cre.columns:
        M[c_] = pd.to_numeric(cre[c_], errors='coerce').fillna(0).values

M['dev_balance_log'] = np.log10(np.maximum(M['dev_balance'].values, 0) + 1.0)
M['is_micro_dev_balance'] = (M['dev_balance'].values < 0.05 * 1e9).astype(np.int8)
M['bundle_buy_density'] = M['bundled_buys_count'].values / np.maximum(M['bundle_size'].values, 1)

if 'token_program' in cre.columns:
    M['token2022'] = (~cre.token_program.fillna('').astype(str).str.startswith('Tokenkeg')).astype(np.int8)
else:
    M['token2022'] = 0

M['created_ts'] = pd.to_numeric(cre.created_ts, errors='coerce').fillna(0).values

mig_path = f'{DATA}/pfamm_migrations.csv'
if not os.path.exists(mig_path):
    mig_path = f'{DATA}/02_sample_pfamm_migrations.csv'

if os.path.exists(mig_path):
    mig = pd.read_csv(mig_path, usecols=['mint', 'sol_amount', 'mint_amount', 'pool_migration_fee']).drop_duplicates('mint')
    M = M.merge(mig, on='mint', how='left')
    for c_ in ['sol_amount', 'mint_amount', 'pool_migration_fee']:
        if c_ in M.columns:
            M[c_] = pd.to_numeric(M[c_], errors='coerce').fillna(0).values
    M['sol_migration_ratio'] = M['sol_amount'].values / 1e9
    M['price_at_migration'] = (M['sol_amount'].values / 1e9) / (M['mint_amount'].values / 1e6 + 1e-9)
else:
    M['sol_migration_ratio'] = 0.0
    M['price_at_migration'] = 0.0

print(f"Creation feature table assembled: {M.shape}")
"""))

# Cell 3: High-Frequency Microstructure & Tape Dynamics
cells.append(nbf.v4.new_markdown_cell("""## 2. High-Frequency Microstructure & Order Flow Acceleration"""))

cells.append(nbf.v4.new_code_cell("""CURVE_COLS = ['failed', 'mint', 'ts_rel', 'slot', 'tx_idx', 'direction', 'sol', 'new_price', 'wallet', 'provided_gas_fee']
AMM_COLS = [
    'mint', 'ts_rel', 'slot', 'tx_idx', 'direction', 'base_token', 'sol', 'new_price', 
    'signing_wallet', 'fee_payer', 'provided_gas_fee', 'coin_creator',
    'pool_base_token_reserves_after', 'pool_quote_token_reserves_after'
]

def compute_gini(arr):
    vals = np.sort(arr[arr > 0])
    n = len(vals)
    if n <= 1:
        return 0.0
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * vals)) / (n * np.sum(vals)))

def tape_features(c_df, a_df):
    if 'signing_wallet' in a_df.columns:
        a_df = a_df.rename(columns={'signing_wallet': 'wallet'})
        
    sort_cols = [c for c in ['mint', 'slot', 'tx_idx'] if c in a_df.columns]
    if sort_cols:
        a_df = a_df.sort_values(sort_cols, kind='mergesort')
        
    for df in (c_df, a_df):
        if 'new_price' in df.columns:
            df['price'] = df.new_price.replace(0, np.nan)
        else:
            df['price'] = np.nan

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

    a_df['buy_sol'] = a_df['sol'] * a_df['is_buy']
    a_df['sell_sol'] = a_df['sol'] * (1 - a_df['is_buy'])
    
    if 'fee_payer' in a_df.columns and 'wallet' in a_df.columns:
        a_df['mismatch'] = (a_df['wallet'].astype(str) != a_df['fee_payer'].astype(str)).astype(np.int8)
    else:
        a_df['mismatch'] = 0
        
    if 'coin_creator' in a_df.columns and 'wallet' in a_df.columns:
        a_df['is_creator'] = (a_df['wallet'].astype(str) == a_df['coin_creator'].astype(str)).astype(np.int8)
        a_df['creator_sell_sol'] = a_df['sol'] * a_df['is_creator'] * (1 - a_df['is_buy'])
        a_df['creator_buy_sol'] = a_df['sol'] * a_df['is_creator'] * a_df['is_buy']
    else:
        a_df['is_creator'] = 0
        a_df['creator_sell_sol'] = 0.0
        a_df['creator_buy_sol'] = 0.0

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
    
    F['amm_n'] = g_amm.size()
    F['amm_sol'] = g_amm['sol'].sum()
    F['amm_uw'] = g_amm['wallet'].nunique()
    F['trades_per_wallet'] = F['amm_n'] / F['amm_uw'].replace(0, np.nan)
    
    ws = a_df.groupby(['mint', 'wallet'])['sol'].sum()
    ws_tot = ws.groupby('mint').sum().replace(0, np.nan)
    F['amm_top1'] = ws.groupby('mint').max() / ws_tot
    ws_norm = ws / ws.groupby('mint').transform('sum').replace(0, np.nan)
    F['amm_hhi'] = (ws_norm ** 2).groupby('mint').sum()
    
    # Gini coefficient of buy volumes
    ginis = ws.groupby('mint').apply(lambda s: compute_gini(s.values)).reindex(F.index).fillna(0.0)
    F['amm_gini'] = ginis
    
    # Wash trading detection
    F['unique_buyers_ratio'] = F['amm_uw'] / F['amm_n'].replace(0, np.nan)
    max_tx_w = a_df.groupby(['mint', 'wallet']).size().groupby('mint').max()
    F['wallet_max_tx_share'] = (max_tx_w / F['amm_n'].replace(0, np.nan)).reindex(F.index).fillna(0.0)
    
    # Launch block early slot sniping (slots 0 to 2)
    if 'slot' in a_df.columns:
        min_slot = a_df.groupby('mint')['slot'].min()
        a_df['slot_offset'] = a_df['slot'] - a_df['mint'].map(min_slot)
        early_snipes = a_df[a_df.slot_offset <= 2]
        
        early_sol = early_snipes.groupby('mint')['sol'].sum().reindex(F.index).fillna(0.0)
        early_buys = early_snipes.groupby('mint')['buy_sol'].sum().reindex(F.index).fillna(0.0)
        early_tx = early_snipes.groupby('mint').size().reindex(F.index).fillna(0)
        
        F['slot_snipe_vol_share'] = early_sol / (F['amm_sol'] + 1e-4)
        F['slot_snipe_buy_share'] = early_buys / (early_sol + 1e-4)
        F['slot_snipe_tx_count'] = early_tx
    else:
        F['slot_snipe_vol_share'] = 0.0
        F['slot_snipe_buy_share'] = 0.0
        F['slot_snipe_tx_count'] = 0
        
    # Liquidity curve velocity & acceleration (W0: 0-20s, W1: 20-40s, W2: 40-60s)
    v0 = a_df[a_df.ts_rel < 20].groupby('mint')['sol'].sum().reindex(F.index).fillna(0.0)
    v1 = a_df[(a_df.ts_rel >= 20) & (a_df.ts_rel < 40)].groupby('mint')['sol'].sum().reindex(F.index).fillna(0.0)
    v2 = a_df[a_df.ts_rel >= 40].groupby('mint')['sol'].sum().reindex(F.index).fillna(0.0)
    
    F['d2v_dt2'] = v2 - 2.0 * v1 + v0
    F['rel_flow_accel'] = (v2 - 2.0 * v1 + v0) / (v0 + v1 + v2 + 1e-4)
    
    tot_buy_sol = a_df.groupby('mint')['buy_sol'].sum().reindex(F.index).fillna(0.0)
    tot_sell_sol = a_df.groupby('mint')['sell_sol'].sum().reindex(F.index).fillna(0.0)
    F['buysell_sol_ratio'] = tot_buy_sol / (tot_sell_sol + 1e-4)
    F['net_sol_flow'] = tot_buy_sol - tot_sell_sol
    
    buy_tx = a_df[a_df.is_buy == 1].groupby('mint').size().reindex(F.index).fillna(0)
    sell_tx = a_df[a_df.is_buy == 0].groupby('mint').size().reindex(F.index).fillna(0)
    F['buysell_tx_ratio'] = buy_tx / (sell_tx + 1e-4)
    
    early_dump = a_df[(a_df.ts_rel <= 15) & (a_df.is_buy == 0)].groupby('mint')['sol'].sum().reindex(F.index).fillna(0.0)
    F['early_dump_sol'] = early_dump
    F['early_dump_share'] = early_dump / (tot_sell_sol + 1e-4)

    w0 = a_df[a_df.ts_rel < 15]
    w2 = a_df[a_df.ts_rel >= 30]
    w0_sol = w0.groupby('mint')['sol'].sum().reindex(F.index).fillna(0.0)
    w0_buy = w0.groupby('mint')['buy_sol'].sum().reindex(F.index).fillna(0.0)
    w2_sol = w2.groupby('mint')['sol'].sum().reindex(F.index).fillna(0.0)
    w2_buy = w2.groupby('mint')['buy_sol'].sum().reindex(F.index).fillna(0.0)
    
    F['flow_accel'] = (w2_sol / (w0_sol + 1e-4)) - 1.0
    F['bp_0_15'] = w0_buy / (w0_sol + 1e-4)
    F['bp_30_60'] = w2_buy / (w2_sol + 1e-4)
    F['bp_delta'] = F['bp_30_60'] - F['bp_0_15']
    
    late = a_df[a_df.ts_rel >= 45]
    F['late_sol'] = late.groupby('mint')['sol'].sum().reindex(F.index).fillna(0.0)
    F['late_over_early'] = F['late_sol'] / (w0_sol + 1e-4)
    late_buys = late.groupby('mint')['buy_sol'].sum().reindex(F.index).fillna(0.0)
    F['late_buyshare'] = late_buys / (F['late_sol'] + 1e-4)
    
    w0_wallets = w0.groupby('mint')['wallet'].apply(set)
    late_wallets = late.groupby('mint')['wallet'].apply(set)
    F['fresh_wallets'] = pd.Series({
        m: len(s - w0_wallets.get(m, set())) for m, s in late_wallets.items()
    }).reindex(F.index).fillna(0)
    
    buyers_15 = a_df[(a_df.ts_rel <= 15) & (a_df.is_buy == 1)].groupby('mint')['wallet'].apply(set)
    sellers_60 = a_df[a_df.is_buy == 0].groupby('mint')['wallet'].apply(set)
    F['flip_rate'] = pd.Series({
        m: len(s & sellers_60.get(m, set())) / max(len(s), 1) for m, s in buyers_15.items()
    }).reindex(F.index).fillna(0.0)
    
    F['fee_mismatch_rate'] = a_df.groupby('mint')['mismatch'].mean().reindex(F.index).fillna(0.0)
    F['creator_sell_sol'] = a_df.groupby('mint')['creator_sell_sol'].sum().reindex(F.index).fillna(0.0)
    F['creator_buy_sol'] = a_df.groupby('mint')['creator_buy_sol'].sum().reindex(F.index).fillna(0.0)
    
    buys = a_df[a_df.is_buy == 1]
    F['buy_max'] = buys.groupby('mint')['sol'].max().reindex(F.index).fillna(0.0)
    F['whale_buys'] = buys[buys.sol >= 5.0].groupby('mint').size().reindex(F.index).fillna(0)
    
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
    
    if 'depth_sol' in a_df.columns:
        d_first = g_amm['depth_sol'].first()
        d_last = g_amm['depth_sol'].last()
        F['depth_sol'] = d_last
        F['depth_ret'] = (d_last / (d_first + 1e-4)) - 1.0
    else:
        F['depth_sol'] = 0.0
        F['depth_ret'] = 0.0
        
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

if os.path.exists(f'{DATA}/trades'):
    DAYS = [(d.split('=')[-1], f'{d}/pumpfun_all_swaps.parquet', f'{d}/pumpswap_all_swaps.parquet')
            for d in sorted(glob.glob(f'{DATA}/trades/day=*'))]
else:
    DAYS = [(os.path.basename(f)[4:14], f, f.replace('pumpfun_all_swaps', 'pumpswap_all_swaps'))
            for f in sorted(glob.glob(f'{DATA}/day_*_pumpfun_all_swaps.parquet'))]

frames = []
edges = []
t0 = time.time()

if len(DAYS) > 0:
    for i, (day, cp, ap) in enumerate(DAYS, 1):
        if not os.path.exists(cp) or not os.path.exists(ap):
            continue
        c = pd.read_parquet(cp, columns=[col for col in CURVE_COLS if col in pd.read_parquet(cp).columns[:5].tolist() or True])
        a = pd.read_parquet(ap)
        frames.append(tape_features(c, a))
        
        a_buys = a[a.direction.astype(str).str.upper().str.startswith('B') & (a.ts_rel <= 60)]
        w_col = 'signing_wallet' if 'signing_wallet' in a_buys.columns else 'wallet'
        edges.append(a_buys.groupby(['mint', w_col], as_index=False)['sol'].sum().rename(columns={w_col: 'wallet'}))
        del c, a; gc.collect()
        if i % 10 == 0 or i == len(DAYS):
            print(f"  Processed tape day {i}/{len(DAYS)} ({time.time()-t0:.1f}s)")
            
    TAPE_DF = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    AMM_EDGES = pd.concat(edges, ignore_index=True) if edges else pd.DataFrame()
else:
    c_sample = f'{DATA}/00_sample_pumpfun_swaps.csv'
    a_sample = f'{DATA}/01_sample_pumpswap_swaps.csv'
    if os.path.exists(c_sample) and os.path.exists(a_sample):
        c = pd.read_csv(c_sample)
        a = pd.read_csv(a_sample)
        TAPE_DF = tape_features(c, a)
        a_buys = a[a.ts_rel <= 60].rename(columns={'signing_wallet': 'wallet'})
        AMM_EDGES = a_buys.groupby(['mint', 'wallet'], as_index=False)['sol'].sum()
    else:
        TAPE_DF = pd.DataFrame()
        AMM_EDGES = pd.DataFrame()

print(f"Tape ingestion finished. Shape: {TAPE_DF.shape}")
"""))

# Cell 4: Causal Participant & Deployer History
cells.append(nbf.v4.new_markdown_cell("""## 3. Causal Creator History & Participant Reputation Engine"""))

cells.append(nbf.v4.new_code_cell("""REPUTATION_COLS = [
    'w_known', 'w_hit_max', 'w_hit_mean', 'w_hit_top3', 
    'w_elite_n', 'w_elite_share', 'w_eb_score',
    'w_decay_eb_score', 'w_decay_hits',
    'c_known', 'c_hit_rate', 'c_decay_rate',
    'c_launches', 'c_median_peak', 'c_max_peak',
    'c_hit_3x', 'c_hit_5x', 'c_last_peak', 'c_time_since_last'
]

def build_causal_reputation(tokens_df, amm_edges_df, creator_map=None, target_k=5.0, win=3600, tau_decay=5*86400):
    tok = tokens_df[['mint', 'mt']].copy()
    tok['max_x'] = tokens_df['max_x'].values if 'max_x' in tokens_df.columns else np.nan
    tok = tok.sort_values('mt', kind='stable').reset_index(drop=True)
    
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
    decay_appear = np.zeros(nW, dtype=np.float32)
    decay_hit = np.zeros(nW, dtype=np.float32)
    
    creator_appear = {}
    creator_hit = {}
    creator_decay_ap = {}
    creator_decay_ht = {}
    creator_past_peaks = {}
    creator_last_ts = {}
    
    mts = tok.mt.values
    mxs = tok.max_x.values
    mints = tok.mint.values
    
    out = np.zeros((len(tok), len(REPUTATION_COLS)), dtype=float)
    j = 0
    alpha_prior = 1.0
    beta_prior = 15.0
    
    for i in range(len(tok)):
        cur_t = mts[i]
        while j < len(tok) and mts[j] <= cur_t - win:
            t_res = mts[j]
            if not np.isnan(mxs[j]):
                w = by_mint.get(mints[j])
                if w is not None:
                    wids = w[0]
                    np.add.at(appear, wids, 1)
                    decay_factor = np.exp(-(cur_t - t_res) / max(tau_decay, 1.0))
                    np.add.at(decay_appear, wids, decay_factor)
                    if mxs[j] >= target_k:
                        np.add.at(hit, wids, 1)
                        np.add.at(decay_hit, wids, decay_factor)
                        
                if creator_map is not None:
                    c_wallet = creator_map.get(mints[j])
                    if c_wallet and str(c_wallet) != 'nan' and len(str(c_wallet)) > 0:
                        decay_factor = np.exp(-(cur_t - t_res) / max(tau_decay, 1.0))
                        creator_appear[c_wallet] = creator_appear.get(c_wallet, 0) + 1
                        creator_decay_ap[c_wallet] = creator_decay_ap.get(c_wallet, 0.0) + decay_factor
                        
                        if c_wallet not in creator_past_peaks:
                            creator_past_peaks[c_wallet] = []
                        creator_past_peaks[c_wallet].append(float(mxs[j]))
                        creator_last_ts[c_wallet] = t_res
                        
                        if mxs[j] >= target_k:
                            creator_hit[c_wallet] = creator_hit.get(c_wallet, 0) + 1
                            creator_decay_ht[c_wallet] = creator_decay_ht.get(c_wallet, 0.0) + decay_factor
            j += 1
            
        w = by_mint.get(mints[i])
        if w is not None:
            wids, sols = w
            ap = appear[wids]
            known = ap >= 3
            if known.sum() > 0:
                ht = hit[wids][known]
                ap_k = ap[known]
                rate = ht / ap_k
                eb_rates = (ht + alpha_prior) / (ap_k + alpha_prior + beta_prior)
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
                
        if creator_map is not None:
            c_wallet = creator_map.get(mints[i])
            if c_wallet and str(c_wallet) != 'nan':
                c_ap = creator_appear.get(c_wallet, 0)
                if c_ap > 0:
                    c_ht = creator_hit.get(c_wallet, 0)
                    out[i, 9] = 1.0
                    out[i, 10] = float(c_ht / c_ap)
                    out[i, 11] = float((creator_decay_ht.get(c_wallet, 0.0) + alpha_prior) / 
                                       (creator_decay_ap.get(c_wallet, 0.0) + alpha_prior + beta_prior))
                    
                if c_wallet in creator_past_peaks:
                    peaks = creator_past_peaks[c_wallet]
                    out[i, 12] = len(peaks)
                    out[i, 13] = float(np.median(peaks))
                    out[i, 14] = float(np.max(peaks))
                    out[i, 15] = float(np.mean([p >= 3.0 for p in peaks]))
                    out[i, 16] = float(np.mean([p >= 5.0 for p in peaks]))
                    out[i, 17] = float(peaks[-1])
                    out[i, 18] = float(cur_t - creator_last_ts.get(c_wallet, cur_t))
                    
    rep_df = pd.DataFrame(out, columns=REPUTATION_COLS)
    rep_df['mint'] = mints
    return rep_df

creator_dict = dict(zip(M.mint, M.creator))
tok_universe = pd.concat([
    train[['mint', 'mt', 'max_x']],
    test[['mint', 'mt']].assign(max_x=np.nan)
], ignore_index=True)

REP_DF = build_causal_reputation(tok_universe, AMM_EDGES, creator_map=creator_dict)
print(f"Causal reputation computed across {len(REP_DF):,} tokens.")
"""))

# Cell 5: Assembly
cells.append(nbf.v4.new_code_cell("""def assemble(df):
    d = df.merge(M, on='mint', how='left')
    if len(TAPE_DF) > 0 and 'mint' in TAPE_DF.columns:
        d = d.merge(TAPE_DF, on='mint', how='left')
    d = d.merge(REP_DF, on='mint', how='left')
    d['age_s'] = d.mt - d.created_ts
    return d.sort_values('mt', kind='stable').reset_index(drop=True)

TR = assemble(train)
TE = assemble(test)

exclude = {'mint', 'creator', 'mt', 'day', 'max_x', 'final_x', 'created_ts', 'dt'}
FE = [c for c in TR.columns if c not in exclude and np.issubdtype(TR[c].dtype, np.number)]
print(f"Final feature space: {len(FE)} features.")
"""))

# Cell 6: Decoupled Two-Stage Model
cells.append(nbf.v4.new_markdown_cell("""## 4. Decoupled Two-Stage Gatekeeper & Utility Maximizer Engine"""))

cells.append(nbf.v4.new_code_cell("""LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.04, num_leaves=31, min_child_samples=40,
    subsample=0.80, subsample_freq=1, colsample_bytree=0.70, verbose=-1, n_jobs=-1
)

class TwoStageDecisionEngine:
    def __init__(self, feature_cols, levels=LEVELS, seeds=SEEDS, lgb_params=LGB_PARAMS, cost=COST):
        self.feature_cols = feature_cols
        self.levels = np.array(levels, dtype=float)
        self.seeds = seeds
        self.lgb_params = lgb_params
        self.cost = cost
        self.calibrator = None
        self.gatekeeper_models = []
        self.level_models = {lv: [] for lv in self.levels}

    def fit_gatekeeper(self, train_df):
        X_tr = train_df[self.feature_cols].astype(float).fillna(0).values
        y_tr_3x = (train_df['max_x'].values >= 3.0).astype(int)
        
        self.gatekeeper_models = [
            lgb.LGBMClassifier(random_state=s, **self.lgb_params)
            .fit(X_tr, y_tr_3x)
            for s in self.seeds
        ]
        
        d_trn = sorted(train_df.day.unique())
        cal_days = d_trn[-HOLD_DAYS:]
        trn_cal = train_df[train_df.day.isin(cal_days)].reset_index(drop=True)
        trn_old = train_df[(train_df.day < cal_days[0]) & (train_df.mt <= trn_cal.mt.min() - WIN - EMBARGO)].reset_index(drop=True)
        
        if len(trn_old) >= 500:
            X_old = trn_old[self.feature_cols].astype(float).fillna(0).values
            y_old = (trn_old['max_x'].values >= 3.0).astype(int)
            X_cal = trn_cal[self.feature_cols].astype(float).fillna(0).values
            y_cal = (trn_cal['max_x'].values >= 3.0).astype(int)
            
            p_cal_raw = np.mean([
                lgb.LGBMClassifier(random_state=s, **self.lgb_params)
                .fit(X_old, y_old)
                .predict_proba(X_cal)[:, 1]
                for s in self.seeds
            ], axis=0)
            
            self.calibrator = LogisticRegression()
            self.calibrator.fit(p_cal_raw.reshape(-1, 1), y_cal)
        else:
            self.calibrator = None

    def predict_gatekeeper(self, test_df):
        X_te = test_df[self.feature_cols].astype(float).fillna(0).values
        p_raw = np.mean([m.predict_proba(X_te)[:, 1] for m in self.gatekeeper_models], axis=0)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        return p_raw

    def fit_levels(self, train_df):
        X_tr = train_df[self.feature_cols].astype(float).fillna(0).values
        mxs_tr = train_df['max_x'].values
        for lv in self.levels:
            y_lv = (mxs_tr >= lv).astype(int)
            self.level_models[lv] = [
                lgb.LGBMClassifier(random_state=s, **self.lgb_params)
                .fit(X_tr, y_lv)
                for s in self.seeds
            ]

    def predict_levels(self, test_df):
        X_te = test_df[self.feature_cols].astype(float).fillna(0).values
        probs = np.zeros((len(test_df), len(self.levels)), dtype=float)
        for j, lv in enumerate(self.levels):
            probs[:, j] = np.mean([m.predict_proba(X_te)[:, 1] for m in self.level_models[lv]], axis=0)
        return np.minimum.accumulate(probs, axis=1)

    def compute_miss_payoffs(self, train_df):
        mxs = train_df['max_x'].values
        fxs = train_df['final_x'].values if 'final_x' in train_df.columns else train_df['finalx'].values
        held = np.maximum(fxs - 1.0 - self.cost, -1.0)
        held_by_level = []
        for lv in self.levels:
            miss = mxs < lv
            held_by_level.append(float(held[miss].mean()) if miss.any() else -0.95)
        return np.array(held_by_level, dtype=float)

    def calculate_expected_values(self, probs_monotonic, held_by_level):
        win_pnl = self.levels[None, :] - 1.0 - self.cost
        held_row = held_by_level[None, :]
        return probs_monotonic * win_pnl + (1.0 - probs_monotonic) * held_row

    def decide(self, p_gatekeeper, ev_matrix, top_n_daily=2, min_prob_hurdle=0.20, min_ev_margin=0.0):
        n = len(p_gatekeeper)
        final_tp = np.zeros(n, dtype=float)
        best_lv_idx = np.argmax(ev_matrix, axis=1)
        best_ev = ev_matrix[np.arange(n), best_lv_idx]
        best_tp = self.levels[best_lv_idx]
        
        valid_mask = (p_gatekeeper >= min_prob_hurdle) & (best_ev >= min_ev_margin)
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0 and top_n_daily > 0:
            scores = p_gatekeeper[valid_indices]
            order = np.argsort(-scores)
            selected = valid_indices[order[:top_n_daily]]
            final_tp[selected] = best_tp[selected]
        elif (final_tp > 0).sum() == 0 and top_n_daily > 0:
            order = np.argsort(-best_ev)
            top_cand = [idx for idx in order[:top_n_daily] if best_ev[idx] >= -0.20]
            if top_cand:
                final_tp[top_cand] = best_tp[top_cand]
        return final_tp, best_ev
"""))

# Cell 7: Full Production Fit & Submission
cells.append(nbf.v4.new_markdown_cell("""## 5. Full Production Fit & Validated Submission Generation"""))

cells.append(nbf.v4.new_code_cell("""print("Training production Two-Stage Decoupled Engine on full historical universe...")
engine = TwoStageDecisionEngine(feature_cols=FE)
engine.fit_gatekeeper(TR)
engine.fit_levels(TR)

print("Generating calibrated Stage 1 probabilities & Stage 2 expected values...")
test_p_gate = engine.predict_gatekeeper(TE)
test_p_levels = engine.predict_levels(TE)
miss_payoffs = engine.compute_miss_payoffs(TR)
test_ev = engine.calculate_expected_values(test_p_levels, miss_payoffs)

test_tp, best_ev = engine.decide(test_p_gate, test_ev, top_n_daily=2, min_prob_hurdle=0.20, min_ev_margin=0.0)

sub = pd.DataFrame({'mint': TE['mint'], 'tp': test_tp})

sample_path = f'{DATA}/sample_submission.csv'
if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    sub = sample_sub[['mint']].merge(sub, on='mint', how='left')
    sub['tp'] = sub['tp'].fillna(0.0).astype(float)
    assert len(sub) == len(sample_sub), "Row count mismatch"
    assert sub['mint'].equals(sample_sub['mint']), "Mint alignment mismatch"

valid_mask = (sub['tp'] == 0.0) | ((sub['tp'] >= 3.0) & (sub['tp'] <= 100.0))
assert valid_mask.all(), "Invalid take-profit values detected outside allowed bounds."

out_paths = ['submission.csv']
if os.path.exists('/kaggle/working'):
    out_paths.append('/kaggle/working/submission.csv')

for out_p in out_paths:
    sub.to_csv(out_p, index=False)
    print(f"Exported: {out_p}")

num_buys = int((sub['tp'] > 0).sum())
print(f"\\nVerification Complete:")
print(f"Total test tokens: {len(sub)}")
print(f"Tokens allocated long entries: {num_buys} ({num_buys / len(sub) * 100:.2f}%)")
print(f"Tokens skipped at zero cost (tp = 0): {len(sub) - num_buys}")
print("Take-profit level allocation counts:")
print(sub[sub.tp > 0]['tp'].value_counts().sort_index())
if num_buys > 0:
    display(sub[sub.tp > 0].head(10))
"""))

nb['cells'] = cells

with open('notebooks/submission_pipeline.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully generated upgraded notebooks/submission_pipeline.ipynb with {len(cells)} cells.")
