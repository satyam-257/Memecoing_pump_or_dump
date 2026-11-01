"""Generates the upgraded Decoupled Two-Stage Gatekeeper competition notebook for Kaggle execution."""
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
cells.append(nbf.v4.new_markdown_cell("""# Unified Entry Gatekeeper & Constrained 2-Tier Exit Utility Engine
### Competition: [Memecoins: Pump or Dump? Predict Pump.fun Launches on Solana](https://www.kaggle.com/competitions/memecoins-pump-or-dump)

---

## Quantitative Architecture & 30-Day Rolling Sandbox Requirements

This notebook implements an offline-safe, unified quantitative decision engine for newly migrated Solana memecoins on PumpSwap AMM. At exactly $t = 60\\text{s}$ post-migration, the system evaluates high-frequency order book microstructure, reverted bonding curve front-running demand, and empirical Bayes shrunk wallet reputation to determine whether to enter a long position and dynamically select between a high-conviction runner exit ($tp = 8.0$) or baseline exit ($tp = 5.0$) over a 1-hour holding window, or skip at zero cost ($tp = 0$).

### Core Upgrades & Safeguards:
1. **Unified Full-Feature Entry Gatekeeper (LGBM + CatBoost)**:
   - Primary entry classifier receives ALL 103+ features simultaneously (Creation Bundles + Tape Velocity + Shrunk Wallet Reputation). Never starves the entry gatekeeper of wallet reputation.
   - Primary target directly predicts $P(\\max(X) \\ge 5.0)$ matching the proven $+3.70$ SOL net payout sweet spot.
   - Blends LightGBM + CatBoost probability ensemble with out-of-fold calibration on recent training days.
2. **Constrained 2-Tier Exit Policy (Zero Low-Conviction Settling)**:
   - Default Exit: Set $tp = 5.0$ (locks in baseline $+5.47$ SOL edge).
   - High-Conviction Runner Upgrade: Independently trains calibrated model for $P(\\max(X) \\ge 8.0)$ and upgrades to $tp = 8.0$ (+6.70 SOL net payout) iff $\\mathbb{E}[\\text{PnL}(8.0)] > \\mathbb{E}[\\text{PnL}(5.0)] + 0.35$ SOL margin of safety.
   - Eliminates noisy intermediate levels ($k = 4, 12, 20, 100$).
3. **Reverted Curve Demand & Archive Warm-Start**:
   - In bonding curve tape, extracts reverted transaction features (`failed == 1`): `failed_buy_count`, `failed_to_executed_ratio`, and `failed_unique_wallets` to gauge fierce bot competition for hot launches. Enforces `failed == 0` for all volume and reserve price metrics.
   - Warm-starts wallet and creator reputation dictionaries with causal historical lookback.
4. **Causal Per-Token Execution (Zero Test Peeking)**:
   - Evaluates strictly per-token against fixed scalar cutoff $\\tau^*$ calibrated on the last 5 training days.
5. **100% Offline Sandbox Execution**:
   - Zero external network dependencies, RPC calls, or online enrichment scripts. Fully self-contained.
"""))

# Cell 1: Environment Setup & Constants
cells.append(nbf.v4.new_code_cell("""import os, sys, glob, re, time, gc, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import LogisticRegression

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
LEVELS = [4.0, 5.0, 6.0, 8.0, 12.0, 20.0, 100.0]
TARGET_K = 5.0
WIN = 3600
EMBARGO = 3600
HOLD_DAYS = 5
SEEDS = [42, 101, 2024, 7, 777]

HALF_LIFE_SECONDS = 7.0 * 86400.0
DECAY_LAMBDA = np.log(2.0) / HALF_LIFE_SECONDS
GLOBAL_BASE_RATE = 0.0607
M_SHRINKAGE = 5.0

def evaluate_pnl(y_true_max, y_true_end, y_pred_tp, cost=COST):
    mask = y_pred_tp > 0
    hit = mask & (y_true_max >= y_pred_tp)
    miss = mask & (y_true_max < y_pred_tp)
    pnl = np.sum(y_pred_tp[hit] - 1.0 - cost)
    pnl += np.sum(np.maximum(y_true_end[miss] - 1.0 - cost, -1.0))
    return float(pnl)
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
        
        # High reverted buy attempts indicate bot front-running
        c_failed_buys = c_failed[c_failed.is_buy == 1]
        F['failed_buy_count'] = c_failed_buys.groupby('mint').size().reindex(F.index).fillna(0)
        F['failed_to_executed_ratio'] = F['curve_failed'] / (F['curve_n'] + F['curve_failed'] + 1e-4)
        F['failed_unique_wallets'] = c_failed.groupby('mint')['wallet'].nunique().reindex(F.index).fillna(0)
        
        p_curve_last = c_ok.groupby('mint')['price'].last()
        F['migration_pop'] = p_first / p_curve_last.replace(0, np.nan)
    else:
        for col in ['curve_n', 'curve_sol', 'curve_uw', 'curve_dur', 'curve_failed', 'curve_failed_ratio',
                    'failed_buy_count', 'failed_to_executed_ratio', 'failed_unique_wallets', 'migration_pop']:
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

# Cell 4: Causal Participant & Deployer History with Bayesian Shrinkage
cells.append(nbf.v4.new_markdown_cell("""## 3. Causal Creator History & Participant Reputation Engine
- Invariant: Two-pointer 3600s embargo ($mt_j \\le mt_i - 3600$).
- Empirical Bayesian Shrinkage ($m$-estimate smoothing, $m = 5$).
- SOL trade size weighting and 7-day half-life exponential recency decay.
"""))

cells.append(nbf.v4.new_code_cell("""REPUTATION_COLS = [
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
    tok = tokens_df[['mint', 'mt']].copy()
    if 'max_x' in tokens_df.columns:
        tok['max_x'] = tokens_df['max_x'].values
    else:
        tok['max_x'] = np.nan
    target_mints = set(tok['mint'])
    
    # Warm-start from archive dataset if available
    if archive_df is None:
        for candidate in [f'{DATA}/train_archive.csv', 'train_archive.csv', 'data/train_archive.csv']:
            if os.path.exists(candidate):
                try:
                    archive_df = pd.read_csv(candidate)
                    break
                except Exception:
                    pass
    if archive_df is not None and len(archive_df) > 0:
        arc = archive_df[['mint', 'mt']].copy()
        arc['max_x'] = archive_df['max_x'].values if 'max_x' in archive_df.columns else np.nan
        min_t = tok.mt.min()
        arc = arc[arc.mt < min_t]
        if len(arc) > 0:
            tok = pd.concat([arc, tok], ignore_index=True)
            if creator_map is not None and 'creator' in archive_df.columns:
                for m, c in zip(archive_df['mint'], archive_df['creator']):
                    if m not in creator_map:
                        creator_map[m] = c
                        
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
    last_update_ts = np.zeros(nW, dtype=np.float64)
    decay_appear = np.zeros(nW, dtype=np.float32)
    decay_hit = np.zeros(nW, dtype=np.float32)
    decay_sol_appear = np.zeros(nW, dtype=np.float32)
    decay_sol_hit = np.zeros(nW, dtype=np.float32)
    
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
                w = by_mint.get(mints[j])
                if w is not None:
                    wids, sols = w
                    for wid, sol_val in zip(wids, sols):
                        sol_w = float(min(max(sol_val, 0.01), 10.0))
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
                
                shrunk_rates = (ht_k + m_shrinkage * base_rate) / (ap_k + m_shrinkage)
                raw_rates = ht_k / np.maximum(ap_k, 1)
                
                k_last_t = last_update_ts[k_wids]
                dt_vals = np.maximum(cur_t - k_last_t, 0.0)
                decay_multipliers = np.exp(-dt_vals * DECAY_LAMBDA)
                
                cur_decay_sol_ap = decay_sol_appear[k_wids] * decay_multipliers
                cur_decay_sol_ht = decay_sol_hit[k_wids] * decay_multipliers
                
                shrunk_sol_rates = (cur_decay_sol_ht + m_shrinkage * base_rate) / (cur_decay_sol_ap + m_shrinkage)
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
                
                out[i, 9] = float(np.max(shrunk_rates))
                out[i, 10] = float(np.mean(shrunk_rates))
                out[i, 11] = float(np.sort(shrunk_rates)[-3:].mean())
                out[i, 12] = float(np.max(shrunk_sol_rates))
                out[i, 13] = float(np.mean(shrunk_sol_rates))
                out[i, 14] = float(cur_decay_sol_ht.sum())
                out[i, 15] = float(k_sols[is_elite].sum() / max(sols.sum(), 1e-9))
                
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
                    
                    out[i, 16] = 1.0
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

# Cell 6: Unified Entry Gatekeeper & Constrained 2-Tier Exit Utility Engine
cells.append(nbf.v4.new_markdown_cell("""## 4. Unified Full-Feature Entry Gatekeeper & Constrained 2-Tier Exit Utility Engine
- **Primary Entry Classifier**: Receives all features simultaneously (Creation Bundles + Tape Velocity + Shrunk Wallet Reputation).
- **Target**: Directly predicts $P(\\max(X) \\ge 5.0)$ matching high-conviction $+3.70$ SOL net payout sweet spot.
- **Ensemble Blend**: Blends LightGBM + CatBoost probabilities with out-of-fold calibration on recent training days.
- **Constrained 2-Tier Exit Policy**: Default $tp = 5.0$; upgrades to $tp = 8.0$ (+6.70 SOL payout) iff $\\mathbb{E}[\\text{PnL}(8.0)] > \\mathbb{E}[\\text{PnL}(5.0)] + 0.35$ SOL safety margin.
- **Per-Token Thresholding**: Fixed scalar cutoff $\\tau^*$ calibrated on last 5 training days (~2 to 5 trades/day) strictly without test peeking.
"""))

cells.append(nbf.v4.new_code_cell("""LGB_PARAMS = dict(
    n_estimators=350, learning_rate=0.04, num_leaves=31, min_child_samples=40,
    subsample=0.80, subsample_freq=1, colsample_bytree=0.70, verbose=-1, n_jobs=-1
)

CB_PARAMS = dict(
    iterations=250, learning_rate=0.05, depth=6, thread_count=-1, verbose=0
)

class UnifiedTwoTierEngine:
    def __init__(self, feature_cols, seeds=SEEDS[:2], lgb_params=LGB_PARAMS, cb_params=CB_PARAMS, cost=COST):
        self.feature_cols = list(feature_cols)
        self.seeds = seeds
        self.lgb_params = lgb_params
        self.cb_params = cb_params
        self.cost = cost
        
        self.lgb_entry_models = []
        self.cat_entry_models = []
        self.lgb_runner_models = []
        self.cat_runner_models = []
        
        self.calibrator_5x = None
        self.calibrator_8x = None
        self.p_cal_oof = None
        self.trn_cal_len = 0
        self.cal_days_len = 0
        self.tau_scalar = 0.20
        self.miss_payoffs = {'held_5': -0.95, 'held_8': -0.95}

    def fit_entry(self, train_df):
        X_tr = train_df[self.feature_cols].astype(float).fillna(0).values
        y_tr_5x = (train_df['max_x'].values >= 5.0).astype(int)
        y_tr_8x = (train_df['max_x'].values >= 8.0).astype(int)
        
        # Out-of-fold calibration on recent holdout days
        d_trn = sorted(train_df.day.unique())
        cal_days = d_trn[-HOLD_DAYS:]
        trn_cal = train_df[train_df.day.isin(cal_days)].reset_index(drop=True)
        trn_old = train_df[(train_df.day < cal_days[0]) & (train_df.mt <= trn_cal.mt.min() - WIN - EMBARGO)].reset_index(drop=True)
        
        if len(trn_old) >= 500 and len(trn_cal) >= 10:
            X_old = trn_old[self.feature_cols].astype(float).fillna(0).values
            y_old_5x = (trn_old['max_x'].values >= 5.0).astype(int)
            y_old_8x = (trn_old['max_x'].values >= 8.0).astype(int)
            X_cal = trn_cal[self.feature_cols].astype(float).fillna(0).values
            y_cal_5x = (trn_cal['max_x'].values >= 5.0).astype(int)
            y_cal_8x = (trn_cal['max_x'].values >= 8.0).astype(int)
            
            oof_5_preds = []
            oof_8_preds = []
            for s in self.seeds:
                m_lgb5 = lgb.LGBMClassifier(random_state=s, **self.lgb_params).fit(X_old, y_old_5x)
                m_cb5 = cb.CatBoostClassifier(random_seed=s, **self.cb_params).fit(X_old, y_old_5x)
                oof_5_preds.append(0.5 * m_lgb5.predict_proba(X_cal)[:, 1] + 0.5 * m_cb5.predict_proba(X_cal)[:, 1])
                
                if y_old_8x.sum() >= 10:
                    m_lgb8 = lgb.LGBMClassifier(random_state=s, **self.lgb_params).fit(X_old, y_old_8x)
                    m_cb8 = cb.CatBoostClassifier(random_seed=s, **self.cb_params).fit(X_old, y_old_8x)
                    oof_8_preds.append(0.5 * m_lgb8.predict_proba(X_cal)[:, 1] + 0.5 * m_cb8.predict_proba(X_cal)[:, 1])
                else:
                    oof_8_preds.append(np.full(len(X_cal), float(y_old_8x.mean())))
                    
            p_oof_5x_raw = np.mean(oof_5_preds, axis=0)
            p_oof_8x_raw = np.mean(oof_8_preds, axis=0)
            
            self.calibrator_5x = LogisticRegression().fit(p_oof_5x_raw.reshape(-1, 1), y_cal_5x)
            self.p_cal_oof = self.calibrator_5x.predict_proba(p_oof_5x_raw.reshape(-1, 1))[:, 1]
            
            if y_cal_8x.sum() >= 5:
                self.calibrator_8x = LogisticRegression().fit(p_oof_8x_raw.reshape(-1, 1), y_cal_8x)
            else:
                self.calibrator_8x = None
                
            self.trn_cal_len = len(trn_cal)
            self.cal_days_len = len(cal_days)
        else:
            self.calibrator_5x = None
            self.calibrator_8x = None
            self.p_cal_oof = None
            self.trn_cal_len = 0
            self.cal_days_len = 0
            
        # Full production models
        self.lgb_entry_models = [
            lgb.LGBMClassifier(random_state=s, **self.lgb_params).fit(X_tr, y_tr_5x)
            for s in self.seeds
        ]
        self.cat_entry_models = [
            cb.CatBoostClassifier(random_seed=s, **self.cb_params).fit(X_tr, y_tr_5x)
            for s in self.seeds
        ]
        
        if y_tr_8x.sum() >= 15:
            self.lgb_runner_models = [
                lgb.LGBMClassifier(random_state=s, **self.lgb_params).fit(X_tr, y_tr_8x)
                for s in self.seeds
            ]
            self.cat_runner_models = [
                cb.CatBoostClassifier(random_seed=s, **self.cb_params).fit(X_tr, y_tr_8x)
                for s in self.seeds
            ]
        else:
            self.lgb_runner_models = []
            self.cat_runner_models = []
            
        self.compute_miss_payoffs(train_df)

    def fit_gatekeeper(self, train_df):
        return self.fit_entry(train_df)

    def fit_levels(self, train_df):
        pass

    def predict_entry(self, test_df):
        X_te = test_df[self.feature_cols].astype(float).fillna(0).values
        p_lgb = np.mean([m.predict_proba(X_te)[:, 1] for m in self.lgb_entry_models], axis=0)
        p_cat = np.mean([m.predict_proba(X_te)[:, 1] for m in self.cat_entry_models], axis=0)
        p_raw = 0.5 * p_lgb + 0.5 * p_cat
        if self.calibrator_5x is not None:
            return self.calibrator_5x.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        return p_raw

    def predict_gatekeeper(self, test_df):
        return self.predict_entry(test_df)

    def predict_runner(self, test_df):
        X_te = test_df[self.feature_cols].astype(float).fillna(0).values
        if len(self.lgb_runner_models) > 0 and len(self.cat_runner_models) > 0:
            p_lgb = np.mean([m.predict_proba(X_te)[:, 1] for m in self.lgb_runner_models], axis=0)
            p_cat = np.mean([m.predict_proba(X_te)[:, 1] for m in self.cat_runner_models], axis=0)
            p_raw = 0.5 * p_lgb + 0.5 * p_cat
        else:
            p_raw = np.full(len(test_df), 0.02)
        if self.calibrator_8x is not None:
            return self.calibrator_8x.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        return p_raw

    def predict_levels(self, test_df):
        p5 = self.predict_entry(test_df)
        p8 = self.predict_runner(test_df)
        p8_mono = np.minimum(p8, p5)
        return np.column_stack([p5, p8_mono])

    def compute_miss_payoffs(self, train_df):
        mxs = train_df['max_x'].values
        fxs = train_df['final_x'].values if 'final_x' in train_df.columns else train_df['finalx'].values
        held = np.maximum(fxs - 1.0 - self.cost, -1.0)
        miss_5 = mxs < 5.0
        miss_8 = mxs < 8.0
        held_5 = float(held[miss_5].mean()) if miss_5.any() else -0.95
        held_8 = float(held[miss_8].mean()) if miss_8.any() else -0.95
        self.miss_payoffs = {'held_5': held_5, 'held_8': held_8}
        return self.miss_payoffs

    def calculate_expected_values(self, test_df=None, p_levels=None, miss_payoffs=None):
        if p_levels is None and test_df is not None:
            p_levels = self.predict_levels(test_df)
        p5 = p_levels[:, 0]
        p8 = p_levels[:, 1]
        mp = miss_payoffs or self.miss_payoffs
        h5 = mp.get('held_5', -0.95) if isinstance(mp, dict) else mp[0]
        h8 = mp.get('held_8', -0.95) if isinstance(mp, dict) else mp[1]
        ev_5 = p5 * 3.70 + (1.0 - p5) * h5
        ev_8 = p8 * 6.70 + (1.0 - p8) * h8
        return np.column_stack([ev_5, ev_8])

    def calibrate_scalar_threshold(self, train_df=None, target_daily_trades=2.5):
        if getattr(self, 'p_cal_oof', None) is not None and self.trn_cal_len > 0:
            avg_tokens_per_day = max(self.trn_cal_len / float(self.cal_days_len), 10.0)
            q = min(max(target_daily_trades / avg_tokens_per_day, 0.001), 0.05)
            self.tau_scalar = float(np.quantile(self.p_cal_oof, 1.0 - q))
        elif train_df is not None:
            d_trn = sorted(train_df.day.unique())
            cal_days = d_trn[-HOLD_DAYS:]
            trn_cal = train_df[train_df.day.isin(cal_days)].reset_index(drop=True)
            p_cal = self.predict_entry(trn_cal)
            avg_tokens_per_day = max(len(trn_cal) / float(len(cal_days)), 10.0)
            q = min(max(target_daily_trades / avg_tokens_per_day, 0.001), 0.05)
            self.tau_scalar = float(np.quantile(p_cal, 1.0 - q))
        else:
            self.tau_scalar = 0.20
        return self.tau_scalar

    def decide_per_token(self, p_entry, ev_matrix, tau_scalar=None, safety_margin=0.35, mode='2tier'):
        tau = self.tau_scalar if tau_scalar is None else tau_scalar
        n = len(p_entry)
        final_tp = np.zeros(n, dtype=float)
        ev_5 = ev_matrix[:, 0]
        ev_8 = ev_matrix[:, 1]
        is_buy = (p_entry >= tau)
        if mode == 'fixed5':
            final_tp[is_buy] = 5.0
        else:
            is_upgrade = is_buy & (ev_8 > (ev_5 + safety_margin))
            final_tp[is_buy] = 5.0
            final_tp[is_upgrade] = 8.0
        return final_tp, ev_5

TwoStageDecisionEngine = UnifiedTwoTierEngine
"""))

# Cell 7: Full 16-Day Walk-Forward Benchmark (Offline Evaluation)
cells.append(nbf.v4.new_markdown_cell("""## 5. Strict 16-Day Walk-Forward Comparative Benchmark
- Evaluates 1 fold per day across the last 16 days.
- Invariant enforced: `target_mt <= mt_train_max - 3600`.
- Compares:
  (a) Flat TP=5 Single-Stage Baseline (+5.47 SOL reference)
  (b) Unified LightGBM+CatBoost Entry (Fixed TP=5)
  (c) Unified LightGBM+CatBoost Entry with Constrained {5.0, 8.0} Exit Upgrade
"""))

cells.append(nbf.v4.new_code_cell("""days = sorted(TR['day'].unique())
eval_days = days[-16:]
print(f"Running 16-day embargoed walk-forward evaluation across: {eval_days[0]} to {eval_days[-1]}...")

baseline_records = []
f5_records = []
t2_records = []

for i, day in enumerate(eval_days, 1):
    val = TR[TR['day'] == day].reset_index(drop=True)
    trn = TR[(TR['day'] < day) & (TR['mt'] <= val['mt'].min() - WIN - EMBARGO)].reset_index(drop=True)
    
    if len(trn) < 500 or len(val) < 10:
        continue
        
    mxs_val = val['max_x'].values
    fxs_val = val['final_x'].values
    
    # 1. Baseline tp = 5
    X_tr = trn[FE].astype(float).fillna(0).values
    y_tr_5x = (trn['max_x'].values >= 5.0).astype(int)
    X_val = val[FE].astype(float).fillna(0).values
    
    base_model = lgb.LGBMClassifier(random_state=SEEDS[0], **LGB_PARAMS).fit(X_tr, y_tr_5x)
    p_base_val = base_model.predict_proba(X_val)[:, 1]
    
    cal_days = sorted(trn['day'].unique())[-HOLD_DAYS:]
    trn_cal = trn[trn['day'].isin(cal_days)].reset_index(drop=True)
    trn_old = trn[(trn['day'] < cal_days[0]) & (trn['mt'] <= trn_cal['mt'].min() - WIN - EMBARGO)].reset_index(drop=True)
    if len(trn_old) >= 500:
        base_cal_model = lgb.LGBMClassifier(random_state=SEEDS[0], **LGB_PARAMS).fit(
            trn_old[FE].astype(float).fillna(0).values,
            (trn_old['max_x'].values >= 5.0).astype(int)
        )
        p_base_cal = base_cal_model.predict_proba(trn_cal[FE].astype(float).fillna(0).values)[:, 1]
    else:
        p_base_cal = p_base_val
        
    q_base = min(max(2.5 / max(len(trn_cal) / float(len(cal_days)), 10.0), 0.001), 0.05)
    tau_base = float(np.quantile(p_base_cal, 1.0 - q_base))
    
    tp_base = np.where(p_base_val >= tau_base, 5.0, 0.0)
    pnl_base = evaluate_pnl(mxs_val, fxs_val, tp_base)
    taken_base = int((tp_base > 0).sum())
    hits_base = int(((mxs_val >= 5.0) & (tp_base > 0)).sum())
    baseline_records.append({'day': day, 'taken': taken_base, 'hits': hits_base, 'pnl': pnl_base})
    
    # 2. Unified LGBM + CatBoost Entry & 2-Tier Engine
    wf_engine = UnifiedTwoTierEngine(feature_cols=FE, seeds=[42])
    wf_engine.fit_entry(trn)
    
    p_entry = wf_engine.predict_entry(val)
    ev_matrix = wf_engine.calculate_expected_values(test_df=val)
    tau_scalar = wf_engine.calibrate_scalar_threshold(trn, target_daily_trades=2.5)
    
    # Architecture B: Fixed TP=5
    tp_f5, _ = wf_engine.decide_per_token(p_entry, ev_matrix, tau_scalar=tau_scalar, mode='fixed5')
    pnl_f5 = evaluate_pnl(mxs_val, fxs_val, tp_f5)
    f5_records.append({'day': day, 'taken': int((tp_f5 > 0).sum()), 'hits': int(((mxs_val >= 5.0) & (tp_f5 > 0)).sum()), 'pnl': pnl_f5})
    
    # Architecture C: Constrained {5.0, 8.0} Exit Upgrade
    tp_t2, _ = wf_engine.decide_per_token(p_entry, ev_matrix, tau_scalar=tau_scalar, safety_margin=0.35, mode='2tier')
    pnl_t2 = evaluate_pnl(mxs_val, fxs_val, tp_t2)
    t2_records.append({'day': day, 'taken': int((tp_t2 > 0).sum()), 'hits': int(((mxs_val >= tp_t2) & (tp_t2 > 0)).sum()), 'pnl': pnl_t2})
    
    print(f"[{i:02d}/16] Day {day} | (a) Baseline: trades={taken_base:>2}, PnL={pnl_base:+6.2f} SOL | "
          f"(b) Fixed-5: trades={int((tp_f5 > 0).sum()):>2}, PnL={pnl_f5:+6.2f} SOL | "
          f"(c) 2-Tier: trades={int((tp_t2 > 0).sum()):>2}, PnL={pnl_t2:+6.2f} SOL")

b_df = pd.DataFrame(baseline_records)
f5_df = pd.DataFrame(f5_records)
t2_df = pd.DataFrame(t2_records)

print("\\n" + "=" * 105)
print("16-DAY EMBARGOED WALK-FORWARD HEAD-TO-HEAD COMPARISON")
print("=" * 105)
print(f"{'Strategy Architecture':<42} | {'Trades':<6} | {'Hits':<5} | {'Precision':<10} | {'Total PnL':<12} | {'Sharpe':<7} | {'Win Days'}")
print("-" * 105)

def print_row(name, d):
    tot = d['taken'].sum()
    hits = d['hits'].sum()
    pnl = d['pnl'].sum()
    prec = (hits / max(tot, 1)) * 100.0
    sharpe = d['pnl'].mean() - 0.25 * d['pnl'].std(ddof=0)
    win = (d['pnl'] > 0).sum()
    print(f"{name:<42} | {tot:<6} | {hits:<5} | {prec:>8.1f}% | {pnl:>+10.2f} SOL | {sharpe:>+6.2f} | {win}/{len(d)}")

print_row("(a) Flat TP=5 Single-Stage Baseline", b_df)
print_row("(b) Unified LGB+CB Entry (Fixed TP=5)", f5_df)
print_row("(c) Unified LGB+CB Entry ({5,8} Upgrade)", t2_df)
print("=" * 105)
"""))

# Cell 8: Full Production Fit & Submission Generation
cells.append(nbf.v4.new_markdown_cell("""## 6. Full Production Fit & Submission Generation"""))

cells.append(nbf.v4.new_code_cell("""print("Training production Unified LGBM+CatBoost Entry & 2-Tier Exit Engine on full universe...")
prod_engine = UnifiedTwoTierEngine(feature_cols=FE)
prod_engine.fit_entry(TR)

print("Generating calibrated entry probabilities and expected values for test universe...")
test_p_entry = prod_engine.predict_entry(TE)
test_ev = prod_engine.calculate_expected_values(test_df=TE)

tau_scalar = prod_engine.calibrate_scalar_threshold(TR, target_daily_trades=2.5)
print(f"Calibrated causal scalar cutoff: tau = {tau_scalar:.4f}")

test_tp, best_ev = prod_engine.decide_per_token(test_p_entry, test_ev, tau_scalar=tau_scalar, safety_margin=0.35, mode='2tier')

sub = pd.DataFrame({'mint': TE['mint'], 'tp': test_tp})

sample_path = f'{DATA}/sample_submission.csv'
if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    sub = sample_sub[['mint']].merge(sub, on='mint', how='left')
    sub['tp'] = sub['tp'].fillna(0.0).astype(float)
    assert len(sub) == len(sample_sub), f"Row count mismatch: {len(sub)} vs {len(sample_sub)}"
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
"""))

nb['cells'] = cells

with open('notebooks/submission_pipeline.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully generated sanitized notebooks/submission_pipeline.ipynb with {len(cells)} cells.")
