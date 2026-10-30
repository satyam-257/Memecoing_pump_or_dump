"""Generates the multi-level Expected Value competition notebook for Kaggle execution."""
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
cells.append(nbf.v4.new_markdown_cell("""# Quantitative Microstructure & Dynamic Multi-Level Take-Profit Engine
### Competition: [Memecoins: Pump or Dump? Predict Pump.fun Launches on Solana](https://www.kaggle.com/competitions/memecoins-pump-or-dump)

---

## Research Abstract & Quantitative Architecture

This notebook implements an advanced quantitative trading system for newly migrated Solana memecoin pools (Pump.fun to PumpSwap AMM). At exactly $t = 60\\text{s}$ post-migration, the system evaluates high-frequency microstructure and participant reputation to determine whether to enter a long position and dynamically select an optimal take-profit target $tp \\in [3, 100]$ over a 1-hour holding window, or skip at zero cost ($tp = 0$).

### Core Quantitative Pillars:
1. **WSOL Base Token Inversion Correction**: On PumpSwap pools where WSOL is the base token, `'B'` swaps dump the memecoin for SOL. This pipeline corrects the trade direction using pool pair tokens.
2. **Three-Window Order Flow Dynamics**: Decomposes the first minute into $W_0 \\in [0, 15\\text{s})$, $W_1 \\in [15, 30\\text{s})$, $W_2 \\in [30, 60\\text{s}]$ to measure late liquidity acceleration (`flow_accel`) and buy pressure delta (`bp_delta`).
3. **Causal Empirical Bayes Reputation with Recency Decay**: Tracks early participant and deployer success rates under a strict 1-hour resolution embargo ($mt_j \\le mt_i - 3600$) with a 5-day exponential decay half-life.
4. **Multi-Level Expected Value Maximization**:
   Rather than hardcoding static $tp = 5.0$, the model predicts monotonic probabilities across multiple targets $\\text{LEVELS} = (3, 4, 5, 7, 10, 15)$ and selects:
   $$tp^*(\mathbf{x}) = \\arg\\max_{k \\in \\text{LEVELS}} \\mathbb{E}[\\text{PnL}(k \\mid \\mathbf{x})]$$
   subject to a risk-adjusted daily Sharpe margin hurdle.
"""))

# Cell 1: Environment Setup & Constants
cells.append(nbf.v4.new_code_cell("""import os, sys, glob, re, time, gc, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Set deterministic seed for full reproducibility
SEED = 42
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Token & Market Constants
WSOL = "So11111111111111111111111111111111111111112"
COST = 0.30          # Round-trip execution friction in SOL (fees + slippage buffer)
LEVELS = [3.0, 4.0, 5.0, 7.0, 10.0, 15.0]  # Take-profit hurdle grid
WIN = 3600           # Evaluation holding horizon (1 hour in seconds)
EMBARGO = 3600       # Embargo window before validation day start
HOLD_DAYS = 5        # Training days held back for hurdle calibration
TAU_DECAY = 5 * 86400 # 5-day exponential half-life for reputation recency decay

# Multi-seed ensemble configuration
SEEDS = [42, 101, 2024, 7, 777]

# Resolve data path across Kaggle and local environments
CANDIDATES = [
    '/kaggle/input/competitions/memecoins-pump-or-dump',
    '/kaggle/input/memecoins-pump-or-dump',
    'data',
    '../data'
]
DATA = next(p for p in CANDIDATES if os.path.exists(p))
print(f"Data root resolved to: {DATA}")
"""))

# Cell 2: PnL Function Definition
cells.append(nbf.v4.new_markdown_cell("""## 1. PnL Payoff Evaluation Operator

The competition metric is total simulated trading PnL:
$$\\Pi(tp) = \\begin{cases}
0, & \\text{if } tp = 0 \\\\
tp - 1.30, & \\text{if } tp \\in [3, 100] \\text{ and } \\widetilde{M} \\ge tp \\\\
\\max(\\text{final\\_x} - 1.30, -1.0), & \\text{if } tp \\in [3, 100] \\text{ and } \\widetilde{M} < tp
\\end{cases}$$
"""))

cells.append(nbf.v4.new_code_cell("""def pnl(tp, max_x, final_x, cost=COST):
    tp = np.asarray(tp, float)
    hit = (max_x >= tp) & (tp > 0)
    held = np.maximum(final_x - 1.0 - cost, -1.0)
    return np.where(tp <= 0, 0.0, np.where(hit, tp - 1.0 - cost, held))
"""))

# Cell 3: Data Ingestion & Launch Features
cells.append(nbf.v4.new_markdown_cell("""## 2. Ingestion & Token Launch Feature Extraction

Extracting pre-trade launch attributes from `pumpfun_token_creation.csv` and `pfamm_migrations.csv`:
- Deployer initial balance (`dev_balance`)
- Bundle mechanics: `bundle_size`, `bundled_buys`, `bs_ops`, `bs_amt_sum`
- Compute stats: `gas_used`, `amount_of_instructions`
- Program version: `token2022` flag
- Token name and symbol structure
"""))

cells.append(nbf.v4.new_code_cell("""train = pd.read_csv(f'{DATA}/train.csv')
test = pd.read_csv(f'{DATA}/test.csv')

if 'finalx' in train.columns:
    train = train.rename(columns={'finalx': 'final_x'})
    
for _d in (train, test):
    _d['day'] = pd.to_datetime(_d.mt, unit='s').dt.strftime('%Y-%m-%d')
    
print(f"Train universe: {len(train):,} tokens over {train.day.nunique()} days")
print(f"Test universe:  {len(test):,} tokens on {test.day.iloc[0]}")

# Process token creation features
cre_path = f'{DATA}/pumpfun_token_creation.csv'
if not os.path.exists(cre_path):
    cre_path = f'{DATA}/03_sample_pumpfun_token_creation.csv'

cre = pd.read_csv(cre_path).drop_duplicates('mint').copy()
name = cre.name.fillna('').astype(str)
sym = cre.symbol.fillna('').astype(str)
bs = cre.bundle_structure.fillna('').astype(str)

M = pd.DataFrame({'mint': cre.mint, 'creator': cre.creator.fillna('').astype(str)})
M['name_len'] = name.str.len()
M['symbol_len'] = sym.str.len()
M['name_upper'] = name.apply(lambda s: sum(ch.isupper() for ch in s) / max(len(s), 1))
M['name_eq_symbol'] = (name.str.lower().str[:4] == sym.str.lower().str[:4]).astype(np.int8)

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

if 'token_program' in cre.columns:
    M['token2022'] = (~cre.token_program.fillna('').astype(str).str.startswith('Tokenkeg')).astype(np.int8)
else:
    M['token2022'] = 0

M['created_ts'] = pd.to_numeric(cre.created_ts, errors='coerce').fillna(0).values

# Ingest migration fee and liquidity attributes
mig_path = f'{DATA}/pfamm_migrations.csv'
if not os.path.exists(mig_path):
    mig_path = f'{DATA}/02_sample_pfamm_migrations.csv'
if os.path.exists(mig_path):
    mig = pd.read_csv(mig_path, usecols=['mint', 'sol_amount', 'mint_amount', 'pool_migration_fee']).drop_duplicates('mint')
    M = M.merge(mig, on='mint', how='left')
    for c_ in ['sol_amount', 'mint_amount', 'pool_migration_fee']:
        if c_ in M.columns:
            M[c_] = pd.to_numeric(M[c_], errors='coerce').fillna(0).values

print(f"Launch metadata processed for {len(M):,} unique tokens.")
"""))

# Cell 4: High-Frequency Microstructure Feature Extraction
cells.append(nbf.v4.new_markdown_cell("""## 3. High-Frequency Microstructure Feature Extraction

Reading the high-frequency transaction tapes (bonding curve swaps and the first 60 seconds of AMM trading):
- **Direction Correction**: Inverts `'B'` when `base_token == WSOL`
- **Window Slices**: Volume acceleration and buy pressure delta between $W_2 \\in [30, 60\\text{s}]$ and $W_0 \\in [0, 15\\text{s}]$
- **MEV Sniper Detection**: Flagging `signing_wallet != fee_payer`
- **Creator First-Minute Participation**: Tracks creator buy/sell volume
- **Reserve Depth Dynamics**: Pool liquidity depth return in SOL
"""))

cells.append(nbf.v4.new_code_cell("""CURVE_COLS = ['failed', 'mint', 'ts_rel', 'slot', 'tx_idx', 'direction', 'sol', 'new_price', 'wallet', 'provided_gas_fee']
AMM_COLS   = [
    'mint', 'ts_rel', 'slot', 'tx_idx', 'direction', 'base_token', 'sol', 'new_price', 
    'signing_wallet', 'fee_payer', 'provided_gas_fee', 'coin_creator',
    'pool_base_token_reserves_after', 'pool_quote_token_reserves_after'
]

def extract_tape_features(c_df, a_df):
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

print(f"Discovered {len(DAYS)} daily transaction tape partitions.")

tape_frames, edges_list = [], []
t0_tape = time.time()

if len(DAYS) > 0:
    for i, (day, cp, ap) in enumerate(DAYS, 1):
        if not os.path.exists(cp) or not os.path.exists(ap):
            continue
        c = pd.read_parquet(cp, columns=[c for c in CURVE_COLS if c in pd.read_parquet(cp).columns])
        a = pd.read_parquet(ap, columns=[c for c in AMM_COLS if c in pd.read_parquet(ap).columns]).rename(columns={'signing_wallet': 'wallet'})
        
        tape_frames.append(extract_tape_features(c, a))
        
        a_buys = a[(a.ts_rel <= 60)]
        edges_list.append(a_buys.groupby(['mint', 'wallet'], as_index=False)['sol'].sum())
        
        del c, a; gc.collect()
        if i % 10 == 0 or i == len(DAYS):
            print(f"  Processed {i}/{len(DAYS)} tapes ({time.time() - t0_tape:.0f}s)")
            
    TAPE_DF = pd.concat(tape_frames, ignore_index=True).drop_duplicates('mint')
    AMM_EDGES = pd.concat(edges_list, ignore_index=True)
else:
    c_sample = f'{DATA}/00_sample_pumpfun_swaps.csv'
    a_sample = f'{DATA}/01_sample_pumpswap_swaps.csv'
    if os.path.exists(c_sample) and os.path.exists(a_sample):
        c = pd.read_csv(c_sample)
        a = pd.read_csv(a_sample).rename(columns={'signing_wallet': 'wallet'})
        TAPE_DF = extract_tape_features(c, a)
        a_buys = a[(a.ts_rel <= 60)]
        AMM_EDGES = a_buys.groupby(['mint', 'wallet'], as_index=False)['sol'].sum()
    else:
        TAPE_DF = pd.DataFrame({'mint': []})
        AMM_EDGES = pd.DataFrame()
"""))

# Cell 5: Causal Reputation with Recency Decay
cells.append(nbf.v4.new_markdown_cell("""## 4. Causal Empirical Bayes Participant & Deployer Reputation (Recency Weighted)

Enforcing a strict 1-hour resolution embargo ($mt_j \\le mt_i - 3600$) with a 5-day exponential decay half-life to capture active participant hot streaks without lookahead leakage.
"""))

cells.append(nbf.v4.new_code_cell("""REPUTATION_COLS = [
    'w_known', 'w_hit_max', 'w_hit_mean', 'w_hit_top3', 
    'w_elite_n', 'w_elite_share', 'w_eb_score',
    'w_decay_eb_score', 'w_decay_hits',
    'c_known', 'c_hit_rate', 'c_decay_rate'
]

def build_causal_reputation(tokens_df, amm_edges_df, creator_map=None, target_k=5.0, win=WIN, tau_decay=TAU_DECAY):
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
    
    creator_appear, creator_hit = {}, {}
    creator_decay_ap, creator_decay_ht = {}, {}
    
    mts, mxs, mints = tok.mt.values, tok.max_x.values, tok.mint.values
    out = np.zeros((len(tok), len(REPUTATION_COLS)), dtype=float)
    j = 0
    alpha_prior, beta_prior = 1.0, 15.0
    
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
                    cw = creator_map.get(mints[j])
                    if cw and str(cw) != 'nan' and len(str(cw)) > 0:
                        decay_factor = np.exp(-(cur_t - t_res) / max(tau_decay, 1.0))
                        creator_appear[cw] = creator_appear.get(cw, 0) + 1
                        creator_decay_ap[cw] = creator_decay_ap.get(cw, 0.0) + decay_factor
                        if mxs[j] >= target_k:
                            creator_hit[cw] = creator_hit.get(cw, 0) + 1
                            creator_decay_ht[cw] = creator_decay_ht.get(cw, 0.0) + decay_factor
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
            cw = creator_map.get(mints[i])
            if cw and str(cw) != 'nan':
                c_ap = creator_appear.get(cw, 0)
                if c_ap > 0:
                    out[i, 9] = 1.0
                    out[i, 10] = float(creator_hit.get(cw, 0) / c_ap)
                    out[i, 11] = float((creator_decay_ht.get(cw, 0.0) + alpha_prior) / 
                                       (creator_decay_ap.get(cw, 0.0) + alpha_prior + beta_prior))
                    
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

# Cell 6: Feature Assembly & Merging
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

# Cell 7: Multi-Level Expected Value Modeling Engine
cells.append(nbf.v4.new_markdown_cell("""## 5. Multi-Level Expected Value Prediction Engine

Architecture:
- Models $y_k = \\mathbb{I}(\\text{max\\_x} \\ge k)$ across $\\text{LEVELS} = (3, 4, 5, 7, 10, 15)$.
- Enforces monotonic probabilities: $P(\\ge 15) \\le P(\\ge 10) \\le \\dots \\le P(\\ge 3)$.
- Calculates empirical held payoffs $\\text{held\\_miss}(k) = \\mathbb{E}[\\max(\\text{final\\_x} - 1.3, -1.0) \\mid \\text{max\\_x} < k]$.
- Evaluates Expected Value: $\\mathbb{E}[\\text{PnL}(k)] = p_k \\cdot (k - 1.30) + (1 - p_k) \\cdot \\text{held\\_miss}(k)$.
- Assigns $\\arg\\max_k \\mathbb{E}[\\text{PnL}(k)]$ subject to a risk-adjusted daily Sharpe margin hurdle.
"""))

cells.append(nbf.v4.new_code_cell("""LGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.04, num_leaves=31, min_child_samples=40,
    subsample=0.80, subsample_freq=1, colsample_bytree=0.70, verbose=-1, n_jobs=-1
)

class MultiLevelExpectedValueEngine:
    def __init__(self, feature_cols, levels=LEVELS, seeds=SEEDS, lgb_params=LGB_PARAMS, cost=COST):
        self.feature_cols = feature_cols
        self.levels = np.array(levels, dtype=float)
        self.seeds = seeds
        self.lgb_params = lgb_params
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
                lgb.LGBMClassifier(random_state=s, **self.lgb_params).fit(X_tr, y_lv).predict_proba(X_te)[:, 1]
                for s in self.seeds
            ], axis=0)
            probs[:, j] = p_seed
            
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
        levels_row = self.levels[None, :]
        win_pnl = levels_row - 1.0 - self.cost
        held_row = held_by_level[None, :]
        return probs * win_pnl + (1.0 - probs) * held_row

    def decide(self, ev_matrix, min_ev_margin=0.20, fallback_q=1.0):
        best_idx = np.argmax(ev_matrix, axis=1)
        best_ev = ev_matrix[np.arange(len(ev_matrix)), best_idx]
        best_level = self.levels[best_idx]
        
        tp = np.where(best_ev >= min_ev_margin, best_level, 0.0)
        if (tp > 0).sum() == 0 and fallback_q > 0:
            top_k = max(1, int(len(ev_matrix) * fallback_q / 100.0))
            tp = np.zeros(len(ev_matrix), dtype=float)
            if best_ev.max() > best_ev.min():
                order = np.argsort(-best_ev)
                top_indices = order[:top_k]
                tp[top_indices] = best_level[top_indices]
        return tp, best_ev

def optimize_ev_margin(model, train_df, margin_grid=(0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0), hold_days=HOLD_DAYS):
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
        pnl_vals = np.where(tp <= 0, 0.0, np.where(hit, tp - 1.0 - COST, held_loss))
        daily = pd.Series(pnl_vals).groupby(cal.day).sum()
        if (tp > 0).sum() < 3:
            continue
        robust_sharpe = float(daily.mean() - 0.25 * daily.std(ddof=0))
        if robust_sharpe > best_score:
            best_score = robust_sharpe
            best_margin = margin
    return best_margin
"""))

# Cell 8: Embargoed Walk-Forward Backtest
cells.append(nbf.v4.new_markdown_cell("""## 6. Embargoed Walk-Forward Backtesting

Evaluating the multi-level policy across out-of-fold days under strict 2-hour embargo buffers.
"""))

cells.append(nbf.v4.new_code_cell("""days = sorted(TR.day.unique())
eval_days = days[max(5, len(days) - 10):]
ev_engine = MultiLevelExpectedValueEngine(feature_cols=FE)

wf_rows = []
print(f"Running walk-forward evaluation across {len(eval_days)} days...")

for day in eval_days:
    val = TR[TR.day == day].reset_index(drop=True)
    trn = TR[(TR.day < day) & (TR.mt <= val.mt.min() - WIN - EMBARGO)].reset_index(drop=True)
    if len(trn) < 1000 or len(val) < 20:
        continue
        
    probs = ev_engine.fit_predict_levels(trn, val)
    held = ev_engine.compute_held_payoffs(trn)
    ev = ev_engine.calculate_expected_values(probs, held)
    
    for margin in [0.10, 0.20, 0.30, 0.50]:
        tp_alloc, _ = ev_engine.decide(ev, min_ev_margin=margin, fallback_q=1.0)
        pnl_val = float(pnl(tp_alloc, val.max_x.values, val.final_x.values).sum())
        n_buy = int((tp_alloc > 0).sum())
        hits = int(((val.max_x.values >= tp_alloc) & (tp_alloc > 0)).sum())
        wf_rows.append(dict(day=day, margin=margin, taken=n_buy, hits=hits, pnl=pnl_val))

WF_DF = pd.DataFrame(wf_rows)
print("\\n" + "=" * 75)
print("EMBARGOED MULTI-LEVEL WALK-FORWARD SUMMARY")
print("=" * 75)
for m, g in WF_DF.groupby('margin'):
    tot_trades = g.taken.sum()
    tot_hits = g.hits.sum()
    tot_pnl = g.pnl.sum()
    prec = 100.0 * tot_hits / max(tot_trades, 1)
    win_days = (g.pnl > 0).sum()
    print(f"Margin {m:>4.2f} | Avg Trades/Day: {g.taken.mean():4.1f} | Precision: {prec:5.1f}% | Total PnL: {tot_pnl:+7.2f} SOL | Profitable Days: {win_days}/{len(g)}")
print("=" * 75)
"""))

# Cell 9: Production Fit & Submission Generation
cells.append(nbf.v4.new_markdown_cell("""## 7. Full Production Retraining & Dynamic Take-Profit Allocation

Retraining the multi-level expected value engine on the entire historical dataset, calibrating the optimal EV margin, and generating the verified `submission.csv`.
"""))

cells.append(nbf.v4.new_code_cell("""print("Fitting production multi-level engine on full historical dataset...")
production_engine = MultiLevelExpectedValueEngine(feature_cols=FE)

optimal_margin = optimize_ev_margin(production_engine, TR)
print(f"Optimal risk-adjusted EV margin: {optimal_margin:.3f}")

test_probs = production_engine.fit_predict_levels(TR, TE)
held_payoffs = production_engine.compute_held_payoffs(TR)
test_ev = production_engine.calculate_expected_values(test_probs, held_payoffs)

test_tp, best_ev = production_engine.decide(test_ev, min_ev_margin=optimal_margin, fallback_q=1.0)

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
display(sub[sub.tp > 0].head(10))
"""))

nb['cells'] = cells

with open('notebooks/submission_pipeline.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Successfully generated upgraded notebooks/submission_pipeline.ipynb with {len(cells)} cells.")
