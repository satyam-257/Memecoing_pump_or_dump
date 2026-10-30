"""Feature engineering for token creation metadata and AMM migration parameters."""
import os
import re
import numpy as np
import pandas as pd

def extract_creation_features(data_dir):
    """
    Extracts structured features from pumpfun_token_creation.csv and pfamm_migrations.csv.
    """
    cre_path = os.path.join(data_dir, 'pumpfun_token_creation.csv')
    if not os.path.exists(cre_path):
        cre_path = os.path.join(data_dir, '03_sample_pumpfun_token_creation.csv')
        
    cre = pd.read_csv(cre_path).drop_duplicates('mint').copy()
    
    # Textual features from name and symbol
    name = cre.name.fillna('').astype(str)
    sym = cre.symbol.fillna('').astype(str)
    bs = cre.bundle_structure.fillna('').astype(str)
    
    df = pd.DataFrame({'mint': cre.mint, 'creator': cre.creator.fillna('').astype(str)})
    df['name_len'] = name.str.len()
    df['symbol_len'] = sym.str.len()
    df['name_upper'] = name.apply(lambda s: sum(ch.isupper() for ch in s) / max(len(s), 1))
    df['name_eq_symbol'] = (name.str.lower().str[:4] == sym.str.lower().str[:4]).astype(np.int8)
    
    # Bundle structure analysis
    df['bs_ops'] = bs.str.split().str.len().fillna(0)
    df['bs_len'] = bs.str.len().fillna(0)
    df['bs_buys'] = bs.apply(lambda s: len(re.findall(r'\dB\d', s))).astype(np.int16)
    
    amt = bs.apply(lambda s: [int(x) for x in re.findall(r':(\d{6,})', s)])
    df['bs_amt_sum'] = amt.apply(lambda v: float(np.sum(v)) if v else 0.0)
    df['bs_amt_max'] = amt.apply(lambda v: float(np.max(v)) if v else 0.0)
    
    # Numerical opcode and compute stats
    num_cols = [
        'gas_used', 'amount_of_instructions', 'amount_of_lookup_reads', 'amount_of_lookup_writes',
        'creation_ix_index', 'pf_program_index', 'direct_pf_invocation', 'is_cashback_enabled',
        'bundle_size', 'bundled_buys', 'bundled_buys_count', 'dev_balance'
    ]
    for col in num_cols:
        if col in cre.columns:
            df[col] = pd.to_numeric(cre[col], errors='coerce').fillna(0).values
            
    # Token 2022 flag
    if 'token_program' in cre.columns:
        df['token2022'] = (~cre.token_program.fillna('').astype(str).str.startswith('Tokenkeg')).astype(np.int8)
    else:
        df['token2022'] = 0
        
    df['created_ts'] = pd.to_numeric(cre.created_ts, errors='coerce').fillna(0).values
    
    # Migration parameters
    mig_path = os.path.join(data_dir, 'pfamm_migrations.csv')
    if not os.path.exists(mig_path):
        mig_path = os.path.join(data_dir, '02_sample_pfamm_migrations.csv')
        
    if os.path.exists(mig_path):
        mig = pd.read_csv(mig_path, usecols=['mint', 'sol_amount', 'mint_amount', 'pool_migration_fee']).drop_duplicates('mint')
        df = df.merge(mig, on='mint', how='left')
        for c in ['sol_amount', 'mint_amount', 'pool_migration_fee']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).values
                
    return df
