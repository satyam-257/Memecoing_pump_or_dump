"""Evaluation metric and PnL calculation matching competition specifications."""
import numpy as np
import pandas as pd
from src.config import COST, TARGET_K

def compute_pnl(tp, max_x, final_x, cost=COST):
    """
    Computes realized PnL per token according to official competition rules:
    - tp <= 0: Skip (PnL = 0.0)
    - tp in [3, 100] and max_x >= tp: Hit take profit (PnL = tp - 1 - cost)
    - tp in [3, 100] and max_x < tp: Hold to end of hour (PnL = max(final_x - 1 - cost, -1.0))
    """
    tp = np.asarray(tp, dtype=float)
    max_x = np.asarray(max_x, dtype=float)
    final_x = np.asarray(final_x, dtype=float)
    
    hit = (max_x >= tp) & (tp > 0)
    held = np.maximum(final_x - 1.0 - cost, -1.0)
    
    return np.where(tp <= 0, 0.0, np.where(hit, tp - 1.0 - cost, held))

def summarize_performance(tp, max_x, final_x, target_k=TARGET_K, cost=COST):
    """
    Computes key performance metrics across a set of trades.
    """
    pnl = compute_pnl(tp, max_x, final_x, cost=cost)
    tp = np.asarray(tp, dtype=float)
    max_x = np.asarray(max_x, dtype=float)
    
    taken = int((tp > 0).sum())
    total_tokens = len(tp)
    hits = int(((max_x >= target_k) & (tp > 0)).sum())
    total_pnl = float(pnl.sum())
    
    precision = (hits / taken * 100.0) if taken > 0 else 0.0
    mean_pnl_trade = (total_pnl / taken) if taken > 0 else 0.0
    
    return {
        'total_tokens': total_tokens,
        'taken_trades': taken,
        'trade_pct': (taken / total_tokens * 100.0) if total_tokens > 0 else 0.0,
        'hits': hits,
        'precision_pct': precision,
        'total_pnl': total_pnl,
        'mean_pnl_per_trade': mean_pnl_trade
    }
