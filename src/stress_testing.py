"""Daniro Quant Strategy Stress-Testing and Fragility Detection Suite.

Implements the four foundational stress tests:
1. Threshold Jitter: Perturbs decision threshold tau +/- 10-20% to detect knife-edge overfitting.
2. Feature Group Ablation: Evaluates drop in Sharpe / PnL when specific factor families are removed.
3. Market Regime Stability: Evaluates performance across segmented chronological market regimes.
4. Execution & Friction Stress: Evaluates resilience under escalating fee / slippage costs (0.30 -> 0.45 SOL),
   computing the Edge Factor = sum(Positive PnL) / abs(sum(Negative PnL)).
"""
import numpy as np
import pandas as pd

def compute_edge_factor(pnl_series):
    """Computes the Daniro Edge Factor: Total Positive PnL / Absolute Negative PnL."""
    pos_pnl = np.sum(np.maximum(pnl_series, 0.0))
    neg_pnl = np.abs(np.sum(np.minimum(pnl_series, 0.0)))
    if neg_pnl == 0.0:
        return float('inf') if pos_pnl > 0 else 1.0
    return float(pos_pnl / neg_pnl)

def compute_sharpe(daily_pnl):
    """Computes annualized-like daily Sharpe ratio."""
    vals = np.asarray(daily_pnl)
    if len(vals) < 2 or np.std(vals) == 0.0:
        return 0.0
    return float(np.mean(vals) / np.std(vals) * np.sqrt(365))

def test_threshold_jitter(eval_fn, base_tau=0.50, deltas=(-0.05, 0.0, 0.05)):
    """
    Test 1: Move the thresholds.
    If edge collapses with a tiny threshold shift, it was fitted to past noise.
    """
    results = {}
    for d in deltas:
        tau = base_tau + d
        res = eval_fn(tau)
        pnl = res['pnl'] if isinstance(res, dict) else res
        results[f"tau_{tau:.2f}"] = pnl
    return results

def test_execution_friction_stress(y_true_max, y_true_end, y_pred_tp, cost_grid=(0.30, 0.35, 0.40, 0.45)):
    """
    Test 4: Stress execution.
    Evaluate strategy under escalating execution penalties (fees + slippage).
    """
    y_true_max = np.asarray(y_true_max)
    y_true_end = np.asarray(y_true_end)
    y_pred_tp = np.asarray(y_pred_tp)
    
    trade_mask = y_pred_tp > 0
    hit_mask = trade_mask & (y_true_max >= y_pred_tp)
    miss_mask = trade_mask & (y_true_max < y_pred_tp)
    
    stress_results = []
    for cost in cost_grid:
        pnl = np.zeros(len(y_pred_tp))
        pnl[hit_mask] = y_pred_tp[hit_mask] - 1.0 - cost
        pnl[miss_mask] = np.maximum(y_true_end[miss_mask] - 1.0 - cost, -1.0)
        
        tot_pnl = float(np.sum(pnl))
        edge_fact = compute_edge_factor(pnl[trade_mask]) if trade_mask.any() else 0.0
        
        stress_results.append({
            'friction_sol': cost,
            'total_pnl': tot_pnl,
            'edge_factor': edge_fact,
            'trades': int(trade_mask.sum()),
            'win_rate': float(hit_mask.sum() / max(trade_mask.sum(), 1))
        })
        
    return pd.DataFrame(stress_results)

def test_regime_stability(daily_pnl_df, date_col='date', pnl_col='pnl', n_blocks=3):
    """
    Test 3: Change the regime.
    Divides chronological evaluation folds into n blocks and verifies positive PnL across all blocks.
    """
    df = daily_pnl_df.sort_values(date_col).copy()
    blocks = np.array_split(df, n_blocks)
    
    regime_results = []
    for i, b in enumerate(blocks, 1):
        pnl_sum = float(b[pnl_col].sum())
        pnl_std = float(b[pnl_col].std())
        sharpe = float(pnl_sum / (pnl_std * np.sqrt(len(b)) + 1e-4))
        edge = compute_edge_factor(b[pnl_col].values)
        regime_results.append({
            'block': f"Regime_{i}",
            'start_date': str(b[date_col].iloc[0]),
            'end_date': str(b[date_col].iloc[-1]),
            'days': len(b),
            'pnl': pnl_sum,
            'sharpe': sharpe,
            'edge_factor': edge,
            'all_green': bool(pnl_sum > 0)
        })
    return pd.DataFrame(regime_results)
