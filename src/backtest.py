"""Strict 16-day embargoed walk-forward backtesting protocol."""
import numpy as np
import pandas as pd
from src.config import WIN, EMBARGO, TARGET_K
from src.metrics import compute_pnl
from src.models import TwoStageCascadeEnsemble, calibrate_decision_threshold, make_decisions

def run_walk_forward_backtest(df, feature_cols, depths=(0.5, 1.0, 1.5, 2.0), n_eval_days=14):
    """
    Simulates real-world daily forward evaluation across consecutive days.
    """
    days = sorted(df['day'].unique())
    start_idx = max(5, len(days) - n_eval_days)
    eval_days = days[start_idx:]
    
    model = TwoStageCascadeEnsemble(feature_cols=feature_cols)
    results = []
    
    print(f"Starting walk-forward validation across {len(eval_days)} days ({eval_days[0]} to {eval_days[-1]})...")
    
    for day in eval_days:
        val = df[df['day'] == day].reset_index(drop=True)
        # Enforce strict 2-hour resolution embargo between train and validation days
        trn = df[(df['day'] < day) & (df['mt'] <= val['mt'].min() - WIN - EMBARGO)].reset_index(drop=True)
        
        if len(trn) < 1000 or len(val) < 20:
            continue
            
        scores = model.fit_predict(trn, val)
        
        for q in depths:
            thr = calibrate_decision_threshold(model, trn, q_depth=q)
            tp = make_decisions(scores, thr, target_k=TARGET_K)
            
            pnl_series = compute_pnl(tp, val['max_x'].values, val['final_x'].values)
            taken = int((tp > 0).sum())
            hits = int(((val['max_x'].values >= TARGET_K) & (tp > 0)).sum())
            pnl = float(pnl_series.sum())
            
            results.append({
                'day': day,
                'depth_pct': q,
                'tokens': len(val),
                'taken': taken,
                'hits': hits,
                'precision_pct': (hits / taken * 100.0) if taken > 0 else 0.0,
                'daily_pnl': pnl
            })
            
    res_df = pd.DataFrame(results)
    return res_df

def print_backtest_summary(res_df):
    """Prints a cleanly formatted summary table by depth tier."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD PERFORMANCE SUMMARY BY DEPTH TIER")
    print("=" * 80)
    for q, grp in res_df.groupby('depth_pct'):
        tot_taken = grp['taken'].sum()
        tot_hits = grp['hits'].sum()
        tot_pnl = grp['daily_pnl'].sum()
        green_days = (grp['daily_pnl'] > 0).sum()
        total_days = len(grp)
        prec = (tot_hits / tot_taken * 100.0) if tot_taken > 0 else 0.0
        
        print(f"Depth {q:>4.1f}% | Avg Trades/Day: {grp['taken'].mean():4.1f} | "
              f"Precision: {prec:5.1f}% | Total PnL: {tot_pnl:+7.2f} SOL | "
              f"Win Days: {green_days}/{total_days} ({green_days/total_days*100:.0f}%)")
    print("=" * 80)
