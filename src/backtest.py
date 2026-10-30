"""Strict embargoed walk-forward backtesting protocol and exact competition metric evaluation."""
import numpy as np
import pandas as pd
from src.config import WIN, EMBARGO, LEVELS, COST
from src.models import TwoStageDecisionEngine

def evaluate_pnl(y_true_max, y_true_end, y_pred_tp, cost=COST):
    """
    Official competition offline evaluation metric:
    - y_pred_tp <= 0: Skip (PnL = 0.0)
    - hit: y_true_max >= y_pred_tp -> PnL = y_pred_tp - 1.0 - cost
    - miss: y_true_max < y_pred_tp -> PnL = max(y_true_end - 1.0 - cost, -1.0)
    """
    y_pred_tp = np.asarray(y_pred_tp, dtype=float)
    y_true_max = np.asarray(y_true_max, dtype=float)
    y_true_end = np.asarray(y_true_end, dtype=float)
    
    pnl = 0.0
    mask = y_pred_tp > 0
    hit = mask & (y_true_max >= y_pred_tp)
    miss = mask & (y_true_max < y_pred_tp)
    
    pnl += np.sum(y_pred_tp[hit] - 1.0 - cost)
    pnl += np.sum(np.maximum(y_true_end[miss] - 1.0 - cost, -1.0))
    return float(pnl)

def run_walk_forward_backtest(df, feature_cols, top_n_trades=(2, 3), n_eval_days=7):
    """
    Simulates real-world daily forward evaluation across consecutive calendar days.
    Enforces the strict safety rule: target_mt <= mt_train_max - 3600.
    """
    days = sorted(df['day'].unique())
    start_idx = max(5, len(days) - n_eval_days)
    eval_days = days[start_idx:]
    
    engine = TwoStageDecisionEngine(feature_cols=feature_cols, levels=LEVELS)
    all_results = []
    
    print(f"Starting walk-forward validation across {len(eval_days)} days ({eval_days[0]} to {eval_days[-1]})...")
    
    for top_n in top_n_trades:
        print(f"\n--- Evaluating Policy Tier: Top {top_n} trades/day ---")
        day_rows = []
        for day in eval_days:
            val = df[df['day'] == day].reset_index(drop=True)
            # Enforce strict 1-hour resolution embargo + gap between train and validation days
            trn = df[(df['day'] < day) & (df['mt'] <= val['mt'].min() - WIN - EMBARGO)].reset_index(drop=True)
            
            if len(trn) < 500 or len(val) < 10:
                continue
                
            engine.fit_gatekeeper(trn)
            engine.fit_levels(trn)
            
            p_gate = engine.predict_gatekeeper(val)
            p_levels = engine.predict_levels(val)
            miss_payoffs = engine.compute_miss_payoffs(trn)
            ev = engine.calculate_expected_values(p_levels, miss_payoffs)
            
            tp, _ = engine.decide(p_gate, ev, top_n_daily=top_n, min_prob_hurdle=0.20, min_ev_margin=0.0)
            
            pnl = evaluate_pnl(val['max_x'].values, val['final_x'].values, tp)
            taken = int((tp > 0).sum())
            hits = int(((val['max_x'].values >= tp) & (tp > 0)).sum())
            
            row = {
                'day': day,
                'top_n': top_n,
                'tokens': len(val),
                'taken': taken,
                'hits': hits,
                'precision_pct': (hits / taken * 100.0) if taken > 0 else 0.0,
                'daily_pnl': pnl
            }
            day_rows.append(row)
            all_results.append(row)
            print(f"Day {day}: taken={taken:>2}, hits={hits:>2}, PnL={pnl:+6.2f} SOL")
            
    res_df = pd.DataFrame(all_results)
    return res_df

def print_backtest_summary(res_df):
    """Prints a cleanly formatted summary table by top_n tier."""
    print("\n" + "=" * 80)
    print("WALK-FORWARD PERFORMANCE SUMMARY BY CONVICTION TIER")
    print("=" * 80)
    for top_n, grp in res_df.groupby('top_n'):
        tot_taken = grp['taken'].sum()
        tot_hits = grp['hits'].sum()
        tot_pnl = grp['daily_pnl'].sum()
        green_days = (grp['daily_pnl'] > 0).sum()
        total_days = len(grp)
        prec = (tot_hits / tot_taken * 100.0) if tot_taken > 0 else 0.0
        mean_pnl = grp['daily_pnl'].mean()
        std_pnl = grp['daily_pnl'].std(ddof=0)
        sharpe = mean_pnl - 0.25 * std_pnl
        
        print(f"Tier: Top {top_n} trades/day | Trades: {tot_taken:>2} | Hits: {tot_hits:>2} | "
              f"Precision: {prec:5.1f}% | Total PnL: {tot_pnl:+7.2f} SOL | "
              f"Sharpe: {sharpe:+.2f} | Win Days: {green_days}/{total_days} ({green_days/total_days*100:.0f}%)")
    print("=" * 80)
