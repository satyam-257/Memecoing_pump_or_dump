"""Strict embargoed walk-forward backtesting protocol and 16-day comparative benchmark."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.config import WIN, EMBARGO, LEVELS, COST, SEEDS, LGB_PARAMS, HOLD_DAYS
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

def run_16day_comparison(df, feature_cols):
    """
    Runs a full 16-day walk-forward evaluation (1 fold per day, training only on tokens
    resolved >= 1 hour before the fold day starts) and prints a side-by-side comparison table
    of total PnL, trade count, and precision against a flat tp=5 baseline.
    """
    days = sorted(df['day'].unique())
    eval_days = days[-16:]
    
    engine = TwoStageDecisionEngine(feature_cols=feature_cols, levels=LEVELS)
    
    baseline_records = []
    two_stage_records = []
    
    print("=" * 90)
    print(f"STARTING 16-DAY EMBARGOED WALK-FORWARD BENCHMARK ({eval_days[0]} to {eval_days[-1]})")
    print(f"Safety Rule Enforced: target_mt <= mt_train_max - 3600")
    print("=" * 90)
    
    for i, day in enumerate(eval_days, 1):
        val = df[df['day'] == day].reset_index(drop=True)
        # Strict 1-hour resolution embargo before day start
        trn = df[(df['day'] < day) & (df['mt'] <= val['mt'].min() - WIN - EMBARGO)].reset_index(drop=True)
        
        if len(trn) < 500 or len(val) < 10:
            continue
            
        mxs_val = val['max_x'].values
        fxs_val = val['final_x'].values
        
        # -------------------------------------------------------------
        # 1. FLAT TP=5 BASELINE (Standard single-stage model)
        # -------------------------------------------------------------
        X_tr = trn[feature_cols].astype(float).fillna(0).values
        y_tr_5x = (trn['max_x'].values >= 5.0).astype(int)
        X_val = val[feature_cols].astype(float).fillna(0).values
        
        base_model = lgb.LGBMClassifier(random_state=SEEDS[0], **LGB_PARAMS)
        base_model.fit(X_tr, y_tr_5x)
        p_base_val = base_model.predict_proba(X_val)[:, 1]
        
        # Calibrate baseline scalar threshold using OOF holdout to prevent inflation
        cal_days = sorted(trn['day'].unique())[-HOLD_DAYS:]
        trn_cal = trn[trn['day'].isin(cal_days)].reset_index(drop=True)
        trn_old = trn[(trn['day'] < cal_days[0]) & (trn['mt'] <= trn_cal['mt'].min() - WIN - EMBARGO)].reset_index(drop=True)
        
        if len(trn_old) >= 500:
            base_cal_model = lgb.LGBMClassifier(random_state=SEEDS[0], **LGB_PARAMS).fit(
                trn_old[feature_cols].astype(float).fillna(0).values,
                (trn_old['max_x'].values >= 5.0).astype(int)
            )
            p_base_cal = base_cal_model.predict_proba(trn_cal[feature_cols].astype(float).fillna(0).values)[:, 1]
        else:
            p_base_cal = p_base_val
        
        q_base = min(max(2.5 / max(len(trn_cal) / float(len(cal_days)), 10.0), 0.001), 0.05)
        tau_base = float(np.quantile(p_base_cal, 1.0 - q_base))
        
        tp_base = np.where(p_base_val >= tau_base, 5.0, 0.0)
        pnl_base = evaluate_pnl(mxs_val, fxs_val, tp_base)
        taken_base = int((tp_base > 0).sum())
        hits_base = int(((mxs_val >= 5.0) & (tp_base > 0)).sum())
        
        baseline_records.append({
            'day': day, 'taken': taken_base, 'hits': hits_base, 'pnl': pnl_base
        })
        
        # -------------------------------------------------------------
        # 2. UPGRADED DECOUPLED TWO-STAGE ENGINE (Per-Token Causal Execution)
        # -------------------------------------------------------------
        engine.fit_gatekeeper(trn)
        engine.fit_levels(trn)
        
        p_gate = engine.predict_gatekeeper(val)
        p_levels = engine.predict_levels(val)
        miss_payoffs = engine.compute_miss_payoffs(trn)
        ev_matrix = engine.calculate_expected_values(p_levels, miss_payoffs)
        
        # Per-token scalar cutoff calibrated strictly on trn holdout days (OOF)
        tau_scalar = engine.calibrate_scalar_threshold(trn, target_daily_trades=2.5)
        tp_twostage, _ = engine.decide_per_token(p_gate, ev_matrix, tau_scalar=tau_scalar)
        
        pnl_twostage = evaluate_pnl(mxs_val, fxs_val, tp_twostage)
        taken_twostage = int((tp_twostage > 0).sum())
        hits_twostage = int(((mxs_val >= tp_twostage) & (tp_twostage > 0)).sum())
        
        two_stage_records.append({
            'day': day, 'taken': taken_twostage, 'hits': hits_twostage, 'pnl': pnl_twostage
        })
        
        print(f"[{i:02d}/16] Day {day} | Baseline (tp=5): trades={taken_base:>2}, hits={hits_base:>2}, PnL={pnl_base:+6.2f} SOL | "
              f"Two-Stage: trades={taken_twostage:>2}, hits={hits_twostage:>2}, PnL={pnl_twostage:+6.2f} SOL")
              
    b_df = pd.DataFrame(baseline_records)
    t_df = pd.DataFrame(two_stage_records)
    
    # Telemetry and summary comparison
    print("\n" + "=" * 90)
    print("16-DAY EMBARGOED WALK-FORWARD HEAD-TO-HEAD COMPARISON")
    print("=" * 90)
    print(f"{'Strategy Architecture':<38} | {'Trades':<6} | {'Hits':<5} | {'Precision':<10} | {'Total PnL':<12} | {'Sharpe':<7} | {'Win Days'}")
    print("-" * 90)
    
    b_tot_trades = b_df['taken'].sum()
    b_tot_hits = b_df['hits'].sum()
    b_tot_pnl = b_df['pnl'].sum()
    b_prec = (b_tot_hits / max(b_tot_trades, 1)) * 100.0
    b_mean = b_df['pnl'].mean()
    b_std = b_df['pnl'].std(ddof=0)
    b_sharpe = b_mean - 0.25 * b_std
    b_win = (b_df['pnl'] > 0).sum()
    
    t_tot_trades = t_df['taken'].sum()
    t_tot_hits = t_df['hits'].sum()
    t_tot_pnl = t_df['pnl'].sum()
    t_prec = (t_tot_hits / max(t_tot_trades, 1)) * 100.0
    t_mean = t_df['pnl'].mean()
    t_std = t_df['pnl'].std(ddof=0)
    t_sharpe = t_mean - 0.25 * t_std
    t_win = (t_df['pnl'] > 0).sum()
    
    print(f"{'Flat TP=5 Baseline (Single-Stage)':<38} | {b_tot_trades:<6} | {b_tot_hits:<5} | {b_prec:>8.1f}% | {b_tot_pnl:>+10.2f} SOL | {b_sharpe:>+6.2f} | {b_win}/{len(b_df)} ({b_win/len(b_df)*100:.0f}%)")
    print(f"{'Two-Stage Gatekeeper + Exit Optimizer':<38} | {t_tot_trades:<6} | {t_tot_hits:<5} | {t_prec:>8.1f}% | {t_tot_pnl:>+10.2f} SOL | {t_sharpe:>+6.2f} | {t_win}/{len(t_df)} ({t_win/len(t_df)*100:.0f}%)")
    print("=" * 90)
    
    diff_pnl = t_tot_pnl - b_tot_pnl
    print(f"Net Alpha Gain over Baseline: {diff_pnl:+.2f} SOL")
    return b_df, t_df
