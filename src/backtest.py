"""Strict embargoed walk-forward backtesting protocol and 16-day comparative benchmark across 3 architectures."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.config import WIN, EMBARGO, LEVELS, COST, SEEDS, LGB_PARAMS, HOLD_DAYS
from src.models import UnifiedTwoTierEngine

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
    comparing:
    (a) Flat TP=5 Baseline (Single-Stage LightGBM)
    (b) Unified LightGBM+CatBoost Entry (target >=5x) with Fixed TP=5
    (c) Unified LightGBM+CatBoost Entry with Constrained {5.0, 8.0} Exit Upgrade
    """
    days = sorted(df['day'].unique())
    eval_days = days[-16:]
    
    baseline_records = []
    unified_fixed5_records = []
    unified_2tier_records = []
    
    print("=" * 105)
    print(f"STARTING 16-DAY EMBARGOED WALK-FORWARD BENCHMARK ({eval_days[0]} to {eval_days[-1]})")
    print(f"Safety Protocol Enforced: target_mt <= mt_train_max - 3600 (Strict 1-hour resolution embargo)")
    print("=" * 105)
    
    for i, day in enumerate(eval_days, 1):
        val = df[df['day'] == day].reset_index(drop=True)
        # Strict 1-hour resolution embargo before day start
        trn = df[(df['day'] < day) & (df['mt'] <= val['mt'].min() - WIN - EMBARGO)].reset_index(drop=True)
        
        if len(trn) < 500 or len(val) < 10:
            continue
            
        mxs_val = val['max_x'].values
        fxs_val = val['final_x'].values
        
        # -----------------------------------------------------------------
        # (a) FLAT TP=5 BASELINE (Single-Stage LightGBM reference)
        # -----------------------------------------------------------------
        X_tr = trn[feature_cols].astype(float).fillna(0).values
        y_tr_5x = (trn['max_x'].values >= 5.0).astype(int)
        X_val = val[feature_cols].astype(float).fillna(0).values
        
        base_model = lgb.LGBMClassifier(random_state=SEEDS[0], **LGB_PARAMS)
        base_model.fit(X_tr, y_tr_5x)
        p_base_val = base_model.predict_proba(X_val)[:, 1]
        
        # Calibrate baseline scalar threshold using OOF holdout
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
        
        # -----------------------------------------------------------------
        # (b) & (c) UNIFIED LIGHTGBM + CATBOOST ENTRY & 2-TIER EXIT ENGINE
        # -----------------------------------------------------------------
        engine = UnifiedTwoTierEngine(feature_cols=feature_cols, seeds=[42])
        engine.fit_entry(trn)
        
        p_entry = engine.predict_entry(val)
        ev_matrix = engine.calculate_expected_values(test_df=val)
        tau_scalar = engine.calibrate_scalar_threshold(trn, target_daily_trades=2.5)
        
        # (b) Architecture B: Unified LGB+CatBoost Entry with Fixed TP=5
        tp_fixed5, _ = engine.decide_per_token(p_entry, ev_matrix, tau_scalar=tau_scalar, mode='fixed5')
        pnl_fixed5 = evaluate_pnl(mxs_val, fxs_val, tp_fixed5)
        taken_fixed5 = int((tp_fixed5 > 0).sum())
        hits_fixed5 = int(((mxs_val >= 5.0) & (tp_fixed5 > 0)).sum())
        
        unified_fixed5_records.append({
            'day': day, 'taken': taken_fixed5, 'hits': hits_fixed5, 'pnl': pnl_fixed5
        })
        
        # (c) Architecture C: Unified LGB+CatBoost Entry with Constrained {5.0, 8.0} Exit Upgrade
        tp_2tier, _ = engine.decide_per_token(p_entry, ev_matrix, tau_scalar=tau_scalar, safety_margin=0.35, mode='2tier')
        pnl_2tier = evaluate_pnl(mxs_val, fxs_val, tp_2tier)
        taken_2tier = int((tp_2tier > 0).sum())
        hits_2tier = int(((mxs_val >= tp_2tier) & (tp_2tier > 0)).sum())
        upgrades_2tier = int((tp_2tier == 8.0).sum())
        
        unified_2tier_records.append({
            'day': day, 'taken': taken_2tier, 'hits': hits_2tier, 'upgrades': upgrades_2tier, 'pnl': pnl_2tier
        })
        
        print(f"[{i:02d}/16] Day {day} | (a) Baseline: trades={taken_base:>2}, hits={hits_base:>2}, PnL={pnl_base:+6.2f} SOL | "
              f"(b) Unified Fixed-5: trades={taken_fixed5:>2}, hits={hits_fixed5:>2}, PnL={pnl_fixed5:+6.2f} SOL | "
              f"(c) 2-Tier Upgrade: trades={taken_2tier:>2}, hits={hits_2tier:>2} (upg={upgrades_2tier}), PnL={pnl_2tier:+6.2f} SOL")
              
    b_df = pd.DataFrame(baseline_records)
    f5_df = pd.DataFrame(unified_fixed5_records)
    t2_df = pd.DataFrame(unified_2tier_records)
    
    # Telemetry and summary comparison
    print("\n" + "=" * 105)
    print("16-DAY EMBARGOED WALK-FORWARD HEAD-TO-HEAD BENCHMARK RESULTS")
    print("=" * 105)
    print(f"{'Strategy Architecture':<42} | {'Trades':<6} | {'Hits':<5} | {'Precision':<10} | {'Total PnL':<12} | {'Sharpe':<7} | {'Win Days'}")
    print("-" * 105)
    
    def print_metrics(name, d):
        tot_trades = d['taken'].sum()
        tot_hits = d['hits'].sum()
        tot_pnl = d['pnl'].sum()
        prec = (tot_hits / max(tot_trades, 1)) * 100.0
        mean_pnl = d['pnl'].mean()
        std_pnl = d['pnl'].std(ddof=0)
        sharpe = mean_pnl - 0.25 * std_pnl
        win_days = (d['pnl'] > 0).sum()
        print(f"{name:<42} | {tot_trades:<6} | {tot_hits:<5} | {prec:>8.1f}% | {tot_pnl:>+10.2f} SOL | {sharpe:>+6.2f} | {win_days}/{len(d)} ({win_days/len(d)*100:.0f}%)")
        return tot_pnl, sharpe
        
    pnl_a, sh_a = print_metrics("(a) Flat TP=5 Single-Stage Baseline", b_df)
    pnl_b, sh_b = print_metrics("(b) Unified LGB+CB Entry (Fixed TP=5)", f5_df)
    pnl_c, sh_c = print_metrics("(c) Unified LGB+CB Entry ({5,8} Exit Upgrade)", t2_df)
    print("=" * 105)
    
    diff_b = pnl_b - pnl_a
    diff_c = pnl_c - pnl_a
    print(f"Alpha Gain (b vs Baseline): {diff_b:+.2f} SOL")
    print(f"Alpha Gain (c vs Baseline): {diff_c:+.2f} SOL")
    
    best_arch = "(c) Unified LGB+CB Entry ({5,8} Exit Upgrade)" if pnl_c >= pnl_b else "(b) Unified LGB+CB Entry (Fixed TP=5)"
    print(f"\nWinning Strategy to Deploy: {best_arch}")
    return b_df, f5_df, t2_df
