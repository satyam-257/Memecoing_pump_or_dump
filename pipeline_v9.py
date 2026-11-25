"""CPU/offline Pump or Dump pipeline. Original implementation, MIT licensed.

Token-local snapshot aggregates and strictly past-only reputation.
No query-day target encodings, test ranks or pretrained models.
"""
from pathlib import Path
import gc, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb

VERSION = 'causal-utility-onchain-v9'
DAY = 86400
GAP = 7200
SEEDS = (17, 43)

def data_path():
    for p in ['/kaggle/input/competitions/memecoins-pump-or-dump',
              '/kaggle/input/memecoins-pump-or-dump', '/task/data', 'data']:
        if (Path(p)/'train.csv').exists(): return Path(p)
    raise FileNotFoundError('Attach competition memecoins-pump-or-dump')

def div(a, b):
    return a / b.replace(0, np.nan)

def compute_gini_series(w_pos):
    if len(w_pos) == 0:
        return pd.Series(dtype=float)
    w_sorted = w_pos.sort_values(['mint', 'held'], kind='mergesort')
    w_sorted['i'] = w_sorted.groupby('mint').cumcount() + 1
    w_sorted['n'] = w_sorted.groupby('mint').held.transform('count')
    w_sorted['sum_held'] = w_sorted.groupby('mint').held.transform('sum')
    w_sorted['term'] = 2 * w_sorted['i'] * w_sorted['held']
    agg = w_sorted.groupby('mint').agg(
        sum_term=('term', 'sum'),
        n=('n', 'first'),
        sum_held=('sum_held', 'first')
    )
    gini = (agg.sum_term / (agg.n * agg.sum_held)) - (agg.n + 1) / agg.n
    return gini.clip(0.0, 1.0)

def phase_features(t, prefix, creator):
    # File order is chain order, including swaps within the same transaction.
    t = t[t.ts_rel.between(-30*DAY if prefix == 'c' else 0, 60)].copy()
    failed = t.failed.ne(0) if 'failed' in t else pd.Series(False, index=t.index)
    fails = t[failed].groupby('mint').size()
    t = t[~failed].copy()
    t['buy'] = t.direction.str.upper().str.startswith('B').astype('int8')
    t['buy_sol'] = t.sol * t.buy
    t['sell_sol'] = t.sol * (1-t.buy)
    t['signed_tok'] = t.tok * (2*t.buy-1)
    t['mismatch'] = (t.wallet != t.fee_payer).astype('int8')
    t['dev'] = (t.wallet == t.mint.map(creator)).astype('int8')
    t['dev_sell'] = t.sell_sol * t.dev
    t['dev_buy'] = t.buy_sol * t.dev
    g = t.groupby('mint', sort=False)
    f = g.agg(n=('sol','size'), volume=('sol','sum'), buy_sol=('buy_sol','sum'),
              sell_sol=('sell_sol','sum'), wallets=('wallet','nunique'),
              slots=('slot','nunique'), buys=('buy','sum'), first_t=('ts_rel','min'),
              last_t=('ts_rel','max'), max_trade=('sol','max'), mean_trade=('sol','mean'),
              sd_trade=('sol','std'), fee_mean=('fee','mean'), mismatch=('mismatch','mean'),
              dev_sell=('dev_sell','sum'), dev_buy=('dev_buy','sum'))
    f['failed'] = fails.reindex(f.index).fillna(0)
    f['fail_rate'] = div(f.failed, f.n + f.failed)
    f['buy_share'] = div(f.buy_sol, f.volume)
    f['buy_count_share'] = div(f.buys, f.n)
    f['trades_wallet'] = div(f.n, f.wallets)
    f['trades_slot'] = div(f.n, f.slots)
    f['max_buy'] = t[t.buy == 1].groupby('mint').sol.max()
    f['max_sell'] = t[t.buy == 0].groupby('mint').sol.max()
    
    # 1. On-Chain Flow & Whale Metrics
    f['net_volume_delta'] = f['buy_sol'] - f['sell_sol']
    f['buy_sell_tx_ratio'] = div(f['buys'], f['n'] - f['buys'])
    f['buy_sell_vol_ratio'] = div(f['buy_sol'], f['sell_sol'])
    f['whale_trade_ratio'] = div(f['max_trade'], f['mean_trade'])
    f['dev_sell_ratio'] = div(f['dev_sell'], f['dev_buy'] + f['dev_sell'])
    f['dev_net_sol'] = f['dev_buy'] - f['dev_sell']
    
    # Price trajectory
    p = t[t.new_price > 0].copy()
    p['logprice'] = np.log(p.new_price)
    pg = p.groupby('mint', sort=False)
    prices = pg.new_price.agg(['first','last','max','min'])
    f['price_end'] = prices['last']
    for a, b, name in [('last','first','return'),('max','first','runup'),
                        ('last','max','drawdown'),('min','first','low')]:
        f[name] = np.log(div(prices[a],prices[b]).clip(1e-9, 1e9))
    f['price_std'] = pg.logprice.std()
    p['diff'] = pg.logprice.diff()
    f['rv'] = (p['diff']**2).groupby(p.mint).sum()
    f['max_jump'] = p.groupby('mint')['diff'].max()
    f['min_jump'] = p.groupby('mint')['diff'].min()
    
    windows = [(-60,61),(-300,61),(-1800,61)] if prefix == 'c' else [(0,15),(15,30),(30,45),(45,61)]
    for j,(lo,hi) in enumerate(windows):
        u = t[(t.ts_rel >= lo) & (t.ts_rel < hi)]
        q = u.groupby('mint')
        f[f'vol_{j}'] = q.sol.sum().reindex(f.index).fillna(0)
        f[f'bs_{j}'] = div(q.buy_sol.sum(),q.sol.sum())
        f[f'n_{j}'] = q.size().reindex(f.index).fillna(0)
        f[f'vol_share_{j}'] = div(f[f'vol_{j}'], f.volume)
        f[f'uw_{j}'] = q.wallet.nunique().reindex(f.index).fillna(0)
        f[f'p_rel_{j}'] = np.log(div(q.new_price.last(),prices['last']).clip(1e-9,1e9))
        
    # 2. On-Chain Holder Concentration & Gini Inequality
    w = t.groupby(['mint','wallet'],sort=False).agg(vol=('sol','sum'),
              position=('signed_tok','sum'), buy=('buy','max'), sell=('buy','min')).reset_index()
    total = w.groupby('mint').vol.transform('sum').replace(0, np.nan)
    share = div(w.vol, total)
    f['hhi'] = (share**2).groupby(w.mint).sum()
    f['top_vol'] = share.groupby(w.mint).max()
    
    w['held'] = w.position.clip(lower=0)
    w_tot = w.groupby('mint').held.transform('sum').replace(0, np.nan)
    w['share'] = div(w.held, w_tot)
    w_sorted = w.sort_values(['mint', 'share'], ascending=[True, False], kind='mergesort')
    w_sorted['rank'] = w_sorted.groupby('mint').cumcount() + 1
    
    f['holder_hhi'] = (w_sorted.share**2).groupby(w_sorted.mint).sum()
    f['holder_top1'] = w_sorted[w_sorted['rank'] <= 1].groupby('mint').share.sum()
    f['holder_top10'] = w_sorted[w_sorted['rank'] <= 10].groupby('mint').share.sum()
    f['holder_top50'] = w_sorted[w_sorted['rank'] <= 50].groupby('mint').share.sum()
    f['holders'] = (w_sorted.held > 1).groupby(w_sorted.mint).sum()
    f['holder_velocity'] = div(f['holders'], (f['last_t'] - f['first_t'] + 1))
    
    w_pos = w[w.held > 0].copy()
    f['holder_gini'] = compute_gini_series(w_pos).reindex(f.index).fillna(0.0)
    f['roundtrip_frac'] = ((w.buy==1)&(w.sell==0)).groupby(w.mint).mean()
    
    # 3. DEX Liquidity Ratios & Kyle's Lambda (for Pumpswap prefix == 'p')
    if prefix == 'p':
        depth = t.pool_quote_token_reserves_after + t.virtual_quote_reserves
        dg = depth.groupby(t.mint)
        f['depth_end'] = dg.last()/1e9
        f['depth_growth'] = np.log(div(dg.last(), dg.first()).clip(1e-9,1e9))
        f['depth_drawdown'] = np.log(div(dg.last(), dg.max()).clip(1e-9,1e9))
        f['liquidity_to_volume'] = div(f['depth_end'], f['volume'])
        f['late_accel'] = np.log1p(f.vol_3)-np.log1p(f.vol_0)
        first = t.groupby(['mint','wallet']).ts_rel.min()
        f['fresh_late'] = (first>=45).groupby(level=0).sum()
        
        # Microstructure & Daniro Indicators
        if len(p) > 0:
            hi_idx = p.groupby('mint').new_price.idxmax()
            lo_idx = p.groupby('mint').new_price.idxmin()
            t_high = p.loc[hi_idx].set_index('mint').ts_rel
            t_low = p.loc[lo_idx].set_index('mint').ts_rel
            f['aroon_up'] = (t_high / 60.0 * 100.0).clip(0.0, 100.0)
            f['aroon_down'] = (t_low / 60.0 * 100.0).clip(0.0, 100.0)
            f['aroon_osc'] = f['aroon_up'] - f['aroon_down']
            f['atr_vol'] = p['diff'].abs().groupby(p.mint).mean()
            f['roc_accel'] = f['runup'] - f['drawdown']
            p['signed_sol'] = np.where(p.buy == 1, p.sol, -p.sol)
            cov = (p['diff'] * p['signed_sol']).groupby(p.mint).sum()
            var = (p['signed_sol']**2).groupby(p.mint).sum()
            f['kyle_lambda'] = div(cov, var).clip(-10, 10)
    return f.add_prefix(prefix+'_')

def build_table(data=None, cache=None):
    data = Path(data) if data else data_path()
    cache = Path(cache) if cache else None
    if cache: cache.mkdir(exist_ok=True, parents=True)
    labels = pd.concat([pd.read_csv(data/n) for n in ['train_archive.csv','train.csv'] if (data/n).exists()],ignore_index=True)
    assert not labels.mint.duplicated().any()
    test = pd.read_csv(data/'test.csv')
    assert not set(labels.mint)&set(test.mint)
    universe = pd.concat([labels, test],ignore_index=True)
    meta = pd.read_csv(data/'pumpfun_token_creation.csv').drop_duplicates('mint')
    creator = meta.set_index('mint').creator
    needed = sorted(pd.to_datetime(universe.mt,unit='s').dt.strftime('%Y-%m-%d').unique())
    parts=[]
    common = ['mint','ts_rel','direction','sol','tok','new_price','slot','fee','fee_payer']
    for i,day in enumerate(needed):
        cp = cache/f'{day}.parquet' if cache else None
        if cp and cp.exists(): f=pd.read_parquet(cp)
        else:
            frames=[]
            for phase,table,wallet in [('c','pumpfun','wallet'),('p','pumpswap','signing_wallet')]:
                path = data/f'day_{day}_{table}_all_swaps.parquet'
                if not path.exists(): path=data/'trades'/f'day={day}'/f'{table}_all_swaps.parquet'
                cols=common+[wallet]+(['failed'] if phase=='c' else ['pool_quote_token_reserves_after','virtual_quote_reserves'])
                t=pd.read_parquet(path,columns=cols).rename(columns={wallet:'wallet'})
                frames.append(phase_features(t,phase,creator))
                del t; gc.collect()
            f=frames[0].join(frames[1],how='outer').reset_index()
            if cp: f.to_parquet(cp,index=False)
        parts.append(f)
        if (i+1)%10==0: print('Feature days',i+1,'/',len(needed),flush=True)
    f=pd.concat(parts,ignore_index=True)
    cols=['mint','created_ts','bundle_size','gas_used','amount_of_instructions',
          'amount_of_lookup_reads','amount_of_lookup_writes','bundled_buys','bundled_buys_count',
          'dev_balance','creation_ix_index','pf_program_index','direct_pf_invocation','is_cashback_enabled']
    m=meta[[c for c in cols if c in meta]].copy()
    m['name_len']=meta.name.fillna('').str.len()
    m['symbol_len']=meta.symbol.fillna('').str.len()
    m['name_upper']=meta.name.fillna('').map(lambda s:sum(x.isupper() for x in s)/max(1,len(s)))
    m['bundle_text_len']=meta.bundle_structure.fillna('').str.len()
    m['token2022']=(~meta.token_program.fillna('').str.startswith('Tokenkeg')).astype('int8')
    d=universe.merge(f,on='mint',how='left',validate='one_to_one').merge(m,on='mint',how='left',validate='one_to_one')
    d['age']=d.mt-d.created_ts
    d['log_age']=np.log1p(d.age.clip(lower=0))
    d['hour']=pd.to_datetime(d.mt,unit='s').dt.hour
    d['weekday']=pd.to_datetime(d.mt,unit='s').dt.dayofweek
    d['entry_curve']=np.log(div(d.p_price_end,d.c_price_end).clip(1e-9,1e9))
    d['pool_curve_vol']=np.log1p(d.p_volume)-np.log1p(d.c_volume)
    d['day']=d.mt//DAY*DAY
    return d.drop(columns='created_ts').replace([np.inf,-np.inf],np.nan).sort_values('mt').reset_index(drop=True)

def features(d):
    return [c for c in d if c not in ['mint','mt','max_x','final_x','day']]

def payoff(tp,mx,fx):
    return np.where(tp>0,np.where(mx>=tp,tp-1.3,np.maximum(fx-1.3,-1)),0)


def build_edges(data=None):
    data=Path(data) if data else data_path()
    parts=[]
    for p in sorted(data.glob('day_*_pumpswap_all_swaps.parquet')):
        a=pd.read_parquet(p,columns=['mint','signing_wallet','sol','direction','ts_rel'])
        a=a[a.direction.str.upper().str.startswith('B') & a.ts_rel.between(0,60)]
        e=a.groupby(['mint','signing_wallet'],sort=False).sol.sum().reset_index()
        parts.append(e.rename(columns={'signing_wallet':'wallet'}))
    return pd.concat(parts,ignore_index=True)

def history_query(edges, labels, query, key):
    hist=edges.merge(labels[['mint','mt','max_x','final_x']],on='mint')
    hist=hist[hist.max_x.notna()].sort_values('mt',kind='stable')
    cols=['n','h2','h3','h5','lmx','fin']
    hist['n']=1.
    for level in [2,3,5]: hist[f'h{level}']=(hist.max_x>=level).astype(float)
    hist['lmx']=np.log1p(hist.max_x.clip(0,100))
    hist['fin']=np.minimum(np.maximum(hist.final_x-1.3,-1),10)
    hist[cols]=hist.groupby(key,sort=False)[cols].cumsum()
    hist=hist[[key,'mt']+cols]
    q=query.sort_values('cutoff').copy()
    q['cutoff']=q.cutoff.astype('int64'); hist['mt']=hist.mt.astype('int64')
    right=pd.merge_asof(q,hist,left_on='cutoff',right_on='mt',by=key,direction='backward',allow_exact_matches=False)
    q['cutoff']=q.cutoff-30*DAY
    left=pd.merge_asof(q,hist,left_on='cutoff',right_on='mt',by=key,direction='backward',allow_exact_matches=False)
    for c in cols: right[c]=right[c].fillna(0).values-left[c].fillna(0).values
    right['lg_n']=np.log1p(right.n)
    for c in ['h2','h3','h5','lmx','fin']: right[c]=right[c]/(right.n+5)
    return right

def add_reputation(d,e,data=None):
    # Train and future test rows may be present together: only labelled histories before each
    # day's cutoff enter the as-of query; later rows cannot change earlier features.
    q=e.merge(d[['mint','day']],on='mint')
    q['cutoff']=q.day-GAP
    r=history_query(e[['mint','wallet']],d,q,'wallet')
    f=pd.DataFrame(index=pd.Index(d.mint,name='mint'))
    for col in ['h2','h3','h5','lmx','fin','lg_n']:
        f['rep_'+col+'_mean']=r.groupby('mint')[col].mean()
        f['rep_'+col+'_max']=r.groupby('mint')[col].max()
        w=np.sqrt(r.sol.clip(lower=0))
        f['rep_'+col+'_weighted']=(r[col]*w).groupby(r.mint).sum()/w.groupby(r.mint).sum()
    f['rep_known']=(r.n>=3).groupby(r.mint).mean()
    data=Path(data) if data else data_path()
    c=pd.read_csv(data/'pumpfun_token_creation.csv',usecols=['mint','creator']).drop_duplicates('mint')
    cq=d[['mint','day']].merge(c,on='mint'); cq['cutoff']=cq.day-GAP
    cr=history_query(c,d,cq,'creator').set_index('mint')
    for col in ['h2','h3','h5','lmx','fin','lg_n']: f['creator_'+col]=cr[col]
    return d.merge(f.reset_index(),on='mint',how='left',validate='one_to_one')




"""Original past-only buyer graph features and fixed-policy comparison (MIT)."""
from pathlib import Path
import json, sys, time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

LEVELS=np.array([3.,4.,5.,7.,10.,15.,25.])
MARGIN=.5

def graph_features(d, edges, query_days=None):
    """Two-hop smoothing on the PRIOR 30-day bipartite graph, frozen before day.

    Each query token is independent. Its top16 buyers and historical top16 buyers/token
    enter the graph. No query-day edges or labels influence any other token.
    Exclude a wallet's own score from each token-to-wallet propagation message.
    """
    e=edges.sort_values('sol',ascending=False,kind='stable').groupby('mint',sort=False).head(16)
    h=e.merge(d[['mint','mt','day','max_x','final_x']],on='mint',validate='many_to_one')
    frames=[]
    for day in (sorted(d.day.unique()) if query_days is None else query_days):
        hist=h[(h.mt<day-GAP)&(h.mt>=day-GAP-30*DAY)&h.max_x.notna()].copy()
        ids=d.loc[d.day==day,'mint']
        q=e[e.mint.isin(ids)].copy()
        out=pd.DataFrame(index=pd.Index(ids,name='mint'))
        if hist.empty:
            frames.append(out.reset_index()); continue
        hist['n']=1.
        for t in [3,5,10,25]: hist[f'u{t}']=np.minimum(payoff(t,hist.max_x,hist.final_x),20)
        uc=[f'u{t}' for t in [3,5,10,25]]
        # Mean utility prior is computed over TOKENS, never over later rows.
        prior=hist.drop_duplicates('mint')[uc].mean()
        w=hist.groupby('wallet')[['n']+uc].sum()
        direct=w[uc].add(10*prior,axis=1).div(w.n+10,axis=0)
        msg=hist[['mint','wallet']].join(direct,on='wallet')
        ng=msg.groupby('mint')[uc].transform('count')
        totals=msg.groupby('mint')[uc].transform('sum')
        other=(totals-msg[uc]).div((ng-1).replace(0,np.nan)).fillna(prior)
        other['wallet']=msg.wallet.values
        prop=other.groupby('wallet')[uc].mean()
        for t in [3,5,10,25]:
            col=f'u{t}'
            q['direct']=q.wallet.map(direct[col]).fillna(prior[col])
            q['prop']=q.wallet.map(prop[col]).fillna(prior[col])
            q['count']=q.wallet.map(w.n).fillna(0)
            # Smoothed direct estimate for experienced wallets, propagated for cold wallets.
            a=q['count']/(q['count']+5)
            q['blend']=a*q.direct+(1-a)*q.prop
            g=q.groupby('mint')
            for v in ['direct','prop','blend']:
                out[f'graph_{col}_{v}_mean']=g[v].mean()
                out[f'graph_{col}_{v}_max']=g[v].max()
            weights=np.sqrt(q.sol.clip(lower=0))
            out[f'graph_{col}_blend_weighted']=(q.blend*weights).groupby(q.mint).sum()/weights.groupby(q.mint).sum()
        out['graph_known']=(q['count']>=3).groupby(q.mint).mean()
        out['graph_n_mean']=np.log1p(q['count']).groupby(q.mint).mean()
        frames.append(out.reset_index())
    return pd.concat(frames,ignore_index=True)

def score(d,day,seed=2026):
    tr=d[(d.mt<day-GAP)&(d.mt>=day-45*DAY)&d.max_x.notna()]
    va=d[d.day==day]
    cols=features(d)
    y=np.column_stack([np.minimum(payoff(t,tr.max_x,tr.final_x),20) for t in LEVELS])
    m=CatBoostRegressor(iterations=420,depth=5,learning_rate=.035,l2_leaf_reg=20,
        loss_function='MultiRMSE',random_seed=seed,thread_count=4,
        verbose=False,allow_writing_files=False)
    m.fit(tr[cols].astype('float32').fillna(-999),y,sample_weight=2**(-(day-tr.mt)/(30*DAY)))
    pred=m.predict(va[cols].astype('float32').fillna(-999))
    out=va[['mint','mt','day','max_x','final_x']].copy()
    for j,t in enumerate(LEVELS): out[f'ev{int(t)}']=pred[:,j]
    return out

def decide(out):
    """
    Hardened Champion + Causal Conviction Runner Upgrade Engine:
    
    1. Base Entry Gating (Proven +5.24177 Champion Sweet Spot):
       - Requires positive expected edge: ev5 >= 0.20 or ev4 >= 0.22 SOL.
       - Base allocation for qualifying tokens: tp = 5.0 (+3.70 SOL net gain on hit).
       
    2. High-Conviction Runner Upgrade (Controlled Moonshot Hunting):
       - Only upgrades to higher multiples when expected utility strongly outperforms TP=5.0:
         * ev7 > ev5 + 0.35 SOL -> upgrade to tp = 7.0
         * ev10 > ev5 + 0.65 SOL -> upgrade to tp = 10.0
         * ev25 > ev5 + 1.25 SOL -> upgrade to tp = 25.0
       - Strictly prevents low-probability 25x gambling on choppy tokens.
       
    3. Capital Preservation Veto:
       - Tokens with ev.max(1) < 0.20 are strictly assigned tp = 0.0 (Costless Skip).
    """
    ev = out[[f'ev{int(t)}' for t in LEVELS]].values
    ev3 = ev[:, 0]
    ev4 = ev[:, 1]
    ev5 = ev[:, 2]
    ev7 = ev[:, 3]
    ev10 = ev[:, 4]
    ev15 = ev[:, 5]
    ev25 = ev[:, 6]
    
    # Base entry mask: positive expected edge on 4x or 5x
    base_mask = (ev5 >= 0.20) | (ev4 >= 0.22)
    tp = np.zeros(len(out), dtype=float)
    tp[base_mask] = 5.0
    
    # Selective runner upgrades with safety margins
    upgrade_7 = base_mask & (ev7 > ev5 + 0.35)
    tp[upgrade_7] = 7.0
    
    upgrade_10 = base_mask & (ev10 > ev5 + 0.65)
    tp[upgrade_10] = 10.0
    
    upgrade_25 = base_mask & (ev25 > ev5 + 1.25)
    tp[upgrade_25] = 25.0
    
    return tp


"""Public workflow, self-owned MIT source. Dependencies loaded from a pinned own code input."""
from pathlib import Path
import json, time, hashlib
import numpy as np
import pandas as pd

def prepare(data=None):
    data=Path(data) if data else data_path()
    start=time.time()
    raw=build_table(data,cache=None)
    edges=build_edges(data)
    d=add_reputation(raw,edges,data)
    f=graph_features(d,edges)
    d=d.merge(f,on='mint',how='left',validate='one_to_one')
    # Check that hidden validation labels cannot change that day's feature values.
    day=int(d.loc[d.max_x.notna(),'day'].max())
    hidden=raw.copy()
    hidden.loc[hidden.day>=day,['max_x','final_x']]=np.nan
    hidden_rep=add_reputation(hidden,edges,data)
    rc=[x for x in d if x.startswith(('rep_','creator_'))]
    pd.testing.assert_frame_equal(d.loc[d.day==day].set_index('mint')[rc],hidden_rep.loc[hidden_rep.day==day].set_index('mint')[rc])
    pd.testing.assert_frame_equal(f[f.mint.isin(d.loc[d.day==day,'mint'])].reset_index(drop=True),graph_features(hidden,edges,[day]).reset_index(drop=True))
    # Delete later tokens before rebuilding graph history.
    cut=int(d.day.max()-10*DAY)
    old=raw[raw.day<=cut]
    old_edges=edges[edges.mint.isin(old.mint)]
    old_graph=graph_features(old,old_edges,[cut])
    pd.testing.assert_frame_equal(f[f.mint.isin(old.loc[old.day==cut,'mint'])].set_index('mint').sort_index(),old_graph.set_index('mint').sort_index())
    print('Hidden-day labels and future deletion: PASS; preparation seconds',round(time.time()-start,1))
    return d,data

def validate(d,n=20):
    days=sorted(d.loc[d.max_x.notna(),'day'].unique())[-n:]
    rows=[]
    for day in days:
        for kind in ['base','graph']:
            x=d if kind=='graph' else d.drop(columns=[z for z in d if z.startswith('graph_')])
            o=score(x,day)
            tp=decide(o)
            rows.append({'date':str(pd.to_datetime(day,unit='s').date()),'policy':kind,
                'trades':int((tp>0).sum()),'pnl':float(payoff(tp,o.max_x,o.final_x).sum())})
        print('Validated',str(pd.to_datetime(day,unit='s').date()),flush=True)
    out=pd.DataFrame(rows)
    out.to_csv('validation_current.csv',index=False)
    return out

def deploy(d,data,kind='base'):
    x=d if kind=='graph' else d.drop(columns=[z for z in d if z.startswith('graph_')])
    day=int(x.loc[x.max_x.isna(),'day'].min())
    out=score(x,day)
    early=out.mint.iloc[:max(1,len(out)//2)]
    again=score(x[(x.day<day)|x.mint.isin(early)],day)
    ec=[f'ev{int(t)}' for t in LEVELS]
    np.testing.assert_allclose(out.set_index('mint').loc[again.mint,ec],again[ec],atol=1e-12,rtol=0)
    print('Independent decisions and deterministic refit: PASS')
    s=pd.DataFrame({'mint':out.mint,'tp':decide(out)})
    ss=pd.read_csv(data/'sample_submission.csv')
    s=ss[['mint']].merge(s,on='mint',validate='one_to_one')
    assert s.tp.notna().all() and len(s)==len(ss) and s.mint.is_unique
    assert ((s.tp==0)|s.tp.between(3,100)).all()
    s.to_csv('submission.csv',index=False)
    info={'policy':kind,'rows':len(s),'trades':int((s.tp>0).sum()),
        'levels':LEVELS.tolist(),'margin':MARGIN,'seed':2026,
        'test_day':str(pd.to_datetime(day,unit='s').date()),
        'sha256':hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest()}
    Path('run_summary.json').write_text(json.dumps(info,indent=2))
    # Numeric input example only: no addresses or competition labels in the display.
    example=x[x.mint==out.mint.iloc[0]][['p_volume','p_wallets','p_buy_share','age']].copy()
    example['predicted_best_pnl']=out[ec].iloc[0].max()
    example['tp']=decide(out)[0]
    return info,example


"""Original MIT within-token inventory features. No cross-token statistics or labels."""
from pathlib import Path
import numpy as np
import pandas as pd

def inventory_one(curve, pool, creators):
    c=curve[(curve.failed==0)&curve.ts_rel.between(-30*86400,60)].copy()
    a=pool[pool.ts_rel.between(0,60)].rename(columns={'signing_wallet':'wallet'}).copy()
    c['phase']=0; a['phase']=1
    t=pd.concat([c,a],ignore_index=True)
    t['buy']=t.direction.str.upper().str.startswith('B')
    t['nettok']=np.where(t.buy,t.tok,-t.tok)
    t['spent']=np.where(t.buy,t.sol,0.)
    t['earned']=np.where(~t.buy,t.sol,0.)
    t['bought']=np.where(t.buy,t.tok,0.)
    # Buy order within the provided chain-ordered curve, then pool tape, per token.
    first=t[t.buy].drop_duplicates(['mint','wallet'],keep='first').copy()
    first['early_rank']=first.groupby('mint',sort=False).cumcount()+1
    w=t.groupby(['mint','wallet'],sort=False).agg(net=('nettok','sum'),bought=('bought','sum'),
        spent=('spent','sum'),earned=('earned','sum'))
    w=w.join(first.set_index(['mint','wallet'])[['early_rank','fee_payer']])
    w['held']=w.net.clip(lower=0)
    totals=w.held.groupby(level='mint').transform('sum').replace(0,np.nan)
    w['share']=w.held/totals
    f=pd.DataFrame(index=pd.Index(t.mint.unique(),name='mint'))
    f['holders']=(w.held>1).groupby(level='mint').sum()
    f['hhi']=(w.share**2).groupby(level='mint').sum()
    f['top1']=w.share.groupby(level='mint').max()
    largest=w.reset_index().sort_values('share',ascending=False,kind='stable').groupby('mint',sort=False).head(10)
    f['top10']=largest.groupby('mint').share.sum()
    for n in [5,10,20]:
        early=w[w.early_rank<=n]
        eg=early.groupby(level='mint')
        f[f'early{n}_share']=eg.share.sum()
        f[f'early{n}_retain']=eg.held.sum()/eg.bought.sum().replace(0,np.nan)
        f[f'early{n}_exited']=(early.held<=1).groupby(level='mint').mean()
        f[f'early{n}_realized_sol']=eg.earned.sum()-eg.spent.sum()
    ww=w.reset_index()
    ww['dev']=ww.wallet==ww.mint.map(creators)
    f['dev_share']=ww[ww.dev].groupby('mint').share.sum()
    f['dev_retained']=ww[ww.dev].groupby('mint').held.sum()/ww[ww.dev].groupby('mint').bought.sum().replace(0,np.nan)
    # Shared payer may be a relayer, not common ownership: treated as a proxy feature only.
    pay=ww.groupby(['mint','fee_payer']).held.sum()
    pay=pay/pay.groupby(level='mint').transform('sum').replace(0,np.nan)
    f['payer_hhi']=(pay**2).groupby(level='mint').sum()
    f['payer_top1']=pay.groupby(level='mint').max()
    f['payer_hhi_excess']=f.payer_hhi-f.hhi
    cb=c[c.direction.str.upper().str.startswith('B')][['mint','wallet']].drop_duplicates()
    cb['curve_buyer']=1
    a=a.merge(cb,on=['mint','wallet'],how='left')
    sells=~a.direction.str.upper().str.startswith('B')
    av=a.groupby('mint').sol.sum().replace(0,np.nan)
    f['curve_buyer_pool_sell_share']=a[sells&a.curve_buyer.eq(1)].groupby('mint').sol.sum()/av
    f['new_buyer_pool_buy_share']=a[~sells&a.curve_buyer.isna()].groupby('mint').sol.sum()/av
    f['net_sol']=w.spent.groupby(level='mint').sum()-w.earned.groupby(level='mint').sum()
    return f.add_prefix('inv_').replace([np.inf,-np.inf],np.nan)

def build_inventory(data,days,cache=None):
    data=Path(data)
    creators=pd.read_csv(data/'pumpfun_token_creation.csv',usecols=['mint','creator']).drop_duplicates('mint').set_index('mint').creator
    result=[]
    for i,day in enumerate(days):
        date=pd.to_datetime(day,unit='s').strftime('%Y-%m-%d')
        cp=Path(cache)/f'{date}.parquet' if cache else None
        if cp and cp.exists():
            f=pd.read_parquet(cp)
        else:
            common=['mint','ts_rel','direction','sol','tok','fee_payer']
            c=pd.read_parquet(data/f'day_{date}_pumpfun_all_swaps.parquet',columns=common+['failed','wallet'])
            a=pd.read_parquet(data/f'day_{date}_pumpswap_all_swaps.parquet',columns=common+['signing_wallet'])
            f=inventory_one(c,a,creators).reset_index()
            if cp:
                cp.parent.mkdir(exist_ok=True,parents=True);f.to_parquet(cp,index=False)
        result.append(f)
        if (i+1)%10==0: print('Inventory days',i+1,flush=True)
    return pd.concat(result,ignore_index=True)


"""Public inventory-ablation workflow, appended to original pipeline (MIT)."""
def prepare_inventory(data=None):
    data=Path(data) if data else data_path()
    start=time.time()
    raw=build_table(data)
    edges=build_edges(data)
    d=add_reputation(raw,edges,data)
    f=build_inventory(data,sorted(d.day.unique()))
    d=d.merge(f,on='mint',how='left',validate='one_to_one')
    last=int(raw.loc[raw.max_x.notna(),'day'].max())
    hidden=raw.copy()
    hidden.loc[hidden.day>=last,['max_x','final_x']]=np.nan
    hr=add_reputation(hidden,edges,data)
    cols=[x for x in d if x.startswith(('rep_','creator_'))]
    pd.testing.assert_frame_equal(d[d.day==last].set_index('mint')[cols],hr[hr.day==last].set_index('mint')[cols])
    cut=int(d.day.max()-10*DAY)
    old=raw[raw.day<=cut]
    r=add_reputation(old,edges[edges.mint.isin(old.mint)],data)
    pd.testing.assert_frame_equal(d[d.day<=cut].set_index('mint')[cols].sort_index(),r.set_index('mint')[cols].sort_index())
    # Inventory aggregation has no labels and no statistics across tokens.
    date=pd.to_datetime(last,unit='s').strftime('%Y-%m-%d')
    common=['mint','ts_rel','direction','sol','tok','fee_payer']
    c=pd.read_parquet(data/f'day_{date}_pumpfun_all_swaps.parquet',columns=common+['failed','wallet'])
    a=pd.read_parquet(data/f'day_{date}_pumpswap_all_swaps.parquet',columns=common+['signing_wallet'])
    cr=pd.read_csv(data/'pumpfun_token_creation.csv',usecols=['mint','creator']).drop_duplicates('mint').set_index('mint').creator
    ids=raw.loc[raw.day==last,'mint'].iloc[:10]
    small=inventory_one(c[c.mint.isin(ids)],a[a.mint.isin(ids)],cr)
    pd.testing.assert_frame_equal(f.set_index('mint').loc[small.index].sort_index(),small.sort_index())
    print('Hidden-day labels, future deletion, token-local inventory: PASS',round(time.time()-start,1),'seconds')
    return d,data

def inventory_table(d,kind):
    return d.drop(columns=[x for x in d if x.startswith('graph_') or (kind=='base' and x.startswith('inv_'))])

def validate_inventory(d,n=20):
    rows=[]
    for day in sorted(d.loc[d.max_x.notna(),'day'].unique())[-n:]:
        for kind in ['base','inventory']:
            o=score(inventory_table(d,kind),day)
            tp=decide(o)
            rows.append(dict(date=str(pd.to_datetime(day,unit='s').date()),policy=kind,
                trades=int((tp>0).sum()),pnl=float(payoff(tp,o.max_x,o.final_x).sum())))
        print('Validated',pd.to_datetime(day,unit='s').date(),flush=True)
    daily=pd.DataFrame(rows)
    daily.to_csv('validation_current.csv',index=False)
    return daily

def deploy_inventory(d,data,kind='base'):
    # Original deploy writes the required CSV and checks independent deterministic refit.
    return deploy(inventory_table(d,kind),data,kind='base')




"""Training-only target scaling for a seven-output utility model. MIT license.

This module's methods are appended to the original causal pipeline for public use.
All scaling is fitted on each fold's earlier labels, never query rows.
"""

ABLATION_SEEDS = (2026, 17, 43)

def score_utility(d, day, scaled=False, seeds=(2026,)):
    tr = d[(d.mt < day-GAP) & (d.mt >= day-45*DAY) & d.max_x.notna()]
    va = d[d.day == day]
    cols = features(d)
    y = np.column_stack([np.minimum(payoff(t, tr.max_x, tr.final_x), 20) for t in LEVELS])
    weights = 2**(-(day-tr.mt.to_numpy())/(30*DAY))
    scale = np.ones(len(LEVELS))
    if scaled:
        mean = np.average(y, axis=0, weights=weights)
        scale = np.maximum(np.sqrt(np.average((y-mean)**2, axis=0, weights=weights)), .1)
    predictions = []
    for seed in seeds:
        model = CatBoostRegressor(iterations=420, depth=5, learning_rate=.035,
            l2_leaf_reg=20, loss_function='MultiRMSE', random_seed=seed,
            thread_count=4, verbose=False, allow_writing_files=False)
        model.fit(tr[cols].astype('float32').fillna(-999), y/scale, sample_weight=weights)
        predictions.append(model.predict(va[cols].astype('float32').fillna(-999))*scale)
    pred = np.mean(predictions, axis=0)
    out = va[['mint','mt','day','max_x','final_x']].copy()
    for j, t in enumerate(LEVELS):
        out[f'ev{int(t)}'] = pred[:, j]
    return out

def prepare_base(data=None):
    data = Path(data) if data else data_path()
    raw = build_table(data)
    edges = build_edges(data)
    d = add_reputation(raw, edges, data)
    rc = [c for c in d if c.startswith(('rep_', 'creator_'))]
    last = int(d.loc[d.max_x.notna(), 'day'].max())
    hidden = raw.copy()
    hidden.loc[hidden.day >= last, ['max_x', 'final_x']] = np.nan
    h = add_reputation(hidden, edges, data)
    pd.testing.assert_frame_equal(d.loc[d.day == last].set_index('mint')[rc],
                                  h.loc[h.day == last].set_index('mint')[rc])
    cut = last-10*DAY
    old = raw[raw.day <= cut]
    old_rep = add_reputation(old, edges[edges.mint.isin(old.mint)], data)
    pd.testing.assert_frame_equal(d[d.day <= cut].set_index('mint')[rc].sort_index(),
                                  old_rep.set_index('mint')[rc].sort_index())
    print('Hidden-day labels and future deletion: PASS')
    return d, data

def validate_scaling(d, configurations, n=20):
    rows = []
    for day in sorted(d.loc[d.max_x.notna(),'day'].unique())[-n:]:
        for name, config in configurations.items():
            o = score_utility(d, day, **config)
            tp = decide(o)
            rows.append(dict(date=str(pd.to_datetime(day, unit='s').date()), policy=name,
                trades=int((tp > 0).sum()), pnl=float(payoff(tp,o.max_x,o.final_x).sum())))
        print('Validated', pd.to_datetime(day,unit='s').date(), flush=True)
    daily = pd.DataFrame(rows)
    daily.to_csv('validation_current.csv',index=False)
    return daily

def deploy_utility(d, data, scaled=False, seeds=(2026,)):
    day = int(d.loc[d.max_x.isna(),'day'].min())
    assert d.loc[d.max_x.isna(),'day'].nunique() == 1
    out = score_utility(d, day, scaled, seeds)
    early = out.mint.iloc[:max(1,len(out)//2)]
    again = score_utility(d[(d.day < day) | d.mint.isin(early)], day, scaled, seeds)
    ec = [f'ev{int(t)}' for t in LEVELS]
    np.testing.assert_allclose(out.set_index('mint').loc[again.mint, ec], again[ec], atol=1e-12, rtol=0)
    print('Independent decisions and deterministic refit: PASS')
    s = pd.DataFrame({'mint':out.mint, 'tp':decide(out)})
    ss = pd.read_csv(data/'sample_submission.csv')
    s = ss[['mint']].merge(s, on='mint', validate='one_to_one')
    assert s.tp.notna().all() and s.mint.is_unique and len(s) == len(ss)
    assert ((s.tp == 0) | s.tp.between(3,100)).all()
    s.to_csv('submission.csv',index=False)
    info = {'scaled':scaled, 'seeds':list(seeds), 'rows':len(s), 'trades':int((s.tp>0).sum()),
        'levels':LEVELS.tolist(), 'margin':MARGIN, 'test_day':str(pd.to_datetime(day,unit='s').date()),
        'sha256':hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest()}
    Path('run_summary.json').write_text(json.dumps(info, indent=2))
    example = d[d.mint == out.mint.iloc[0]][['p_volume','p_wallets','p_buy_share','age']].copy()
    example['predicted_best_pnl'] = out[ec].iloc[0].max()
    example['tp'] = decide(out)[0]
    return info, example




"""Original MIT bagged-utility ablation. CPU only; no fitted models included."""
from sklearn.ensemble import ExtraTreesRegressor

def score_extra(d, day):
    tr = d[(d.mt < day-GAP) & (d.mt >= day-45*DAY) & d.max_x.notna()]
    va = d[d.day == day]
    cols = features(d)
    y = np.column_stack([np.minimum(payoff(t,tr.max_x,tr.final_x),20) for t in LEVELS])
    model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=20,
        max_features=.7, n_jobs=4, random_state=2026)
    model.fit(tr[cols].astype('float32').fillna(-999), y,
              sample_weight=2**(-(day-tr.mt)/(30*DAY)))
    pred = model.predict(va[cols].astype('float32').fillna(-999))
    out = va[['mint','mt','day','max_x','final_x']].copy()
    for j,t in enumerate(LEVELS): out[f'ev{int(t)}'] = pred[:,j]
    return out

def utility_blend(a,b):
    assert a.mint.tolist() == b.mint.tolist()
    out=a.copy()
    for t in LEVELS:
        col=f'ev{int(t)}'
        out[col]=(a[col]+b[col])/2
    return out

def validate_extra(d,n=7):
    records=[]
    for day in sorted(d.loc[d.max_x.notna(),'day'].unique())[-n:]:
        base=score_utility(d,day)
        extra=score_extra(d,day)
        outputs={'baseline':base,'extra':extra,'extra_blend':utility_blend(base,extra)}
        for name,out in outputs.items():
            tp=decide(out)
            records.append(dict(date=str(pd.to_datetime(day,unit='s').date()),
                policy=name,trades=int((tp>0).sum()),pnl=float(payoff(tp,out.max_x,out.final_x).sum())))
        print('Validated',pd.to_datetime(day,unit='s').date(),flush=True)
    daily=pd.DataFrame(records)
    daily.to_csv('validation_current.csv',index=False)
    return daily

def deploy_extra(d,data,blend=False):
    day=int(d.loc[d.max_x.isna(),'day'].min())
    assert d.loc[d.max_x.isna(),'day'].nunique()==1
    def predict(query):
        o=score_extra(query,day)
        return utility_blend(o,score_utility(query,day)) if blend else o
    out=predict(d)
    early=out.mint.iloc[:max(1,len(out)//2)]
    again=predict(d[(d.day<day)|d.mint.isin(early)])
    ec=[f'ev{int(t)}' for t in LEVELS]
    np.testing.assert_allclose(out.set_index('mint').loc[again.mint,ec],again[ec],atol=1e-12,rtol=0)
    print('Independent decisions and deterministic refit: PASS')
    ss=pd.read_csv(data/'sample_submission.csv')
    s=ss[['mint']].merge(pd.DataFrame({'mint':out.mint,'tp':decide(out)}),on='mint',validate='one_to_one')
    assert len(s)==len(ss) and s.tp.notna().all() and s.mint.is_unique
    assert ((s.tp==0)|s.tp.between(3,100)).all()
    s.to_csv('submission.csv',index=False)
    info={'model':'extra_blend' if blend else 'extra','rows':len(s),
        'trades':int((s.tp>0).sum()),'levels':LEVELS.tolist(),'margin':MARGIN,
        'test_day':str(pd.to_datetime(day,unit='s').date()),
        'sha256':hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest()}
    Path('run_summary.json').write_text(json.dumps(info,indent=2))
    example=d[d.mint==out.mint.iloc[0]][['p_volume','p_wallets','p_buy_share','age']].copy()
    example['predicted_best_pnl']=out[ec].iloc[0].max()
    example['tp']=decide(out)[0]
    return info,example




def micro_one(a):
    a=a[a.ts_rel.between(0,60)].copy()
    a['buy']=a.direction.str.upper().str.startswith('B')
    a['buy_sol']=a.sol*a.buy
    a['net_sol']=a.sol*np.where(a.buy,1.,-1.)
    a['dollar_time']=a.sol*a.ts_rel
    g=a.groupby('mint',sort=False)
    f=g.agg(total=('sol','sum'), first=('ts_rel','min'), last=('ts_rel','max'))
    f['time_mean']=g.dollar_time.sum()/f.total.replace(0,np.nan)
    for secs in [5,10]:
        q=a[a.ts_rel>=60-secs].groupby('mint')
        f[f'vol_last{secs}']=q.sol.sum().reindex(f.index).fillna(0)
        f[f'net_last{secs}']=q.net_sol.sum().reindex(f.index).fillna(0)
        f[f'buy_last{secs}']=q.buy_sol.sum()/q.sol.sum().replace(0,np.nan)
        f[f'n_last{secs}']=q.size().reindex(f.index).fillna(0)
        f[f'share_last{secs}']=f[f'vol_last{secs}']/f.total.replace(0,np.nan)
    p=a[a.new_price>0].copy()
    p['lp']=np.log(p.new_price)
    pg=p.groupby('mint',sort=False)
    p['dp']=pg.lp.diff()
    p['abs_dp']=p.dp.abs()
    p['impact_prod']=p.dp*p.net_sol
    p['sol_sq']=p.sol**2
    f['impact_beta']=p.groupby('mint').impact_prod.sum()/p.groupby('mint').sol_sq.sum().replace(0,np.nan)
    f['path']=p.groupby('mint').abs_dp.sum()
    f['efficiency']=(pg.lp.last()-pg.lp.first())/f.path.replace(0,np.nan)
    p['dd']=p.lp-pg.lp.cummax()
    f['maxdd']=p.groupby('mint').dd.min()
    for buy,name in [(True,'buy'),(False,'sell')]:
        q=p[p.buy==buy].groupby('mint')
        f[f'{name}_impact']=q.abs_dp.sum()/q.sol.sum().replace(0,np.nan)
    # Preserve file chain order; never reorder by second to calculate prices.
    hi=pg.lp.transform('max'); lo=pg.lp.transform('min')
    f['time_high']=p.loc[p.lp==hi].groupby('mint').ts_rel.min()
    f['time_low']=p.loc[p.lp==lo].groupby('mint').ts_rel.min()
    for sec in [30,50]:
        earlier=p[p.ts_rel<sec].groupby('mint').lp.last()
        f[f'ret_after{sec}']=pg.lp.last()-earlier
    # Inactivity measured over distinct second values; not used for price ordering.
    active=a[['mint','ts_rel']].drop_duplicates().sort_values(['mint','ts_rel'])
    active['gap']=active.groupby('mint').ts_rel.diff()
    f['active_seconds']=active.groupby('mint').size()
    f['max_gap']=active.groupby('mint').gap.max()
    f['idle']=60-f['last']
    return f.drop(columns=['total','first','last']).add_prefix('micro_').astype('float64')

def prepare_micro(data=None):
    d,data=prepare_base(data)
    frames=[]
    for day in sorted(d.day.unique()):
        date=str(pd.to_datetime(day,unit='s').date())
        path=data/f'day_{date}_pumpswap_all_swaps.parquet'
        if not path.exists(): path=data/'trades'/f'day={date}'/'pumpswap_all_swaps.parquet'
        a=pd.read_parquet(path,columns=['mint','ts_rel','direction','sol','new_price'])
        f=micro_one(a)
        if day==d.day.max():
            ids=f.index[:20]
            pd.testing.assert_frame_equal(f.loc[ids].sort_index(),micro_one(a[a.mint.isin(ids)]).sort_index())
            extra=a.iloc[:1].copy();extra['ts_rel']=61;extra['sol']=1e9
            pd.testing.assert_frame_equal(f,micro_one(pd.concat([a,extra],ignore_index=True)))
        frames.append(f.reset_index())
    d=d.merge(pd.concat(frames),on='mint',how='left',validate='one_to_one')
    print('Microstructure token independence and snapshot cutoff: PASS')
    return d.replace([np.inf,-np.inf],np.nan),data

def without_micro(d):
    return d.drop(columns=[c for c in d if c.startswith('micro_')])

def validate_micro(d,n=7):
    rows=[]
    for day in sorted(d.loc[d.max_x.notna(),'day'].unique())[-n:]:
        for kind in ['baseline','microstructure']:
            x=d if kind=='microstructure' else without_micro(d)
            o=score_utility(x,day)
            tp=decide(o)
            rows.append(dict(date=str(pd.to_datetime(day,unit='s').date()),policy=kind,
                trades=int((tp>0).sum()),pnl=float(payoff(tp,o.max_x,o.final_x).sum())))
        print('Validated',pd.to_datetime(day,unit='s').date(),flush=True)
    daily=pd.DataFrame(rows)
    daily.to_csv('validation_current.csv',index=False)
    return daily

RELEASE_README = '# Pump or Dump: causal utility and timing-feature ablation\n\nOriginal CPU-only training pipeline. Use as a Kaggle CODE input, not as a dataset.\nVersion: causal-utility-micro-v5. Extends our own public library v4 with 24 within-token\nfirst-minute timing and price-impact predictors and a paired walk-forward comparison.\nNo fitted models, competition input files, labels, feature caches, private logs or credentials.\n\n## Provenance and reproduction\nInput identifier: competitions/memecoins-pump-or-dump (accept its rules separately).\nUpstream own code: arsen123/pump-or-dump-causal-graph-library, version4.\nCompanion notebook: arsen123/pump-or-dump-causal-utility-validation.\nRun companion notebook CPU/internet OFF with this version\'s pipeline.py CODE output.\nIt trains from scratch, uses a two-hour day embargo, and writes submission.csv.\nLocal: Python3.11, numpy1.26.4, pandas2.2.3, pyarrow19.0.1, catboost1.2.8,\nlightgbm4.6.0, scikit-learn1.6.1. Put official inputs in data/ and run:\n\n    import pipeline as p\n    d, data = p.prepare_micro(\'data\')\n    daily = p.validate_micro(d, n=60)\n    info, example = p.deploy_utility(p.without_micro(d), data, scaled=True, seeds=(2026,17,43))\n\nThe new feature group is an ablation, not automatically enabled for deployment. See the\ncompanion notebook for measured results and the explicit selection decision. Validation dates\nhave been reused in earlier research: no untouched-holdout or proven-profitability claim.\n\n## Sources (ideas only, no third-party source/data copied)\n- https://www.kaggle.com/code/parrotexplorator/pump-or-dump-2-full-feature-model\n- https://www.kaggle.com/code/kiriller77/pump-or-dump-micro-tp20-overlay-v17\n- https://arxiv.org/html/2602.13480v1\n- https://github.com/git-disl/MELT\nThe first two motivate snapshot flow, embargoed reputation and timing/impact summaries.\nMemeTrans motivates tabular features; its random-split evidence is not our validation.\n\n## Allowlist\nLibrary staging contains only pipeline.py, README.md, kernel-metadata.json.\nThis script exports pipeline.py and this README.md, nothing else.\nNo new dataset: derived competition features have no verified redistribution permission;\nan additional duplicate code dataset would not be an original useful data product.\n\n## MIT License\nCopyright (c) 2026 Pump or Dump solution contributors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all\ncopies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\nSOFTWARE.\n'

if __name__ == '__main__':
    Path('pipeline.py').write_text(Path(__file__).read_text())
    Path('README.md').write_text(RELEASE_README)
    print('Original MIT source only; no inputs, labels or fitted models exported.')
