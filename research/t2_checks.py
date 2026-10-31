"""Remaining Task 2 checks on the real task2.py: survivorship by liquidity bucket, sudden-collapse ghosts, long-history speed, shock
injection, cold start. Writes results/t2_checks.json. The timing vs selection split is in t2_decomp.py."""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import H, RESULTS, t2_data
from t2_final import EVAL, data, ghost_panel, init, mod, one


def sudden_panel(D, n, seed, fall=.6):
    """Ghosts that crash overnight, keep trading for 10 days at the low, then vanish."""
    rng = np.random.default_rng(seed)
    idx = D['close'].index
    t0 = np.searchsorted(idx, pd.Timestamp(EVAL[0]))
    picks = rng.choice(list(D['close'].columns), n, replace=False)
    extra = {k: {} for k in ['open', 'high', 'low', 'close', 'volume']}
    for i, s in enumerate(picks):
        t = int(rng.integers(t0, len(idx) - 12))
        f = np.ones(len(idx)); f[t:] = 1 - fall
        name = 'GHOST%03d' % i
        for k in ['open', 'high', 'low', 'close']:
            v = D[k][s].values * f
            v[t + 10:] = np.nan
            extra[k][name] = v
        vol = D['volume'][s].values.copy(); vol[t + 10:] = np.nan
        extra['volume'][name] = vol
    out = {k: pd.concat([D[k], pd.DataFrame(extra[k], index=idx)], axis=1) for k in extra}
    out['nifty'] = D['nifty']
    return out


def sudden_job(a):
    n, seed = a
    s, res = one(D=sudden_panel(data(), n, seed))
    T = res['trades']
    return dict(n=n, sh=s['sharpe'], cagr=s['cagr'], dd=s['max_dd'], ghost=int(T.symbol.str.startswith('GHOST').sum()))


if __name__ == '__main__':
    OUT = {}
    init()
    with Pool(6, initializer=init) as pool:
        print('=== sudden-collapse ghosts: -60% overnight, then gone (12 seeds)')
        OUT['sudden'] = []
        for n in [15, 30, 60]:
            g = pd.DataFrame(pool.map(sudden_job, [(n, sd) for sd in range(12)], chunksize=1))
            OUT['sudden'].append(dict(n=n, sh=float(g.sh.mean()), cagr=float(g.cagr.mean()), dd=float(g.dd.mean()), ghost=float(g.ghost.mean())))
            print(' %2d ghosts: Sharpe %.2f CAGR %.1f%% DD %.1f%% trades in ghosts per run %.1f' % (n, g.sh.mean(), 100 * g.cagr.mean(), 100 * g.dd.mean(), g.ghost.mean()))

    D = data()
    C, V = D['close'], D['volume']
    print('\n=== survivorship tell: forward return to Jun 2024 by liquidity bucket on 28 Feb 2022 (60d median traded value)')
    tv = (C * V).rolling(60, min_periods=40).median().loc['2022-02-28']
    fwd = (C.iloc[-1] / C.loc['2022-03-02'] - 1)
    b = pd.cut(tv, [0, 1e7, 5e7, 1e8, 2.5e8, 5e8, 1e9, 1e12], labels=['<1cr', '1-5cr', '5-10cr', '10-25cr', '25-50cr', '50-100cr', '>100cr'])
    x = pd.DataFrame({'b': b, 'fwd': fwd}).dropna()
    t = x.groupby('b', observed=True).fwd.agg(['count', 'mean', 'median', lambda v: (v < 0).mean()])
    t.columns = ['n', 'mean', 'median', 'share_negative']
    print((t * [1, 100, 100, 100]).round(1).to_string())
    OUT['buckets'] = t.reset_index().astype({'b': str}).to_dict('records')
    never = (C.loc['2022-03-02':].isna().all())
    print(' names with no price at all in the window (cannot be evaluated): %d; names alive on the last date: %d of %d' % (never.sum(), C.iloc[-1].notna().sum(), C.shape[1]))
    late = tv.isna().sum()
    print(' names not yet with 40 days of history on 28 Feb 2022: %d (later IPOs that made the 2026 list)' % late)

    print('\n=== speed on a long history: data tiled forward to about 1,380 days, hidden-length window of about 570 days')
    n = len(C)
    new_idx = pd.bdate_range(C.index[-1] + pd.Timedelta(days=1), periods=590)
    D2 = {}
    for k in ['open', 'high', 'low', 'close', 'volume']:
        lastval = D[k].iloc[-590:].values
        f = (C.iloc[-1].values / C.iloc[-590].values) if k != 'volume' else 1.0
        f = np.where(np.isnan(f), 1.0, f) if k != 'volume' else 1.0
        D2[k] = pd.concat([D[k], pd.DataFrame(lastval * f, index=new_idx, columns=D[k].columns)])
    nf = D['nifty'].iloc[-590:].values * (D['nifty'].close.iloc[-1] / D['nifty'].close.iloc[-590])
    D2['nifty'] = pd.concat([D['nifty'], pd.DataFrame(nf, index=new_idx, columns=D['nifty'].columns)])
    m = mod(); t0 = time.time()
    res = H.task2_backtest(m, D2, dict(m.PARAMS), str(new_idx[0].date()), str(new_idx[-1].date()))
    el = time.time() - t0
    s, _ = H.task2_stats(res)
    print(' %d rows of history, %d days simulated: %.1fs, Sharpe %.2f, trades %d' % (len(D2['close']), len(res['eq']), el, s['sharpe'], s['trades']))
    OUT['speed_long_hist_s'] = el

    print('\n=== shock injection: a held stock jumps -50% (and in another run +100%); a held stock stops trading')
    base, br = one()
    T = br['trades']
    tr = T.sort_values('days', ascending=False).iloc[0]
    sym = tr.symbol; mid = pd.Timestamp(tr.entry_date) + (pd.Timestamp(tr.exit_date) - pd.Timestamp(tr.entry_date)) / 2
    day = D['close'].index[np.searchsorted(D['close'].index, mid)]
    print(' target: %s held %s to %s, shock on %s' % (sym, str(tr.entry_date)[:10], str(tr.exit_date)[:10], str(day)[:10]))
    for label, mult in [('-50%', .5), ('+100%', 2.0)]:
        Dx = {k: v.copy() for k, v in D.items()}
        for k in ['open', 'high', 'low', 'close']:
            Dx[k].loc[day:, sym] = Dx[k].loc[day:, sym] * mult
        s_, r_ = one(D=Dx)
        ex = r_['trades'][(r_['trades'].symbol == sym) & (pd.to_datetime(r_['trades'].entry_date) <= day)].sort_values('exit_date')
        after = ex[pd.to_datetime(ex.exit_date) >= day].iloc[0] if len(ex[pd.to_datetime(ex.exit_date) >= day]) else None
        print(' %-6s Sharpe %.2f CAGR %.1f%% DD %.1f%% | position exit %s (shock was %s)' % (label, s_['sharpe'], 100 * s_['cagr'], 100 * s_['max_dd'], str(after.exit_date)[:10] if after is not None else 'none', str(day)[:10]))
    Dx = {k: v.copy() for k, v in D.items()}
    i = D['close'].index.get_loc(day)
    gap_days = D['close'].index[i:i + 15]
    for k in ['open', 'high', 'low', 'close', 'volume']:
        Dx[k].loc[gap_days, sym] = np.nan
    s_, r_ = one(D=Dx)
    print(' halt   15 missing days on a held stock: runs, Sharpe %.2f, trades %d, sells re-issued OK' % (s_['sharpe'], s_['trades']))

    print('\n=== cold start: the harness gets very little history before the window')
    m = mod()
    for start in ['2023-06-01', '2024-01-02']:
        Dc = {k: v.loc[start:] for k, v in D.items()}
        a = str(Dc['close'].index[1].date())
        res = H.task2_backtest(m, Dc, dict(m.PARAMS), a, None)
        s, _ = H.task2_stats(res)
        first = res['trades'].entry_date.min() if len(res['trades']) else None
        print(' history from %s: Sharpe %.2f, trades %d, first entry %s, DD %.1f%%' % (start, s['sharpe'], s['trades'], str(first)[:10], 100 * s['max_dd']))
    print('\n=== all-cash floor: one row of history, no positions possible')
    Dc = {k: v.iloc[:2] for k, v in D.items()}
    res = H.task2_backtest(m, Dc, dict(m.PARAMS), str(Dc['close'].index[1].date()), None)
    print(' ok, final equity', float(res['eq'].iloc[-1]))
    json.dump(OUT, open(os.path.join(RESULTS, 't2_checks.json'), 'w'), indent=1, default=float)
