"""Attack the breadth filter. It looks too good, so try to break it:
 1. Timing only: apply it to the plain equal-weight universe, with no stock selection at all.
 2. Threshold curve: plateau or spike?
 3. Signal delay: does it die if the regime signal arrives 1, 2 or 5 days late?
 4. Cross-validation by names: measure breadth on half the universe, trade the other half.
 5. Other breadth windows and simple alternatives.
 6. Next-day return by breadth bucket, to see the mechanism.
All portfolio numbers are means over 4 rebalance offsets."""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import H
from t2_lab import BASE, RESULTS, make_orders, run, world, features
import t2_lab
import types

REF = dict(BASE, score='sh126', every=10, buffer=40, off='sell', breakout='hi10', extra=10, fill_daily=True, maxvol=.65)


def ew_timing():
    w = world(); D = w['D']; F, B, cols = w['X']
    C = D['close']; O = D['open']
    ret = C.pct_change(fill_method=None)
    clean = ret.where((ret.abs() < .45) | ret.isna(), 0.0)
    liqm = pd.DataFrame(F['liq'], index=C.index, columns=C.columns) >= 1e8
    ew = clean.where(liqm.shift(1)).mean(axis=1)
    br = pd.Series(B['breadth50'], index=C.index)
    win = {'eval': ('2022-03-01', '2024-06-30'), 'warm': ('2021-10-15', '2022-02-28')}
    print('=== 1. plain EW universe (close to close, signal at t applies to the return of t+1, no costs)')
    print(' eval buy&hold: Sharpe %.2f CAGR %.1f%% DD %.1f%%' % stats(ew.loc[win['eval'][0]:win['eval'][1]]))
    for th in [.3, .4, .5, .55, .6, .65, .7, .8]:
        on = (br.shift(1) >= th).astype(float)
        e = (ew * on).loc[win['eval'][0]:win['eval'][1]]
        wr = (ew * on).loc[win['warm'][0]:win['warm'][1]]
        sw = on.diff().abs().loc[win['eval'][0]:win['eval'][1]].sum() / 2.33
        print(' breadth >= %.2f: eval Sharpe %.2f CAGR %5.1f%% DD %6.1f%% in-mkt %3.0f%% switches/yr %3.0f | warm total %6.1f%%' % (th, *stats(e), 100 * on.loc[win['eval'][0]:win['eval'][1]].mean(), sw, 100 * ((1 + wr).prod() - 1)))
    print('\n=== 6. next-day EW return by breadth bucket (bps, t), eval and warm')
    nxt = ew.shift(-1)
    x = pd.DataFrame({'b': br, 'n': nxt}).dropna()
    for nm, (a, z) in win.items():
        s = x.loc[a:z]
        g = s.groupby(pd.cut(s.b, [0, .3, .4, .5, .6, .7, .8, 1.0]), observed=True).n
        print(nm); print((pd.DataFrame({'bps': g.mean() * 1e4, 't': g.mean() / g.std() * g.count() ** .5, 'n': g.count()})).round(1).to_string())
    print(' autocorrelation of daily EW returns, eval: lag1 %.3f lag2 %.3f lag5 %.3f; Nifty lag1 %.3f' % (ew.loc[win['eval'][0]:win['eval'][1]].autocorr(1), ew.loc[win['eval'][0]:win['eval'][1]].autocorr(2), ew.loc[win['eval'][0]:win['eval'][1]].autocorr(5), D['nifty'].close.pct_change().loc[win['eval'][0]:win['eval'][1]].autocorr(1)))


def stats(r):
    eq = (1 + r.fillna(0)).cumprod()
    yrs = len(r) / 252
    return (r.mean() / r.std() * 252 ** .5, 100 * (eq.iloc[-1] ** (1 / yrs) - 1), 100 * (eq / eq.cummax() - 1).min())


def mean_run(p, which='eval'):
    e = p['every']; v = []
    for o in [0, e // 4, e // 2, 3 * e // 4]:
        s, _ = run(dict(p, offset=o), which)
        v.append((s['sharpe'], s['max_dd'], s['cagr'], s['trades_per_yr'], s['time_in_market'], s['total_return']))
    return np.array(v).mean(axis=0)


def job(a):
    name, p = a
    e_, t_, w_ = mean_run(p), mean_run(p, 'tough'), mean_run(p, 'warm')
    return name, e_, t_, w_


def line(r):
    n, e, t, w = r
    return '%-30s eval Sh %5.2f DD %6.1f CAGR %5.1f tr %4.0f tim %3.0f%% | tough Sh %5.2f DD %6.1f CAGR %5.1f | warm %6.1f%% DD %6.1f' % (n, e[0], 100 * e[1], 100 * e[2], e[3], 100 * e[4], t[0], 100 * t[1], 100 * t[2], 100 * w[5], 100 * w[1])


def shift_breadth(X, k):
    F, B, c = X
    B2 = dict(B)
    for key in list(B2):
        v = B2[key].copy(); v[k:] = B[key][:-k]; v[:k] = np.nan
        B2[key] = v
    return F, B2, c


if __name__ == '__main__':
    world()
    ew_timing()
    with Pool(3, initializer=world) as pool:
        print('\n=== 2. threshold curve on the strategy (band (x, x), sell when off)')
        for r in pool.map(job, [('band %.2f' % x, dict(REF, band=(x, x))) for x in [.3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8]], chunksize=1): print(line(r))
        print('\n=== 5a. other breadth windows')
        print('\n=== 5b. hysteresis around 0.6')
        for r in pool.map(job, [('band (%.2f, %.2f)' % b, dict(REF, band=b)) for b in [(.6, .6), (.55, .6), (.5, .6), (.6, .65), (.6, .7), (.55, .65), (.5, .5), (.45, .55)]], chunksize=1): print(line(r))
