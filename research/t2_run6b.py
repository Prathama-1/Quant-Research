"""Attack the breadth filter, part 2: signal delay, other breadth windows, cross-validation by names, cost multiples.
Implemented by editing the precomputed breadth arrays, so strategy code is untouched."""
import types
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import H
from t2_lab import BASE, TOUGH_G, make_orders, world, EVAL, WARM

REF = dict(BASE, score='sh126', every=10, buffer=40, off='sell', breakout='hi10', extra=10, fill_daily=True, maxvol=.65, band=(.6, .6))


def extra_breadth(D, F, names=None):
    """Breadth on arbitrary windows and optional subset of names. Same formula as the lab: liquid names above their MA."""
    C = D['close']
    ret = C.pct_change(fill_method=None)
    clean = ret.where((ret.abs() < .45) | ret.isna(), 0.0)
    P = (1 + clean.fillna(0)).cumprod().where(C.notna())
    liq = pd.DataFrame(F['liq'], index=C.index, columns=C.columns) >= 1e8
    out = {}
    for n in [20, 30, 50, 75, 100, 200]:
        ab = (P > P.rolling(n, min_periods=max(10, 3 * n // 5)).mean()).where(C.notna()).where(liq)
        out['b%d' % n] = (ab if names is None else ab[names]).mean(axis=1).values
    return out


def run_x(p, X, D, which='eval', universe=None, cost=H.T2_COST):
    a, z = WARM if which == 'warm' else EVAL
    m = types.SimpleNamespace(orders=make_orders(X, universe))
    res = H.task2_backtest(m, D, p, a, z, cost=cost)
    s, _ = H.task2_stats(res)
    return s


def mean_x(p, X, D, which='eval', universe=None, cost=H.T2_COST):
    e = p['every']; v = []
    for o in [0, e // 4, e // 2, 3 * e // 4]:
        s = run_x(dict(p, offset=o), X, D, which, universe, cost)
        v.append((s['sharpe'], s['max_dd'], s['cagr'], s['trades'] / max(s['years'], 1e-9), s['time_in_market'], s['total_return']))
    return np.array(v).mean(axis=0)


def job(a):
    name, kind, arg = a
    w = world()
    F, B, cols = w['X']; FT, BT, _ = w['XT']
    D, DT = w['D'], w['DT']
    X, XT = (F, dict(B), cols), (FT, dict(BT), cols)
    p = dict(REF); uni = None; cost = H.T2_COST
    if kind == 'delay':
        for xb in [X[1], XT[1]]:
            v = xb['breadth50'].copy(); v[arg:] = xb['breadth50'][:-arg] if arg else v[arg:]; v[:arg] = np.nan
            xb['breadth50'] = v
    elif kind == 'window':
        n, th = arg
        X[1]['bx'] = extra_breadth(D, F)['b%d' % n]; XT[1]['bx'] = extra_breadth(DT, FT)['b%d' % n]
        p['band'] = (th, th); p['band_on'] = 'bx'
    elif kind == 'split':
        rng = np.random.default_rng(arg)
        perm = rng.permutation(cols)
        a_, b_ = list(perm[:len(perm) // 2]), set(perm[len(perm) // 2:])
        X[1]['bx'] = extra_breadth(D, F, a_)['b50']; XT[1]['bx'] = extra_breadth(DT, FT, a_)['b50']     # measured on half A
        p['band_on'] = 'bx'; uni = b_                                                                    # traded on half B
    elif kind == 'same':
        rng = np.random.default_rng(arg)
        perm = rng.permutation(cols)
        uni = set(perm[len(perm) // 2:])                                                                 # traded on half B, breadth on everything
    elif kind == 'cost':
        cost = arg * H.T2_COST
    e_ = mean_x(p, X, D, 'eval', uni, cost); t_ = mean_x(p, XT, DT, 'eval', uni, cost); w_ = mean_x(p, X, D, 'warm', uni, cost)
    return name, e_, t_, w_


def line(r):
    n, e, t, w = r
    return '%-30s eval Sh %5.2f DD %6.1f CAGR %5.1f tr %4.0f tim %3.0f%% | tough Sh %5.2f DD %6.1f CAGR %5.1f | warm %6.1f%% DD %6.1f' % (n, e[0], 100 * e[1], 100 * e[2], e[3], 100 * e[4], t[0], 100 * t[1], 100 * t[2], 100 * w[5], 100 * w[1])


if __name__ == '__main__':
    world()
    with Pool(3, initializer=world) as pool:
        print('=== 3. regime signal arrives k days late')
        for r in pool.map(job, [('delay %d' % k, 'delay', k) for k in [1, 2, 3, 5, 10]], chunksize=1): print(line(r))
        print('\n=== 5. other breadth windows x thresholds')
        for r in pool.map(job, [('MA%d, th %.2f' % (n, th), 'window', (n, th)) for n in [20, 30, 50, 75, 100, 200] for th in [.5, .6, .7]], chunksize=1): print(line(r))
        print('\n=== 4. breadth measured on half A, traded on half B (10 random splits)')
        rows = pool.map(job, [('split %d' % i, 'split', i) for i in range(10)], chunksize=1)
        for r in rows: print(line(r))
        print(' mean eval Sharpe %.2f, min %.2f; mean DD %.1f%%; mean tough Sharpe %.2f' % (np.mean([r[1][0] for r in rows]), np.min([r[1][0] for r in rows]), 100 * np.mean([r[1][1] for r in rows]), np.mean([r[2][0] for r in rows])))
        print('\n    same halves traded, breadth on the whole universe (for comparison)')
        rows = pool.map(job, [('same %d' % i, 'same', i) for i in range(10)], chunksize=1)
        print(' mean eval Sharpe %.2f, min %.2f; mean DD %.1f%%; mean tough Sharpe %.2f' % (np.mean([r[1][0] for r in rows]), np.min([r[1][0] for r in rows]), 100 * np.mean([r[1][1] for r in rows]), np.mean([r[2][0] for r in rows])))
        print('\n=== cost multiples')
        for r in pool.map(job, [('cost x%d' % k, 'cost', k) for k in [1, 2, 3, 5]], chunksize=1): print(line(r))
