"""Focused sweep of the overnight family. Vectorised so hundreds of configs run quickly; every config is logged to
results/t1_lab2.csv. OOS is the mean over several synthetic path seeds so that no conclusion rests on one random path."""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import H, RESULTS, Clock, hm, t1_data, t1_eval, yearly
from synth import extended

LOG = os.path.join(RESULTS, 't1_lab2.csv')
SEEDS = [11, 12, 13]
E_1515 = hm('15:15')
_cache = {}


def world():
    if not _cache:
        b, e = t1_data()
        _cache['is'] = (b, e, Clock(b))
        _cache['oos'] = []
        for s in SEEDS:
            ab, ae, st = extended(s)
            _cache['oos'].append((ab, ae, Clock(ab), str(st.date())))
    return _cache


def decision(bars, ck, dec):
    idx = np.flatnonzero(ck.tod == dec)
    return idx, bars.close.values[idx], ck.day_id[idx]


def signal(bars, ck, p):
    """Returns (bar index of decision, boolean per decision) for config p. Uses only data up to each decision bar."""
    dec = hm(p.get('dec', '14:30'))
    idx, px, day = decision(bars, ck, dec)
    s = pd.Series(px)
    on = np.ones(len(px), bool)
    if 'ma' in p:
        on &= (s > s.shift(1).rolling(p['ma'], min_periods=max(2, p['ma'] // 2)).mean()).values
    if 'ma2' in p:
        on &= (s > s.shift(1).rolling(p['ma2'], min_periods=max(2, p['ma2'] // 2)).mean()).values
    if 'slope' in p:                                    # MA(n) higher than it was m days ago
        n, m = p['slope']
        ma = s.rolling(n, min_periods=max(2, n // 2)).mean()
        on &= (ma > ma.shift(m)).values
    if 'mom' in p:                                      # time-series momentum
        on &= (s > s.shift(p['mom'])).values
    if 'vote' in p:                                     # fraction of a lookback ladder with price above its MA
        ladder, th = p['vote']
        v = np.mean([(s > s.shift(1).rolling(n, min_periods=max(2, n // 2)).mean()).values for n in ladder], axis=0)
        on &= v >= th
    if 'rng' in p:                                      # position in the n-day decision-price range
        n, th = p['rng']
        hi, lo = s.shift(1).rolling(n, min_periods=max(2, n // 2)).max(), s.shift(1).rolling(n, min_periods=max(2, n // 2)).min()
        on &= ((s - lo) / (hi - lo)).fillna(0).values >= th
    if 'rvmax' in p:                                    # no overnight risk in a violent market
        n, cap = p['rvmax']
        rv = s.pct_change().rolling(n, min_periods=5).std().fillna(0).values * 252 ** .5
        on &= rv < cap
    if 'sofar' in p:                                    # skip after a weak session so far (open to decision bar)
        first = np.flatnonzero(ck.first)
        o_day = bars.open.values[first][day]
        on &= (px / o_day - 1) >= p['sofar']
    if 'premium' in p:                                  # is the overnight premium still paying? trailing mean gap, bps
        m, g = p['premium']
        first = np.flatnonzero(ck.first)
        o = bars.open.values[first]
        c = bars.close.values[first - 1]; c[0] = o[0]
        gaps = pd.Series(o / c - 1)
        tr = gaps.rolling(m, min_periods=m // 2).mean().fillna(1.0).values          # today's gap is already known at decision time
        on &= tr[day] * 1e4 >= g
    if 'dd_max' in p:                                   # stand aside when the index is far under its recent peak
        n, x = p['dd_max']
        on &= (s / s.rolling(n, min_periods=2).max() - 1).values >= -x
    return idx, on, day


def want_of(bars, ck, p):
    dec = hm(p.get('dec', '14:30'))
    idx, on, day = signal(bars, ck, p)
    by_day = np.zeros(ck.day_id[-1] + 1, bool)
    by_day[day] = on
    mode = p.get('mode', 'night')
    if mode == 'night':
        return (by_day[ck.day_id] & (ck.tod >= dec) & (ck.tod < E_1515)).astype(float)
    # hold: enter at a decision bar, exit (at the next open) from the last bar of the first day the signal is off
    want = np.zeros(len(bars)); hold = 0.0; dec_bar = np.zeros(len(bars), bool); dec_bar[idx] = True
    has_dec = np.zeros(ck.day_id[-1] + 1, bool); has_dec[day] = True
    for i in range(len(bars)):
        d = ck.day_id[i]
        if dec_bar[i] and by_day[d]: hold = 1.0
        if ck.tod[i] == E_1515 and has_dec[d] and not by_day[d]: hold = 0.0
        want[i] = hold
    return want


def score(p):
    w = world()
    bars, ex, ck = w['is']
    x = want_of(bars, ck, p)
    base, _, _ = t1_eval(x, bars, ex)
    d1, _, _ = t1_eval(x, bars, ex, delay=1)
    d2, _, _ = t1_eval(x, bars, ex, delay=2)
    c2, _, _ = t1_eval(x, bars, ex, cost=2 * H.T1_COST)
    dev, _, _ = t1_eval(x, bars, ex, end='2019-12-31')
    val, _, _ = t1_eval(x, bars, ex, start='2020-01-01')
    # year-wise excess sharpe stability: worst full year
    r = H.task1_backtest(bars, x, ex)['daily']
    ys = r.groupby(r.index.year).apply(lambda v: v.mean() / v.std() * 252 ** .5 if v.std() > 0 else 0.0)
    oos = []
    for ab, ae, ack, st in w['oos']:
        o, _, _ = t1_eval(want_of(ab, ack, p), ab, ae, start=st)
        oos.append((o['sharpe'], o['max_dd'], o['cagr'], o['yearly'].get('2022', np.nan)))
    oos = np.array(oos)
    return dict(p=str(p), cagr=base['cagr'], dd=base['max_dd'], sh=base['sharpe'], d1=d1['sharpe'], d2=d2['sharpe'], c2=c2['sharpe'], d1dd=d1['max_dd'], c2dd=c2['max_dd'],
                dev=dev['sharpe'], val=val['sharpe'], worst_yr=float(ys.iloc[:-1].min()), tr=base['trades_per_yr'],
                oos_sh=oos[:, 0].mean(), oos_dd=oos[:, 1].mean(), oos_cagr=oos[:, 2].mean(), oos_22=oos[:, 3].mean())


def show(rows, sort=None):
    df = pd.DataFrame(rows)
    if sort: df = df.sort_values(sort, ascending=False)
    for r in df.to_dict('records'):
        print('%-62s CAGR %5.1f DD %6.1f Sh %5.2f | d1 %5.2f d2 %5.2f 2x %5.2f | dev %5.2f val %5.2f wy %5.2f | OOS Sh %5.2f DD %6.1f CAGR %5.1f y22 %5.1f | tr %3.0f' % (
            r['p'][:62], 100 * r['cagr'], 100 * r['dd'], r['sh'], r['d1'], r['d2'], r['c2'], r['dev'], r['val'], r['worst_yr'], r['oos_sh'], 100 * r['oos_dd'], 100 * r['oos_cagr'], 100 * r['oos_22'], r['tr']))


def run(name, configs, pool):
    print('\n=== ' + name)
    rows = pool.map(score, configs, chunksize=1)
    pd.DataFrame(rows).assign(group=name).to_csv(LOG, mode='a', header=not os.path.exists(LOG), index=False)
    show(rows)
    return rows


if __name__ == '__main__':
    if os.path.exists(LOG): os.remove(LOG)
    world()
    L = [5, 8, 10, 13, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    with Pool(6, initializer=world) as pool:
        run('MA lookback (night)', [dict(ma=n) for n in L], pool)
        run('MA lookback (hold)', [dict(ma=n, mode='hold') for n in L], pool)
        run('decision time x MA', [dict(ma=n, dec=t) for n in [10, 20, 40] for t in ['13:30', '14:00', '14:15', '14:30', '14:45', '15:00']], pool)
        run('time-series momentum', [dict(mom=n) for n in [5, 10, 20, 40, 60, 120]], pool)
        run('MA slope', [dict(slope=(n, m)) for n in [10, 20, 50] for m in [1, 5, 10]], pool)
        run('range position', [dict(rng=(n, th)) for n in [10, 20, 50] for th in [.25, .4, .5, .6]], pool)
        run('vote ladder', [dict(vote=(ld, th)) for ld in [(5, 10, 20), (10, 20, 40), (10, 20, 40, 80), (5, 10, 20, 40, 80, 160), (20, 40, 80)] for th in [.34, .5, .67, 1.0]], pool)
        run('vol cap on MA20', [dict(ma=20, rvmax=(n, c)) for n in [10, 20] for c in [.15, .20, .25, .30, .40]], pool)
        run('session-so-far on MA20', [dict(ma=20, sofar=x) for x in [-.01, -.005, -.0025, 0.0]], pool)
        run('premium gate on MA20', [dict(ma=20, premium=(m, g)) for m in [20, 40, 60, 120] for g in [0, 5, 7, 10]], pool)
        run('drawdown gate on MA20', [dict(ma=20, dd_max=(n, x)) for n in [20, 60, 120] for x in [.03, .05, .08]], pool)
