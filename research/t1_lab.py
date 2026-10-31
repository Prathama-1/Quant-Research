"""Task 1 laboratory. Each family maps (bars, params) to a want-array. Every config is scored on: full in-sample, 1- and 2-bar
delay, 2x costs, dev (2017-19), val (2020-Jan22) and the OOS extension (Feb22-Jun24: real gaps and day ranges, synthetic
intraday paths). Everything run here is appended to results/t1_lab.csv so the number of configs tried is on record."""
import os
import sys

import numpy as np
import pandas as pd

from common import H, RESULTS, Clock, hm, t1_data, t1_eval, yearly
from synth import extended

IS_BARS, IS_EX = t1_data()
ALL_BARS, ALL_EX, OOS_START = extended()
OOS_START = str(OOS_START.date())
LOG = os.path.join(RESULTS, 't1_lab.csv')


def decision_series(bars, tod):
    """One value per day: the close of the decision bar and its bar index. Causal: only bars <= decision time."""
    ck = Clock(bars)
    mask = ck.tod == tod
    idx = np.flatnonzero(mask)
    return ck, idx, bars.close.values[idx], ck.day_id[idx]


def daily_ohl(bars, ck):
    g = pd.DataFrame({'h': bars.high.values, 'l': bars.low.values, 'c': bars.close.values, 'o': bars.open.values, 'd': ck.day_id}).groupby('d')
    return g.o.first().values, g.h.max().values, g.l.min().values, g.c.last().values


def night_want(bars, ck, idx, day, sig, end='15:15'):
    """Hold from the fill after the decision bar through the next open: want = sig on bars from decision up to, not incl., 15:15."""
    want = np.zeros(len(bars))
    by_day = dict(zip(day, sig))
    e = hm(end)
    start_tod = dict(zip(day, ck.tod[idx]))
    for i in range(len(bars)):
        d = ck.day_id[i]
        if by_day.get(d, False) and start_tod[d] <= ck.tod[i] < e:
            want[i] = 1.0
    return want


def state_want(bars, ck, idx, day, sig, exit_tod='15:15'):
    """Multi-day hold: enter at a decision bar when sig is on; exit decision taken at exit_tod (next open fill) once sig is off."""
    want = np.zeros(len(bars)); hold = 0.0
    by_day = dict(zip(day, sig))
    dec = dict(zip(idx, sig))
    x = hm(exit_tod)
    for i in range(len(bars)):
        if i in dec and dec[i]:
            hold = 1.0
        if ck.tod[i] == x and not by_day.get(ck.day_id[i], True):
            hold = 0.0
        want[i] = hold
    return want


def ma_sig(px, n):
    s = pd.Series(px)
    return (s > s.shift(1).rolling(n, min_periods=max(2, n // 2)).mean()).values


# ------------------------------------------------------------------------------------------------- families
def f_night_ma(bars, p):
    ck, idx, px, day = decision_series(bars, hm(p.get('dec', '14:30')))
    return night_want(bars, ck, idx, day, ma_sig(px, p['n']))


def f_night_ma_fast_slow(bars, p):
    ck, idx, px, day = decision_series(bars, hm(p.get('dec', '14:30')))
    return night_want(bars, ck, idx, day, ma_sig(px, p['fast']) & ma_sig(px, p['slow']))


def f_night_ma_vol(bars, p):
    ck, idx, px, day = decision_series(bars, hm(p.get('dec', '14:30')))
    r = pd.Series(px).pct_change()
    rv = (r.shift(1).rolling(p['rvn'], min_periods=5).std() * 252 ** .5).fillna(0).values
    return night_want(bars, ck, idx, day, ma_sig(px, p['n']) & (rv < p['rvmax']))


def f_hold_ma(bars, p):
    ck, idx, px, day = decision_series(bars, hm(p.get('dec', '14:30')))
    return state_want(bars, ck, idx, day, ma_sig(px, p['n']))


def f_hold_ma_vol(bars, p):
    ck, idx, px, day = decision_series(bars, hm(p.get('dec', '14:30')))
    r = pd.Series(px).pct_change()
    rv = (r.shift(1).rolling(p['rvn'], min_periods=5).std() * 252 ** .5).fillna(0).values
    return state_want(bars, ck, idx, day, ma_sig(px, p['n']) & (rv < p['rvmax']))


def f_hold_two_ma(bars, p):
    ck, idx, px, day = decision_series(bars, hm(p.get('dec', '14:30')))
    return state_want(bars, ck, idx, day, ma_sig(px, p['fast']) & ma_sig(px, p['slow']))


def f_ma_bars(bars, p):
    """Bar-level MA crossover. side = long-flat or long-short."""
    c = bars.close
    f, s = c.rolling(p['fast'], min_periods=p['fast'] // 2).mean(), c.rolling(p['slow'], min_periods=p['slow'] // 2).mean()
    up = (f > s).astype(float).values
    return up if p['side'] == 'lf' else 2 * up - 1


def f_donchian(bars, p):
    """N-bar breakout with M-bar exit channel. Long-flat or long-short, checked on every bar."""
    h = bars.high.shift(1).rolling(p['n'], min_periods=p['n'] // 2).max().values
    l = bars.low.shift(1).rolling(p['n'], min_periods=p['n'] // 2).min().values
    hx = bars.high.shift(1).rolling(p['m'], min_periods=p['m'] // 2).max().values
    lx = bars.low.shift(1).rolling(p['m'], min_periods=p['m'] // 2).min().values
    c = bars.close.values; w = np.zeros(len(c)); pos = 0.0
    for i in range(len(c)):
        if pos > 0 and c[i] < lx[i]: pos = 0.0
        elif pos < 0 and c[i] > hx[i]: pos = 0.0
        if c[i] > h[i]: pos = 1.0
        elif c[i] < l[i] and p['side'] == 'ls': pos = -1.0
        w[i] = pos
    return w


def f_orb(bars, p):
    """Opening-range breakout, intraday only: range = first k bars, enter on a close outside, flat by the last bars."""
    ck = Clock(bars); N = len(bars); w = np.zeros(N)
    c, h, l = bars.close.values, bars.high.values, bars.low.values
    k = p['k']; hi = lo = None; pos = 0.0; n_today = 0
    for i in range(N):
        if ck.first[i]: hi, lo, pos, n_today = h[i], l[i], 0.0, 0
        n_today += 1
        if n_today <= k:
            hi, lo = max(hi, h[i]), min(lo, l[i])
        elif ck.tod[i] < hm(p.get('flat', '15:00')):
            if pos == 0 and c[i] > hi: pos = 1.0
            elif pos == 0 and c[i] < lo and p['side'] == 'ls': pos = -1.0
            elif pos == 1.0 and p.get('stop_mid') and c[i] < (hi + lo) / 2: pos = 0.0
            elif pos == -1.0 and p.get('stop_mid') and c[i] > (hi + lo) / 2: pos = 0.0
        else:
            pos = 0.0
        w[i] = pos
    return w


def f_gap_down_short(bars, p):
    """Short the session after a gap of at most -th, measured at the first bar's close. Flat before the close."""
    ck = Clock(bars); N = len(bars); w = np.zeros(N)
    c, o = bars.close.values, bars.open.values
    pos = 0.0
    for i in range(N):
        if ck.first[i] and i > 0:
            pos = -1.0 if (o[i] / c[i - 1] - 1) <= -p['th'] else 0.0
        if ck.tod[i] >= hm('15:00'): pos = 0.0
        w[i] = pos
    return w


FAMILIES = {k[2:]: v for k, v in list(globals().items()) if k.startswith('f_') and callable(v)}


def score(fam, params, log=True, quiet=False):
    f = FAMILIES[fam]
    w_is = np.asarray(f(IS_BARS, params), float)
    w_all = np.asarray(f(ALL_BARS, params), float)
    base, _, _ = t1_eval(w_is, IS_BARS, IS_EX)
    d1, _, _ = t1_eval(w_is, IS_BARS, IS_EX, delay=1)
    d2, _, _ = t1_eval(w_is, IS_BARS, IS_EX, delay=2)
    c2, _, _ = t1_eval(w_is, IS_BARS, IS_EX, cost=2 * H.T1_COST)
    dev, _, _ = t1_eval(w_is, IS_BARS, IS_EX, end='2019-12-31')
    val, _, _ = t1_eval(w_is, IS_BARS, IS_EX, start='2020-01-01')
    oos, _, _ = t1_eval(w_all, ALL_BARS, ALL_EX, start=OOS_START)
    rec = dict(fam=fam, params=str(params), cagr=base['cagr'], dd=base['max_dd'], sharpe=base['sharpe'], tr_yr=base['trades_per_yr'], tim=base['time_in_market'],
               d1_sh=d1['sharpe'], d1_dd=d1['max_dd'], d2_sh=d2['sharpe'], c2_sh=c2['sharpe'], c2_dd=c2['max_dd'], dev_sh=dev['sharpe'], val_sh=val['sharpe'], dev_dd=dev['max_dd'], val_dd=val['max_dd'],
               oos_sh=oos['sharpe'], oos_dd=oos['max_dd'], oos_cagr=oos['cagr'], yearly=str(yearly(base)), oos_yearly=str(yearly(oos)))
    if log:
        pd.DataFrame([rec]).to_csv(LOG, mode='a', header=not os.path.exists(LOG), index=False)
    if not quiet:
        print(fmt(rec))
    return rec


def fmt(r):
    return '%-16s %-52s CAGR %6.2f DD %7.2f Sh %5.2f | d1 %5.2f d2 %5.2f 2x %5.2f | dev %5.2f val %5.2f | OOS Sh %5.2f DD %7.2f CAGR %6.2f | tr %4.0f tim %3.0f%%' % (
        r['fam'], r['params'][:52], 100 * r['cagr'], 100 * r['dd'], r['sharpe'], r['d1_sh'], r['d2_sh'], r['c2_sh'], r['dev_sh'], r['val_sh'], r['oos_sh'], 100 * r['oos_dd'], 100 * r['oos_cagr'], r['tr_yr'], 100 * r['tim'])


if __name__ == '__main__':
    print('--- trend/breakout families people usually start with')
    for fast, slow in [(50, 200), (25, 100), (125, 500), (250, 1000), (625, 2500)]:
        score('ma_bars', dict(fast=fast, slow=slow, side='lf'))
        score('ma_bars', dict(fast=fast, slow=slow, side='ls'))
    for n, m in [(25, 13), (50, 25), (125, 50), (250, 125), (500, 250), (1250, 500)]:
        score('donchian', dict(n=n, m=m, side='lf'))
        score('donchian', dict(n=n, m=m, side='ls'))
    print('--- opening range breakout (intraday only)')
    for k in [2, 4, 8]:
        for side in ['lf', 'ls']:
            score('orb', dict(k=k, side=side, stop_mid=True))
    print('--- gap-down short (intraday only)')
    for th in [0.0025, 0.005, 0.01]:
        score('gap_down_short', dict(th=th))
    print('--- overnight long, day-level MA filter')
    for n in [5, 10, 20, 50, 100, 200]:
        score('night_ma', dict(n=n))
    print('--- overnight long, two MA filters')
    for fast, slow in [(5, 50), (5, 100), (5, 200), (10, 50), (10, 100), (10, 200), (20, 100), (20, 200)]:
        score('night_ma_fast_slow', dict(fast=fast, slow=slow))
    print('--- multi-day long hold, day-level MA filter (exit at next open once the filter is off)')
    for n in [5, 10, 20, 50, 100, 200]:
        score('hold_ma', dict(n=n))
    print('--- multi-day long hold, two MA filters')
    for fast, slow in [(5, 50), (5, 200), (10, 50), (10, 200), (20, 100), (20, 200)]:
        score('hold_two_ma', dict(fast=fast, slow=slow))
