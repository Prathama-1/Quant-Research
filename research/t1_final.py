"""Tests of the real task1.py: official stats, brief's stress cases, sub-periods, OOS extension, decay scenarios, parameter
sensitivity, strict look-ahead, cold start and short sessions. Writes results/t1_final.json for the report."""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from common import H, RESULTS, ROOT, Clock, row, stress_table, t1_data, t1_eval, yearly
from synth import extended
from t1_decay import haircut

spec = importlib.util.spec_from_file_location('task1', os.path.join(ROOT, 'task1.py'))
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
P = dict(T.PARAMS)
bars, ex = t1_data()
SEEDS = [11, 12, 13, 14, 15]
OUT = {}


def run(b, p=None):
    return np.asarray(T.positions(b, dict(p or P)), float)


def oos_mean(p, worlds, x=0):
    v = []
    for b, e, st in worlds:
        hb = haircut(b, st, x)
        s, _, _ = t1_eval(run(hb, p), hb, e, start=st)
        v.append((s['sharpe'], s['max_dd'], s['cagr']))
    return np.mean(v, axis=0)


w = run(bars)
base, eq, res = t1_eval(w, bars, ex)
print('=== official numbers'); print(row(base)); print('yearly %', yearly(base))
st = stress_table(w, bars, ex); print('\n=== stress'); print(st.to_string()); OUT['stress'] = st.to_dict('index')
OUT['base'] = {k: v for k, v in base.items()}

no_gate = run(bars, dict(P, gate_bps=-1e9))
OUT['gate_changed_bars_in_sample'] = int((no_gate != w).sum())
print('\n=== gate: bars where the paper-trade gate changes the position in-sample: %d' % OUT['gate_changed_bars_in_sample'])

print('\n=== cost multiples (break-even cost)')
OUT['cost_mult'] = {}
for m in [1, 2, 3, 4, 5]:
    s, _, _ = t1_eval(w, bars, ex, cost=m * H.T1_COST)
    OUT['cost_mult'][m] = dict(sharpe=s['sharpe'], cagr=s['cagr'], dd=s['max_dd'])
    print('%dx cost: Sharpe %.2f CAGR %.1f%% DD %.1f%%' % (m, s['sharpe'], 100 * s['cagr'], 100 * s['max_dd']))

print('\n=== sub-periods')
OUT['sub'] = {}
for name, a, z in [('dev 2017-19', None, '2019-12-31'), ('val 2020-Jan22', '2020-01-01', None)]:
    s, _, _ = t1_eval(w, bars, ex, start=a, end=z); print(name, row(s))
    OUT['sub'][name] = dict(sharpe=s['sharpe'], cagr=s['cagr'], dd=s['max_dd'])
OUT['nights_held_share'] = float(((pd.Series(w, index=bars.index).groupby(bars.index.normalize()).max()) > 0).mean())
print('share of sessions followed by a night in the market: %.1f%%' % (100 * OUT['nights_held_share']))

# best year removed: drop the year's days from the return stream and recompute
d = res['daily'] + H.R_CASH / 252
yr = d.groupby(d.index.year).apply(lambda r: (1 + r).prod() - 1)
best = int(yr.iloc[:-1].idxmax()) if yr.index[-1] == 2022 else int(yr.idxmax())
kept = d[d.index.year != best]
eq2 = (1 + kept).cumprod()
yrs = (len(kept)) / 252
nob = dict(best_year=best, cagr=float(eq2.iloc[-1] ** (1 / yrs) - 1), sharpe=float((kept.mean() - H.R_CASH / 252) / kept.std() * 252 ** .5), dd=float((eq2 / eq2.cummax() - 1).min()))
OUT['ex_best_year'] = nob
print('\n=== best year removed:', nob)

t = np.array([r for _, r in res['trades']])
pnl = np.sort(t)[::-1]
print('\n=== trades n=%d win=%.1f%% mean=%.2f bps median=%.2f bps best=%.2f%% worst=%.2f%% | top 10 trades share of gross profit %.1f%%, of net profit %.1f%%' % (
    len(t), 100 * (t > 0).mean(), t.mean() * 1e4, np.median(t) * 1e4, 100 * t.max(), 100 * t.min(), 100 * pnl[:10].sum() / t[t > 0].sum(), 100 * pnl[:10].sum() / t.sum()))
OUT['trades'] = dict(n=len(t), win=float((t > 0).mean()), mean_bps=float(t.mean() * 1e4), worst=float(t.min()), best=float(t.max()), top10_of_net=float(pnl[:10].sum() / t.sum()))

# Turnover and cost drag: same positions at zero cost
s0, _, _ = t1_eval(w, bars, ex, cost=0.0)
OUT['cost_drag'] = dict(gross_cagr=s0['cagr'], net_cagr=base['cagr'], turnover_per_yr=base['turnover_per_yr'])
print('\n=== zero-cost CAGR %.2f%%, net %.2f%%, drag %.2f%%/yr; turnover %.0fx/yr (sides)' % (100 * s0['cagr'], 100 * base['cagr'], 100 * (s0['cagr'] - base['cagr']), base['turnover_per_yr']))

worlds = []
for sd in SEEDS:
    b, e, stt = extended(sd)
    worlds.append((b, e, str(stt.date())))
print('\n=== OOS extension Feb22-Jun24 by path seed')
v = []
for b, e, stt in worlds:
    s, _, _ = t1_eval(run(b), b, e, start=stt)
    v.append(s); print(' Sh %.2f DD %.1f%% CAGR %.1f%% tr/yr %.0f %s' % (s['sharpe'], 100 * s['max_dd'], 100 * s['cagr'], s['trades_per_yr'], yearly(s)))
OUT['oos'] = dict(sharpe=float(np.mean([x['sharpe'] for x in v])), dd=float(np.mean([x['max_dd'] for x in v])), cagr=float(np.mean([x['cagr'] for x in v])), trades_per_yr=float(np.mean([x['trades_per_yr'] for x in v])),
                  sharpe_min=float(np.min([x['sharpe'] for x in v])), sharpe_max=float(np.max([x['sharpe'] for x in v])))
print(' mean', OUT['oos'])

print('\n=== decay: X bps removed from each night. OOS mean over seeds, then real IS. Sharpe / DD% / CAGR%')
OUT['decay'] = {}
for x in [0, 3, 6, 9, 12, 15]:
    o = oos_mean(P, worlds[:3], x)
    hb = haircut(bars, str(bars.index[0].date()), x)
    s, _, _ = t1_eval(run(hb), hb, ex)
    OUT['decay'][x] = dict(oos_sh=float(o[0]), oos_dd=float(o[1]), oos_cagr=float(o[2]), is_sh=s['sharpe'], is_dd=s['max_dd'], is_cagr=s['cagr'])
    print(' -%2d bps: OOS %5.2f /%5.1f /%5.1f | IS %5.2f /%5.1f /%5.1f' % (x, o[0], 100 * o[1], 100 * o[2], s['sharpe'], 100 * s['max_dd'], 100 * s['cagr']))

print('\n=== sensitivity grid: MA days x session floor (IS Sharpe / IS DD / d1 / 2x / OOS Sharpe)')
OUT['grid'] = []
for ma in [10, 15, 20, 25, 30]:
    for fl in [-.0075, -.005, -.0025, 0.0]:
        p = dict(P, ma_days=ma, session_floor=fl)
        x = run(bars, p)
        s, _, _ = t1_eval(x, bars, ex); s1, _, _ = t1_eval(x, bars, ex, delay=1); s2, _, _ = t1_eval(x, bars, ex, cost=2 * H.T1_COST)
        o = oos_mean(p, worlds[:3])
        OUT['grid'].append(dict(ma=ma, floor=fl, sh=s['sharpe'], dd=s['max_dd'], cagr=s['cagr'], d1=s1['sharpe'], c2=s2['sharpe'], oos=float(o[0])))
        print(' ma %2d floor %6.2f%%: Sh %.2f DD %5.1f%% CAGR %4.1f%% | d1 %.2f 2x %.2f | OOS %.2f' % (ma, 100 * fl, s['sharpe'], 100 * s['max_dd'], 100 * s['cagr'], s1['sharpe'], s2['sharpe'], o[0]))

print('\n=== one parameter at a time')
OUT['oat'] = {}
for k, vals in [('decide_at', ['13:45', '14:00', '14:15', '14:30']), ('vol_cap', [.25, .30, .35, .45, 9.9]), ('gate_nights', [60, 90, 120, 180, 250]), ('gate_bps', [-999, 3, 5, 7, 10, 15])]:
    OUT['oat'][k] = []
    for val in vals:
        p = dict(P); p[k] = val
        x = run(bars, p)
        s, _, _ = t1_eval(x, bars, ex); s1, _, _ = t1_eval(x, bars, ex, delay=1); s2, _, _ = t1_eval(x, bars, ex, delay=2); s3, _, _ = t1_eval(x, bars, ex, cost=2 * H.T1_COST)
        o = oos_mean(p, worlds[:3]); dec = oos_mean(p, worlds[:3], 9)
        OUT['oat'][k].append(dict(value=val, sh=s['sharpe'], dd=s['max_dd'], cagr=s['cagr'], d1=s1['sharpe'], d2=s2['sharpe'], c2=s3['sharpe'], oos=float(o[0]), oos_dd=float(o[1]), oos_decay9=float(dec[0])))
        print(' %-12s %-6s Sh %.2f DD %5.1f%% CAGR %4.1f%% tr %3.0f | d1 %.2f d2 %.2f 2x %.2f | OOS %.2f DD %.1f%% | OOS at -9bps %.2f' % (k, val, s['sharpe'], 100 * s['max_dd'], 100 * s['cagr'], s['trades_per_yr'], s1['sharpe'], s2['sharpe'], s3['sharpe'], o[0], 100 * o[1], dec[0]))

print('\n=== strict look-ahead: 400 random cuts + every bar of 5 random days + short session day')
rng = np.random.default_rng(3)
full = run(bars)
cuts = list(rng.integers(H.T1_WARMUP + 1, len(bars) - 1, 400))
ck = Clock(bars)
for d in rng.choice(ck.day_id[-1], 5, replace=False):
    cuts += list(np.flatnonzero(ck.day_id == d))
cuts += list(np.flatnonzero(bars.index.normalize() == pd.Timestamp('2021-11-04')))
t0 = time.time()
bad = [int(k) for k in cuts if run(bars.iloc[:k + 1])[-1] != full[k]]
print(' cuts tested %d, mismatches %d, %.1fs' % (len(cuts), len(bad), time.time() - t0))
OUT['lookahead'] = dict(cuts=len(cuts), mismatches=len(bad))
# stronger: flip the future. Corrupt everything after a cut and check nothing up to the cut changes.
bad2 = 0
for k in rng.integers(H.T1_WARMUP + 1, len(bars) - 2, 50):
    corrupt = bars.copy()
    for col in ['open', 'high', 'low', 'close']:
        v = corrupt[col].to_numpy(copy=True); v[k + 1:] = v[k + 1:] * rng.uniform(.5, 1.5, len(v) - k - 1); corrupt[col] = v
    bad2 += int((run(corrupt)[:k + 1] != full[:k + 1]).sum())
print(' future-corruption test: changed positions before the cut = %d' % bad2)
OUT['lookahead']['future_corruption_changes'] = bad2

print('\n=== cold start: run only on bars from a late date; how fast does it agree with the full run?')
OUT['coldstart'] = {}
for start in ['2019-01-01', '2020-06-01', '2021-02-01']:
    sub = bars[bars.index >= start]
    cold = run(sub); fullsub = full[-len(sub):]
    s, _, _ = t1_eval(cold, sub, ex)
    w0 = H.T1_WARMUP
    diff = np.flatnonzero(cold[w0:] != fullsub[w0:])
    agree60 = float((cold[w0:] == fullsub[w0:]).mean())
    lastdiff = sub.index[w0 + diff[-1]] if len(diff) else None
    OUT['coldstart'][start] = dict(sharpe=s['sharpe'], dd=s['max_dd'], agree=agree60, last_diff=str(lastdiff))
    print(' %s: Sharpe %.2f DD %.1f%% | agreement with full run %.1f%% | last disagreement %s' % (start, s['sharpe'], 100 * s['max_dd'], 100 * agree60, lastdiff))

print('\n=== short sessions and odd calendars: synthetic sessions at various times inserted into the data')
extra = []
for date, times in [('2022-02-05', ['09:15', '09:30', '09:45', '11:30', '11:45', '12:00', '12:15']),     # special Saturday, two blocks
                    ('2022-02-12', ['13:45', '14:00', '14:15', '14:30']),                                 # Muhurat-like ending on the decision bar
                    ('2022-02-19', ['14:45', '15:00', '15:15']),                                          # like 2021-11-04
                    ('2022-02-26', ['14:30'])]:                                                           # one lonely bar
    last = float(bars.close.iloc[-1])
    for t in times:
        extra.append((pd.Timestamp(date + ' ' + t), last, last * 1.001, last * .999, last))
xb = pd.concat([bars, pd.DataFrame(extra, columns=['dt', 'open', 'high', 'low', 'close']).set_index('dt')]).sort_index()
xw = run(xb)
tail = pd.Series(xw, index=xb.index).loc['2022-02-01':]
print(' values', sorted(set(tail.values)), 'non-zero bars after insertion:', int((tail != 0).sum()))
ok = np.isin(xw, [-1, 0, 1]).all() and not np.isnan(xw).any()
print(' valid output:', bool(ok))
OUT['short_session_ok'] = bool(ok and (tail == 0).all())

print('\n=== empty and tiny inputs')
for n in [1, 2, 5, 30, 201]:
    y = run(bars.iloc[:n]); print(' first %d bars: ok, non-zero %d' % (n, int((y != 0).sum())))

json.dump(OUT, open(os.path.join(RESULTS, 't1_final.json'), 'w'), indent=1, default=float)
