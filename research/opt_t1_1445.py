"""Task 1: what does deciding at 14:45 instead of 14:30 change, one parameter at a time, official harness numbers. Run from inside research/."""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from opt_check import boot_diff, delayed, fast_daily, load_t1
from common import H, RESULTS, t1_data
from synth import extended
pd.set_option('display.width', 200)
g = pd.read_csv(os.path.join(RESULTS, 'opt_t1_grid.csv'))
print('Mean across the 800-config grid, by decision time:')
print(g.groupby('dec')[['sharpe', 'dd', 'cagr', 'd1', 'd2', 'c2', 'oos']].mean().round(3).to_string())
print('\nMean by session floor:'); print(g[g.dec != '14:45'].groupby('floor')[['sharpe', 'dd', 'cagr', 'd1', 'd2', 'c2', 'oos']].mean().round(3).to_string())
print('\nMean by MA days:'); print(g[g.dec != '14:45'].groupby('ma')[['sharpe', 'dd', 'cagr', 'd1', 'd2', 'c2', 'oos']].mean().round(3).to_string())
print('\nMean by vol cap:'); print(g[g.dec != '14:45'].groupby('cap')[['sharpe', 'dd', 'cagr', 'd1', 'd2', 'c2', 'oos']].mean().round(3).to_string())
T = load_t1(); P = dict(T.PARAMS)
bars, ex = t1_data(); ex_arr = pd.DatetimeIndex(sorted(ex))
worlds = [extended(s) for s in [11, 12, 13, 14, 15]]
print('\nOfficial harness, changing only the decision time:')
base = None
for dec in ['14:15', '14:30', '14:45', '15:00']:
    p = dict(P, decide_at=dec)
    w = np.asarray(T.positions(bars, p), float)
    s, _ = H.task1_stats(H.task1_backtest(bars, w, ex))
    d1, _ = H.task1_stats(H.task1_backtest(bars, delayed(w, 1), ex)); d2, _ = H.task1_stats(H.task1_backtest(bars, delayed(w, 2), ex))
    c2, _ = H.task1_stats(H.task1_backtest(bars, w, ex, cost=2 * H.T1_COST))
    r = H.task1_backtest(bars, w, ex)['daily']
    y = r.groupby(r.index.year).apply(lambda v: v.mean() / v.std() * 252 ** .5).round(2).to_dict()
    oo = []
    for b, e, st in worlds:
        x, _ = H.task1_stats(H.task1_backtest(b, np.asarray(T.positions(b, p), float), e), start=str(pd.Timestamp(st).date()))
        oo.append(x['sharpe'])
    print('%s: Sharpe %.2f DD %.1f%% CAGR %.1f%% tr/yr %.0f | d1 %.2f d2 %.2f 2x %.2f | OOS mean %.2f %s | by year %s' % (dec, s['sharpe'], 100 * s['max_dd'], 100 * s['cagr'], s['trades'] / s['years'], d1['sharpe'], d2['sharpe'], c2['sharpe'], np.mean(oo), np.round(oo, 2).tolist(), y))
    if dec == '14:30': base = fast_daily(bars, w, ex_arr).values
    else:
        ci, pos = boot_diff(fast_daily(bars, w, ex_arr).values, base) if base is not None else (None, None)
        if ci is not None: print('   Sharpe difference vs 14:30: 95%% CI %+.2f to %+.2f, P(better) %.0f%%' % (ci[0], ci[2], 100 * pos))
