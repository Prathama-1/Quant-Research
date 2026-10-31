"""Task 2 zoom: full factorial around the region the random search pointed to. Means over 4 rebalance offsets."""
import itertools
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from t2_lab import BASE, RESULTS, run, world

OUT = os.path.join(RESULTS, 't2_run5.csv')
SCORES = ['sh126', 'mom_over_vol', 'mix_sh']
EVERY = [10, 21]
BUF = [25, 40]
BAND = [(.5, .5), (.6, .6), (.4, .6), (.3, .6), (.4, .5), (.3, .5), (.5, .7), (.4, .7)]
MV = [0, .65]
BR = ['hi10', 'hi20']


def job(c):
    p = dict(BASE, score=c['score'], every=c['every'], buffer=c['buffer'], band=c['band'], off='sell', breakout=c['br'], extra=10, fill_daily=True)
    if c['mv']: p['maxvol'] = c['mv']
    e = c['every']
    offs = [0, e // 4, e // 2, 3 * e // 4]
    out = dict(c); out['band'] = str(c['band'])
    for name in ['eval', 'tough', 'warm']:
        v = []
        for o in offs:
            s, _ = run(dict(p, offset=o), name)
            v.append((s['sharpe'], s['max_dd'], s['cagr'], s['trades_per_yr'], s['time_in_market'], s['total_return'], s['yearly'].get('2022', np.nan), s['yearly'].get('2023', np.nan), s['yearly'].get('2024', np.nan)))
        v = np.array(v, float)
        for j, k in enumerate(['sh', 'dd', 'cagr', 'tr', 'tim', 'ret', 'y22', 'y23', 'y24']):
            out[name + '_' + k] = v[:, j].mean()
        out[name + '_sh_min'], out[name + '_dd_min'] = v[:, 0].min(), v[:, 1].min()
    return out


if __name__ == '__main__':
    grid = [dict(score=a, every=b, buffer=c, band=d, mv=e, br=f) for a, b, c, d, e, f in itertools.product(SCORES, EVERY, BUF, BAND, MV, BR)]
    world()
    t0 = time.time()
    with Pool(3, initializer=world) as pool:
        rows = pool.map(job, grid, chunksize=2)
    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    print('configs', len(df), 'minutes %.1f' % ((time.time() - t0) / 60))
    cols = ['eval_sh', 'eval_sh_min', 'eval_dd', 'eval_dd_min', 'eval_cagr', 'eval_tr', 'eval_tim', 'eval_y22', 'tough_sh', 'tough_dd', 'tough_cagr', 'warm_ret', 'warm_dd']
    pct = ['eval_dd', 'eval_dd_min', 'eval_cagr', 'eval_tim', 'eval_y22', 'tough_dd', 'tough_cagr', 'warm_ret', 'warm_dd']
    for dim in ['score', 'every', 'buffer', 'band', 'mv', 'br']:
        m = df.groupby(dim)[cols].mean()
        for c in pct: m[c] *= 100
        print('\n--- marginal by', dim); print(m.round(2).to_string())
    x = df.copy()
    x['robust'] = x[['eval_sh_min']].min(axis=1) + x['tough_sh'] + 3 * x['warm_dd'] + 2 * x['tough_dd']
    show = ['score', 'every', 'buffer', 'band', 'mv', 'br', 'eval_sh', 'eval_sh_min', 'eval_dd', 'eval_dd_min', 'eval_cagr', 'eval_tr', 'tough_sh', 'tough_dd', 'tough_cagr', 'warm_ret', 'warm_dd', 'eval_y22', 'eval_y23', 'eval_y24']
    y = x[(x.eval_sh_min > 1.9) & (x.eval_dd_min > -.25)].sort_values('robust', ascending=False)
    print('\n%d of %d have Sharpe > 1.9 and DD > -25%% at EVERY offset' % (len(y), len(x)))
    print(y[show].head(30).round(3).to_string())
