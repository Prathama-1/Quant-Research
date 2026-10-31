"""Which overnight filters earn their place? Full factorial, then MARGINAL effects (mean over all other dimensions), which is far
less prone to picking a lucky cell than reading the best row. Also tests the short side and the tail dependence of the leader."""
import itertools
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import H, RESULTS, hm, t1_eval, yearly
from t1_lab2 import score, show, want_of, world

LOG = os.path.join(RESULTS, 't1_lab3.csv')


def cfg(ma, sofar, cap, prem):
    p = dict(ma=ma)
    if sofar is not None: p['sofar'] = sofar
    if cap is not None: p['rvmax'] = (20, cap)
    if prem is not None: p['premium'] = prem
    return p


if __name__ == '__main__':
    world()
    MA, SOFAR, CAP, PREM = [15, 20, 25], [None, -.005, -.0025, 0.0], [None, .25, .30, .35], [None, (60, 7), (120, 5), (120, 7)]
    grid = list(itertools.product(MA, SOFAR, CAP, PREM))
    with Pool(6, initializer=world) as pool:
        rows = pool.map(score, [cfg(*g) for g in grid], chunksize=4)
    df = pd.DataFrame(rows)
    df['ma'] = [g[0] for g in grid]; df['sofar'] = [str(g[1]) for g in grid]; df['cap'] = [str(g[2]) for g in grid]; df['prem'] = [str(g[3]) for g in grid]
    df.to_csv(LOG, index=False)
    print('configs', len(df), ' all beat the bar (Sh>1, DD>-15%%): %d' % ((df.sh > 1) & (df.dd > -.15)).sum())
    cols = ['cagr', 'dd', 'sh', 'd1', 'd2', 'c2', 'dev', 'val', 'worst_yr', 'oos_sh', 'oos_dd', 'oos_cagr', 'oos_22', 'tr']
    for dim in ['ma', 'sofar', 'cap', 'prem']:
        print('\n--- marginal mean by', dim)
        m = df.groupby(dim)[cols].mean()
        for c in ['cagr', 'dd', 'oos_dd', 'oos_cagr', 'oos_22']: m[c] *= 100
        print(m.round(2).to_string())
    print('\n--- top 12 by the worst of (base, d1, 2x, OOS) so that no single stress column can carry a config')
    df['floor'] = df[['sh', 'd1', 'c2', 'oos_sh']].min(axis=1)
    show(df.sort_values('floor', ascending=False).head(12).to_dict('records'))
    print('\n--- spread of OOS Sharpe across the whole grid: min %.2f, q25 %.2f, median %.2f, q75 %.2f, max %.2f' % tuple(df.oos_sh.quantile([0, .25, .5, .75, 1])))
    print('--- spread of IS Sharpe: min %.2f, median %.2f, max %.2f' % (df.sh.min(), df.sh.median(), df.sh.max()))
