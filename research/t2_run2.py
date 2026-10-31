"""Task 2 factorial: signal x regime filter x stop x sell-buffer. Read as marginal effects, not as a leaderboard, because the eval
window is one bull market and individual cells are noisy. A cell only counts if it holds up in warm and tough too."""
import itertools
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from t2_lab import BASE, RESULTS, fmt, score, world

OUT = os.path.join(RESULTS, 't2_run2.csv')
SIGS = ['res126', 'sh126', 'mix_sh', 'mom126_21', 'tvshock', 'mom_over_vol']
REG = {'none': None, 'n50': {'nifty_ma50': 0}, 'n100': {'nifty_ma100': 0}, 'n200': {'nifty_ma200': 0}, 'b50.4': {'breadth50': .4}, 'b50.5': {'breadth50': .5}, 'e50': {'ew_ma50': 0}, 'e100': {'ew_ma100': 0}, 'b200.5': {'breadth200': .5}}
STOPS = [None, .15, .25]
BUF = [15, 45]
OFF = ['keep', 'sell']

if __name__ == '__main__':
    world()
    grid, meta = [], []
    for s, (rn, r), st, b, off in itertools.product(SIGS, REG.items(), STOPS, BUF, OFF):
        if r is None and off == 'sell': continue
        p = dict(BASE, score=s, buffer=b)
        if r: p['regime'] = r; p['off'] = off
        if st: p['stop'] = st
        grid.append(p); meta.append(dict(sig=s, reg=rn, stop=str(st), buf=b, off=off))
    t0 = time.time()
    with Pool(3, initializer=world) as pool:
        rows = pool.map(score, grid, chunksize=4)
    print('configs', len(grid), 'minutes %.1f' % ((time.time() - t0) / 60))
    df = pd.concat([pd.DataFrame(rows), pd.DataFrame(meta)], axis=1)
    df.to_csv(OUT, index=False)
    cols = ['cagr', 'dd', 'sh', 'tr', 'tim', 'c2_sh', 'warm_ret', 'warm_dd', 't_sh', 't_dd', 't_cagr', 'y22', 'y23', 'y24']
    for dim in ['sig', 'reg', 'stop', 'buf', 'off']:
        m = df.groupby(dim)[cols].mean()
        for c in ['cagr', 'dd', 'tim', 'warm_ret', 'warm_dd', 't_dd', 't_cagr', 'y22', 'y23', 'y24']: m[c] *= 100
        print('\n--- marginal mean by', dim); print(m.round(2).to_string())
    ok = df[(df.sh > 1.9) & (df.dd > -.25)]
    print('\n%d of %d beat the eval bar' % (len(ok), len(df)))
    print('\n--- beat the bar, ranked by tough Sharpe')
    for r in ok.sort_values('t_sh', ascending=False).head(25).to_dict('records'): print(fmt(r))
    print('\n--- beat the bar, ranked by warm-window drawdown')
    for r in ok.sort_values('warm_dd', ascending=False).head(15).to_dict('records'): print(fmt(r))
