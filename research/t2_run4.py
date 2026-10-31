"""Task 2 random search with timing luck averaged out. Shifting the rebalance day alone moves Sharpe by about 0.3, so each config is
run at 3 offsets, on eval, tough and warm, and only the means are compared. Read as marginal effects over many configs."""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from t2_lab import BASE, H, RESULTS, run, world

OUT = os.path.join(RESULTS, 't2_run4.csv')
SPACE = {
    'score': ['mom126_21', 'sh126', 'res126', 'mix_sh', 'mom_over_vol', 'msh_blend', 'mv_tv', 'sh_tv', 'skip_blend'],
    'every': [5, 10, 21],
    'buffer': [15, 25, 35, 50],
    'regime': ['none', 'b50.4', 'b50.5', 'b50.6', 'b200.4', 'b200.5', 'n50', 'n100', 'n200', 'b50.5+n200'],
    'off': ['keep', 'weak', 'sell'],
    'trend': ['none', 'ma50', 'ma50+200', 'ma20+50+200'],
    'maxvol': [0, .55, .65, .8],
    'breakout': ['none', 'hi5', 'hi10', 'hi20'],
    'fill': [False, True],
    'liq': [1e8, 2.5e8],
}
REG = {'none': None, 'b50.4': {'breadth50': .4}, 'b50.5': {'breadth50': .5}, 'b50.6': {'breadth50': .6}, 'b200.4': {'breadth200': .4}, 'b200.5': {'breadth200': .5},
       'n50': {'nifty_ma50': 0}, 'n100': {'nifty_ma100': 0}, 'n200': {'nifty_ma200': 0}, 'b50.5+n200': {'breadth50': .5, 'nifty_ma200': 0}}
TREND = {'none': None, 'ma50': ['ma50gap'], 'ma50+200': ['ma50gap', 'ma200gap'], 'ma20+50+200': ['ma20gap', 'ma50gap', 'ma200gap']}


def to_p(c):
    p = dict(BASE, score=c['score'], every=c['every'], buffer=c['buffer'], liq=c['liq'])
    if REG[c['regime']]: p['regime'] = REG[c['regime']]; p['off'] = c['off']
    if TREND[c['trend']]: p['trend'] = TREND[c['trend']]
    if c['maxvol']: p['maxvol'] = c['maxvol']
    if c['breakout'] != 'none': p['breakout'] = c['breakout']; p['extra'] = 10
    if c['fill'] or c['breakout'] != 'none': p['fill_daily'] = True
    return p


def job(c):
    p = to_p(c)
    e = c['every']
    offs = [0, e // 3, 2 * e // 3]
    out = dict(c)
    for name in ['eval', 'tough', 'warm']:
        v = []
        for o in offs:
            s, _ = run(dict(p, offset=o), name)
            v.append((s['sharpe'], s['max_dd'], s['cagr'], s['trades_per_yr'], s['time_in_market'], s['total_return'], s['yearly'].get('2022', np.nan)))
        v = np.array(v, float)
        out[name + '_sh'], out[name + '_dd'], out[name + '_cagr'], out[name + '_tr'], out[name + '_tim'], out[name + '_ret'], out[name + '_y22'] = v.mean(axis=0)
        out[name + '_sh_min'], out[name + '_dd_min'] = v[:, 0].min(), v[:, 1].min()
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    rng = np.random.default_rng(2024)
    cfgs = [{k: v[rng.integers(len(v))] for k, v in SPACE.items()} for _ in range(n)]
    for c in cfgs:
        c['every'], c['buffer'], c['maxvol'], c['liq'] = int(c['every']), int(c['buffer']), float(c['maxvol']), float(c['liq']); c['fill'] = bool(c['fill'])
    world()
    t0 = time.time()
    with Pool(3, initializer=world) as pool:
        rows = pool.map(job, cfgs, chunksize=2)
    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    print('configs', len(df), 'minutes %.1f' % ((time.time() - t0) / 60))
    cols = ['eval_sh', 'eval_dd', 'eval_cagr', 'eval_tr', 'eval_tim', 'tough_sh', 'tough_dd', 'tough_cagr', 'warm_ret', 'warm_dd', 'eval_y22']
    for dim in SPACE:
        m = df.groupby(dim)[cols].mean()
        for c in ['eval_dd', 'eval_cagr', 'eval_tim', 'tough_dd', 'tough_cagr', 'warm_ret', 'warm_dd', 'eval_y22']: m[c] *= 100
        m['n'] = df.groupby(dim).size()
        print('\n--- marginal by', dim); print(m.round(2).to_string())
