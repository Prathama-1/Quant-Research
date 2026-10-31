"""Task 2, one design choice at a time around a reference design, then a factorial of the choices that matter.
Reference design: skip-month 6m momentum, K=15, monthly, sell-buffer 45, no new buys when breadth is low."""
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from t2_lab import BASE, RESULTS, fmt, score, sweep, world

R = dict(BASE, score='mom126_21', buffer=45, regime={'breadth50': .4})

if __name__ == '__main__':
    world()
    with Pool(3, initializer=world) as pool:
        sweep('reference', [R], pool)
        sweep('signal', [dict(R, score=s) for s in ['mom126', 'mom126_21', 'mom189_21', 'mom252_21', 'skip_blend', 'res126', 'skip_res', 'skip_res_tv', 'res_vs_vol', 'sh126', 'mix_sh', 'mom_over_vol']], pool)
        sweep('absolute trend at entry', [dict(R, trend=t) for t in ['ma50gap', 'ma100gap', 'ma200gap', 'mom126_pos', ['ma50gap', 'ma200gap'], ['ma20gap', 'ma50gap', 'ma200gap']]], pool)
        sweep('per-name trend exit', [dict(R, ma_exit=m) for m in ['ma20gap', 'ma50gap', 'ma100gap', 'ma200gap']], pool)
        sweep('regime', [dict(R, regime=r) for r in [{'breadth50': .3}, {'breadth50': .4}, {'breadth50': .5}, {'breadth50': .6}, {'breadth200': .4}, {'breadth200': .5}, {'nifty_ma50': 0}, {'nifty_ma100': 0}, {'nifty_ma200': 0}, {'ew_ma50': 0}, {'ew_ma100': 0}, {'nifty_ma100': 0, 'breadth50': .4}]], pool)
        sweep('regime, sell when off', [dict(R, regime=r, off='sell') for r in [{'breadth50': .3}, {'breadth50': .4}, {'nifty_ma100': 0}, {'nifty_ma200': 0}, {'ew_ma100': 0}, {'breadth200': .4}]], pool)
        sweep('rebalance + sell buffer', [dict(R, every=e, buffer=b) for e in [0, 5, 10, 21] for b in [15, 30, 45, 60, 90]], pool)
        sweep('fill empty slots daily', [dict(R, fill_daily=True), dict(R, fill_daily=True, every=10), dict(R, fill_daily=True, every=21)], pool)
        sweep('K', [dict(R, K=k, buffer=3 * k) for k in [8, 10, 12, 15]], pool)
        sweep('liquidity floor (Rs)', [dict(R, liq=x) for x in [1e8, 2.5e8, 5e8, 1e9, 2e9]], pool)
        sweep('price floor', [dict(R, minpx=x) for x in [10, 25, 50, 100]], pool)
        sweep('max vol', [dict(R, maxvol=x) for x in [.35, .45, .55, .65]], pool)
        sweep('max beta', [dict(R, maxbeta=x) for x in [1.0, 1.25, 1.5, 2.0]], pool)
        sweep('min age', [dict(R, min_age=x) for x in [60, 126, 252]], pool)
        sweep('inverse-vol sizing', [dict(R, ivw=x) for x in [1.25, 1.5, 2.0]], pool)
        sweep('breakout entry', [dict(R, breakout=b, every=e, fill_daily=True, extra=10) for b in ['hi10', 'hi20'] for e in [0, 10]], pool)
