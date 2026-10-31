"""First Task 2 sweep: benchmarks and single-signal portfolios with no risk overlay, then one overlay at a time."""
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from t2_lab import BASE, LOG, fmt, run, score, sweep, world


def run_random(seed):
    s, _ = run(dict(BASE, score='random'), seed=seed)
    s2, _ = run(dict(BASE, score='random'), 'tough', seed=seed)
    s3, _ = run(dict(BASE, score='random'), 'warm', seed=seed)
    return dict(seed=seed, sh=s['sharpe'], cagr=s['cagr'], dd=s['max_dd'], t_sh=s2['sharpe'], t_cagr=s2['cagr'], warm=s3['total_return'], warm_sh=s3['sharpe'])


if __name__ == '__main__':
    if os.path.exists(LOG): os.remove(LOG)
    world()
    with Pool(3, initializer=world) as pool:
        print('=== random 15 names monthly, 24 seeds: what selection has to beat')
        r = pd.DataFrame(pool.map(run_random, range(24)))
        print(r[['sh', 'cagr', 'dd', 't_sh', 't_cagr', 'warm', 'warm_sh']].describe().loc[['mean', 'std', 'min', '50%', 'max']].round(3).to_string())
        r.to_csv('results/t2_random.csv', index=False)

        sweep('naive: top-10 by 6m return, monthly (brief says it fails OOS)', [dict(BASE, K=10)], pool)
        sigs = ['mom63', 'mom126', 'mom252', 'mom126_21', 'mom252_21', 'sh126', 'sh252', 'res126', 'res252', 'hi252', 'ma200gap', 'tvshock', 'lowvol', 'mom_over_vol', 'rank_mom', 'mix_mr', 'mix_mrt', 'mix_rt', 'mix_sh']
        sweep('signal, K=15, monthly, no overlay', [dict(BASE, score=s) for s in sigs], pool)
        sweep('signal, K=15, monthly, buffer 45', [dict(BASE, score=s, buffer=45) for s in sigs], pool)
