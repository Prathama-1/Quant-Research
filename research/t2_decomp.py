"""What does each layer add? Same rules as task2.py, swapping the layers on and off. Selection-free versions use one fixed random
ranking per seed, so they have no forced turnover. Means over 24 seeds. Also: bootstrap confidence intervals and a Sharpe deflated
for the number of configs tried, for both tasks."""
import glob
import importlib.util
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats as st

from common import H, RESULTS, ROOT, t1_data, t2_data
from t2_lab import BASE, run as lab_run, world

R = dict(BASE, every=10, buffer=40, breakout='hi10', extra=10, fill_daily=True, maxvol=.65)


def job(seed):
    out = {}
    for name, p in [('random_fixed_regime', dict(R, score='random_fixed', off='sell', band=(.6, .6))), ('random_fixed_noregime', dict(R, score='random_fixed'))]:
        s, _ = lab_run(p, seed=seed)
        t, _ = lab_run(p, 'tough', seed=seed)
        out.update({name + '_sh': s['sharpe'], name + '_cagr': s['cagr'], name + '_dd': s['max_dd'], name + '_tr': s['trades_per_yr'], name + '_tsh': t['sharpe']})
    return out


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, name + '.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def boot_sharpe(r, n=4000, block=20, seed=5):
    """Stationary block bootstrap of the daily return series."""
    rng = np.random.default_rng(seed)
    r = np.asarray(r); N = len(r); out = np.empty(n)
    for i in range(n):
        idx = np.empty(N, int); j = rng.integers(N)
        for k in range(N):
            idx[k] = j
            j = rng.integers(N) if rng.random() < 1 / block else (j + 1) % N
        x = r[idx]
        out[i] = x.mean() / x.std(ddof=1) * 252 ** .5
    return np.percentile(out, [2.5, 50, 97.5]), float((out > 0).mean())


def deflated(r, trial_sharpes, n_trials):
    """Bailey and Lopez de Prado. Sharpe inputs annualised, converted to per-day."""
    r = np.asarray(r); T = len(r)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = st.skew(r), st.kurtosis(r, fisher=False)
    v = np.var(np.asarray(trial_sharpes) / 252 ** .5, ddof=1)
    em = 0.5772156649
    sr0 = v ** .5 * ((1 - em) * st.norm.ppf(1 - 1 / n_trials) + em * st.norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * (T - 1) ** .5 / (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) ** .5
    return dict(sr_ann=float(sr * 252 ** .5), sr0_ann=float(sr0 * 252 ** .5), dsr=float(st.norm.cdf(z)), n_trials=n_trials, skew=float(g3), kurt=float(g4))


if __name__ == '__main__':
    OUT = {}
    with Pool(3, initializer=world) as pool:        # each worker holds full feature panels: 8 workers can exhaust an 8 GB laptop
        rows = pd.DataFrame(pool.map(job, range(24), chunksize=1))
    print(rows.describe().loc[['mean', 'std', 'min', 'max']].round(3).T.to_string())
    OUT['decomp'] = {c: float(rows[c].mean()) for c in rows.columns}
    OUT['decomp'].update({c + '_std': float(rows[c].std()) for c in rows.columns if c.endswith('_sh')})

    bars, ex = t1_data()
    m1 = load('task1')
    res = H.task1_backtest(bars, m1.positions(bars, dict(m1.PARAMS)).values, ex)
    d1 = (res['daily'] + 0.0).iloc[:]                                        # excess of cash, exactly what the harness Sharpe measures
    ci, pos = boot_sharpe(d1.values)
    print('\nT1 Sharpe CI 95%%: %.2f to %.2f (median %.2f), P(Sharpe>0) %.3f' % (ci[0], ci[2], ci[1], pos))
    OUT['t1_ci'] = dict(lo=float(ci[0]), med=float(ci[1]), hi=float(ci[2]))
    t1s = pd.concat([pd.read_csv(os.path.join(RESULTS, f)) for f in ['t1_lab2.csv', 't1_lab3.csv']])
    t1l = pd.read_csv(os.path.join(RESULTS, 't1_lab.csv'))
    all_t1 = np.r_[t1s.sh.values, t1l.sharpe.values]
    OUT['t1_dsr'] = deflated(d1.values, all_t1, len(all_t1))
    print('T1 deflated:', OUT['t1_dsr'])

    D = t2_data()
    m2 = load('task2')
    r2 = H.task2_backtest(m2, D, dict(m2.PARAMS), H.T2_START, H.T2_END)
    d2 = r2['eq'].pct_change().dropna()
    ci, pos = boot_sharpe(d2.values)
    print('\nT2 Sharpe CI 95%%: %.2f to %.2f (median %.2f), P(Sharpe>0) %.3f' % (ci[0], ci[2], ci[1], pos))
    OUT['t2_ci'] = dict(lo=float(ci[0]), med=float(ci[1]), hi=float(ci[2]))
    t2s = pd.concat([pd.read_csv(os.path.join(RESULTS, 't2_run2.csv'))[['sh']], pd.read_csv(os.path.join(RESULTS, 't2_run4.csv'))[['eval_sh']].rename(columns={'eval_sh': 'sh'}), pd.read_csv(os.path.join(RESULTS, 't2_run5.csv'))[['eval_sh']].rename(columns={'eval_sh': 'sh'})])
    OUT['t2_dsr'] = deflated(d2.values, t2s.sh.values, len(t2s))
    print('T2 deflated:', OUT['t2_dsr'])
    print('configs logged: T1 %d, T2 %d' % (len(all_t1), len(t2s)))
    # daily return correlation and equity curves for the report
    r2['eq'].to_csv(os.path.join(RESULTS, 't2_equity.csv'), header=['equity'])
    (1 + res['daily'] + H.R_CASH / 252).cumprod().to_csv(os.path.join(RESULTS, 't1_equity.csv'), header=['equity'])
    n = D['nifty'].close.reindex(r2['eq'].index); (n / n.iloc[0]).to_csv(os.path.join(RESULTS, 't2_nifty.csv'), header=['nifty'])
    json.dump(OUT, open(os.path.join(RESULTS, 't_decomp.json'), 'w'), indent=1, default=float)
