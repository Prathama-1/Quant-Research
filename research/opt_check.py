"""Side test, not part of the submission: are task1.py and task2.py 'fully optimised', and would optimising further help?

A higher backtest Sharpe does not answer this, so the main test is combinatorially symmetric cross-validation (CSCV, Bailey et al.):
split the days into blocks, pick the best config on half the blocks, score it on the other half, and repeat for every split.
  PBO: how often the in-sample winner lands in the bottom half out of sample (the probability of backtest overfitting).
  The test that matters here: out of sample, does re-picking the winner beat just keeping the submitted config?
Also checks whether in-sample Sharpe predicts the out-of-sample or tough-case Sharpe across configs, and tests structural ideas from
colab.py on the same stress cases.

    python opt_check.py t1      a few minutes
    python opt_check.py t2      about 10 minutes
"""
import importlib.util
import itertools
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import H, RESULTS, ROOT, Clock, hm, t1_data

pd.set_option('display.width', 220, 'display.max_columns', 40, 'display.max_rows', 200)


def sharpe(r):
    s = r.std(axis=0, ddof=1)
    return np.where(s > 0, r.mean(axis=0) / np.where(s > 0, s, 1) * 252 ** .5, 0.0)


def cscv(R, sub, S):
    """R: days x configs of daily returns. sub: column of the submitted config."""
    blocks = np.array_split(np.arange(len(R)), S)
    rows = []
    for c in itertools.combinations(range(S), S // 2):
        is_ = np.concatenate([blocks[i] for i in c])
        oos = np.concatenate([blocks[i] for i in range(S) if i not in c])
        si, so = sharpe(R[is_]), sharpe(R[oos])
        best = int(np.argmax(si))
        rows.append(dict(best_is=si[best], best_oos=so[best], best_rank=(so < so[best]).mean(), sub_is=si[sub], sub_oos=so[sub], sub_rank_is=(si < si[sub]).mean(), sub_rank_oos=(so < so[sub]).mean(), med_oos=np.median(so), top_oos=so.max()))
    x = pd.DataFrame(rows)
    print('CSCV: %d blocks, %d splits, %d configs' % (S, len(x), R.shape[1]))
    print(' Probability of backtest overfitting (in-sample winner lands in bottom half out of sample): %.0f%%' % (100 * (x.best_rank < .5).mean()))
    print(' Out-of-sample Sharpe, mean across splits:')
    print('   re-pick the in-sample winner each time: %.2f  (in-sample it showed %.2f)' % (x.best_oos.mean(), x.best_is.mean()))
    print('   keep the submitted config:              %.2f  (in-sample it showed %.2f)' % (x.sub_oos.mean(), x.sub_is.mean()))
    print('   the median config in the grid:          %.2f' % x.med_oos.mean())
    print('   hindsight: best config picked after seeing the test half: %.2f' % x.top_oos.mean())
    print(' Winner beat the submitted config out of sample in %.0f%% of splits, by %+.2f Sharpe on average' % (100 * (x.best_oos > x.sub_oos).mean(), (x.best_oos - x.sub_oos).mean()))
    print(' Submitted config percentile within the grid: in-sample %.0f, out-of-sample %.0f' % (100 * x.sub_rank_is.mean(), 100 * x.sub_rank_oos.mean()))
    return x


def boot_diff(a, b, n=4000, block=20, seed=5):
    """Stationary block bootstrap of Sharpe(a) - Sharpe(b) on paired daily returns."""
    rng = np.random.default_rng(seed)
    N = len(a); out = np.empty(n)
    for i in range(n):
        idx = np.empty(N, int); j = rng.integers(N)
        for k in range(N):
            idx[k] = j
            j = rng.integers(N) if rng.random() < 1 / block else (j + 1) % N
        out[i] = sharpe(a[idx]) - sharpe(b[idx])
    return np.percentile(out, [2.5, 50, 97.5]), float((out > 0).mean())


# ----------------------------------------------------------------------------------------------- task 1
def load_t1():
    spec = importlib.util.spec_from_file_location('task1', os.path.join(ROOT, 'task1.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def fast_daily(bars, want, ex_arr, cost=H.T1_COST):
    """task1_backtest's daily return, without its per-trade loop. Checked against the harness below."""
    o, c = bars.open.values, bars.close.values
    pos = np.zeros(len(c)); pos[1:] = want[:-1]; pos[:H.T1_WARMUP] = 0
    prev = np.r_[0.0, pos[:-1]]
    pc = np.r_[c[0], c[:-1]]
    day = bars.index.normalize()
    newday = np.r_[False, day[1:] != day[:-1]]
    prev_day = np.r_[day[:1], day[:-1]]
    is_exp = np.isin(prev_day, ex_arr)
    contrib = prev * (o / pc - 1) + pos * (c / o - 1) - np.abs(pos - prev) * cost
    contrib -= np.where(newday, prev * (H.R_CASH - H.DIV_YIELD) / 252 + is_exp * 2 * cost * np.abs(prev), 0.0)
    return pd.Series(contrib, index=day).groupby(level=0).sum()


def metrics(d, rf=H.R_CASH):
    r = (d + rf / 252).values
    eq = np.cumprod(1 + r)
    return dict(sharpe=float(((r - rf / 252).mean()) / r.std(ddof=1) * 252 ** .5), dd=float((eq / np.maximum.accumulate(eq) - 1).min()), cagr=float(eq[-1] ** (252 / len(r)) - 1))


def delayed(w, k):
    return w if k == 0 else np.r_[np.zeros(k), w[:-k]]


def t1_variants(bars, T, P):
    """Structural ideas from colab.py, built from the submitted rules. Returns name -> want array."""
    ck = Clock(bars)
    base = np.asarray(T.positions(bars, dict(P)), float)
    dec = hm(P['decide_at']); last = hm('15:15')
    days = bars.index.normalize()
    on_day = np.zeros(ck.day_id[-1] + 1, bool)
    on_day[ck.day_id[base > 0]] = True
    has_dec = np.zeros(ck.day_id[-1] + 1, bool)
    has_dec[ck.day_id[(ck.tod == dec)]] = True
    out = {'Submitted': base}
    w = base.copy(); w[np.asarray(days.dayofweek) == 3] = 0
    out['Skip Thursday nights (1.8)'] = w
    w = np.zeros(len(base)); h = 0.0
    for i in range(len(base)):
        d = ck.day_id[i]
        if ck.tod[i] == dec and on_day[d]: h = 1.0
        if ck.tod[i] == last and has_dec[d] and not on_day[d]: h = 0.0
        w[i] = h
    out['Stay long through the day while the signal holds'] = w
    w = base.copy(); short = False
    for i in range(len(base)):
        d = ck.day_id[i]
        if ck.tod[i] == last and on_day[d]: short = True
        if ck.tod[i] == dec: short = False
        if short and base[i] == 0: w[i] = -1.0
    out['Long nights, short the next session (1.1)'] = w
    w = base.copy()
    w[(ck.tod == last) & on_day[ck.day_id]] = -1.0
    out['Exit at the open and short the first bar (1.3)'] = w
    return out


def run_t1():
    bars, ex = t1_data()
    ex_arr = pd.DatetimeIndex(sorted(ex))
    T = load_t1(); P = dict(T.PARAMS)
    base = np.asarray(T.positions(bars, dict(P)), float)
    official = H.task1_backtest(bars, base, ex)['daily']
    fast = fast_daily(bars, base, ex_arr)
    print('fast accounting equals the harness: max difference %.2e' % np.abs(official.values - fast.values).max())

    from synth import extended
    worlds = []
    for sd in [11, 12, 13]:
        b, e, st = extended(sd)
        worlds.append((b, pd.DatetimeIndex(sorted(e)), pd.Timestamp(st)))

    MA, FLOOR, DEC, CAP = [10, 15, 20, 25, 30, 40, 50, 100], [-9., -.0075, -.005, -.0025, 0.], ['14:00', '14:15', '14:30', '14:45'], [.25, .30, .35, .45, 9.9]
    rows, R = [], []
    for ma, fl, de, cp in itertools.product(MA, FLOOR, DEC, CAP):
        p = dict(P, ma_days=ma, session_floor=fl, decide_at=de, vol_cap=cp)
        w = np.asarray(T.positions(bars, p), float)
        d = fast_daily(bars, w, ex_arr)
        m = metrics(d)
        oo = []
        for b, e, st in worlds:
            dd = fast_daily(b, np.asarray(T.positions(b, p), float), e)
            oo.append(metrics(dd[dd.index >= st])['sharpe'])
        rows.append(dict(ma=ma, floor=fl, dec=de, cap=cp, sharpe=m['sharpe'], dd=m['dd'], cagr=m['cagr'], d1=metrics(fast_daily(bars, delayed(w, 1), ex_arr))['sharpe'], d2=metrics(fast_daily(bars, delayed(w, 2), ex_arr))['sharpe'],
                         c2=metrics(fast_daily(bars, w, ex_arr, cost=2 * H.T1_COST))['sharpe'], oos=float(np.mean(oo))))
        R.append(d.values)
    g = pd.DataFrame(rows); R = np.array(R).T
    g.to_csv(os.path.join(RESULTS, 'opt_t1_grid.csv'), index=False)
    sub = g[(g.ma == P['ma_days']) & (g['floor'] == P['session_floor']) & (g.dec == P['decide_at']) & (g.cap == P['vol_cap'])].index[0]
    s = g.loc[sub]
    print('\n=== Grid of %d configs. Submitted: Sharpe %.2f, DD %.1f%%, CAGR %.1f%%, d1 %.2f, d2 %.2f, 2x %.2f, OOS %.2f' % (len(g), s.sharpe, 100 * s.dd, 100 * s.cagr, s.d1, s.d2, s.c2, s.oos))
    print('Submitted percentile within the grid (100 = best):')
    for c in ['sharpe', 'dd', 'cagr', 'd1', 'd2', 'c2', 'oos']:
        print('  %-7s %3.0f   grid range %.2f to %.2f' % (c, 100 * (g[c] < s[c]).mean(), g[c].min(), g[c].max()))
    print('\nHighest in-sample Sharpe configs:'); print(g.sort_values('sharpe', ascending=False).head(8).round(3).to_string(index=False))
    print('\nHighest out-of-sample Sharpe configs (hindsight, for reference):'); print(g.sort_values('oos', ascending=False).head(5).round(3).to_string(index=False))
    g['worst'] = g[['sharpe', 'd1', 'd2', 'c2', 'oos']].min(axis=1)
    print('\nBest worst-case across base, d1, d2, 2x and OOS:'); print(g.sort_values('worst', ascending=False).head(5).round(3).to_string(index=False))
    print('Submitted worst-case: %.2f, percentile %.0f' % (g.loc[sub, 'worst'], 100 * (g.worst < g.loc[sub, 'worst']).mean()))
    keep = g.dec != '14:45'
    print('\nDoes in-sample Sharpe predict out-of-sample Sharpe across configs? Spearman: %.2f (all), %.2f (excluding 14:45)' % (g.sharpe.corr(g.oos, method='spearman'), g[keep].sharpe.corr(g[keep].oos, method='spearman')))
    print('Does 2017-19 Sharpe predict 2020-Jan 2022 Sharpe?')
    day_idx = pd.DatetimeIndex(sorted(set(bars.index.normalize())))
    a, b = day_idx <= pd.Timestamp('2019-12-31'), day_idx > pd.Timestamp('2019-12-31')
    sa, sb = sharpe(R[a]), sharpe(R[b])
    print('  Spearman %.2f' % pd.Series(sa).corr(pd.Series(sb), method='spearman'))
    bd = int(np.argmax(sa))
    print('  Best on 2017-19 (%s): 2017-19 Sharpe %.2f, then 2020-Jan 2022 %.2f, OOS extension %.2f' % (g.loc[bd, ['ma', 'floor', 'dec', 'cap']].tolist(), sa[bd], sb[bd], g.loc[bd, 'oos']))
    print('  Submitted:                            2017-19 Sharpe %.2f, then 2020-Jan 2022 %.2f, OOS extension %.2f' % (sa[sub], sb[sub], g.loc[sub, 'oos']))
    print('\n=== Cross-validation on the real 2017 to Jan 2022 data')
    cscv(R, sub, 10)
    print('  Same, excluding the 14:45 decision, which fails the 2-bar delay:')
    idx = np.flatnonzero(keep)
    cscv(R[:, idx], int(np.flatnonzero(idx == sub)[0]), 10)

    print('\n=== Structural ideas from colab.py, each scored on the same tests')
    V = t1_variants(bars, T, P)
    Vo = [t1_variants(b, T, P) for b, _, _ in worlds]
    rows = []
    for nm, w in V.items():
        d = fast_daily(bars, w, ex_arr); m = metrics(d)
        oo = []
        for k, (b, e, st) in enumerate(worlds):
            dd = fast_daily(b, Vo[k][nm], e)
            oo.append(metrics(dd[dd.index >= st])['sharpe'])
        rows.append(dict(idea=nm, sharpe=m['sharpe'], dd=100 * m['dd'], cagr=100 * m['cagr'], d1=metrics(fast_daily(bars, delayed(w, 1), ex_arr))['sharpe'], d2=metrics(fast_daily(bars, delayed(w, 2), ex_arr))['sharpe'],
                         c2=metrics(fast_daily(bars, w, ex_arr, cost=2 * H.T1_COST))['sharpe'], oos=float(np.mean(oo))))
    v = pd.DataFrame(rows); v.to_csv(os.path.join(RESULTS, 'opt_t1_ideas.csv'), index=False)
    print(v.round(2).to_string(index=False))
    print(' The OOS column is unreliable for first-bar ideas, because its intraday paths are synthetic and have no first-bar drop.')
    base_d = fast_daily(bars, base, ex_arr).values
    for nm, w in V.items():
        if nm == 'Submitted': continue
        ci, pos = boot_diff(fast_daily(bars, w, ex_arr).values, base_d)
        print(' %-52s Sharpe difference vs submitted, 95%% CI %+.2f to %+.2f, P(better) %.0f%%' % (nm, ci[0], ci[2], 100 * pos))
    b_ = g.sharpe.idxmax()
    w = np.asarray(T.positions(bars, dict(P, ma_days=int(g.ma[b_]), session_floor=float(g['floor'][b_]), decide_at=g.dec[b_], vol_cap=float(g.cap[b_]))), float)
    ci, pos = boot_diff(fast_daily(bars, w, ex_arr).values, base_d)
    print(' %-52s Sharpe difference vs submitted, 95%% CI %+.2f to %+.2f, P(better) %.0f%%' % ('Highest in-sample grid config', ci[0], ci[2], 100 * pos))


# ----------------------------------------------------------------------------------------------- task 2
SCORES = ['sh126', 'mom_over_vol', 'mix_sh', 'mom126_21', 'res126', 'ma200gap', 'mom63', 'mom252', 'sh_tv', 'mv_tv']


def t2_cfg(score, band, every, buffer, mv, br, K=15):
    from t2_lab import BASE
    p = dict(BASE, score=score, every=every, buffer=buffer, off='sell', breakout=br, extra=10, fill_daily=True, band=band if isinstance(band, tuple) else (band, band), K=K)
    if mv: p['maxvol'] = mv
    return p


def t2_job(c):
    from t2_lab import run
    p = t2_cfg(**c)
    e = c['every']
    rs, sh, dd, cg, ts, td, tc = [], [], [], [], [], [], []
    for o in [0, e // 4, e // 2, 3 * e // 4]:
        s, res = run(dict(p, offset=o))
        rs.append(res['eq'].pct_change().dropna().values)
        sh.append(s['sharpe']); dd.append(s['max_dd']); cg.append(s['cagr'])
        t_, _ = run(dict(p, offset=o), 'tough')
        ts.append(t_['sharpe']); td.append(t_['max_dd']); tc.append(t_['cagr'])
    w_, _ = run(p, 'warm')
    out = dict(c); out.update(sharpe=np.mean(sh), sh_min=np.min(sh), dd=np.mean(dd), cagr=np.mean(cg), tough=np.mean(ts), tough_dd=np.mean(td), tough_cagr=np.mean(tc), warm=w_['total_return'], warm_dd=w_['max_dd'], sharpe_blend=float(sharpe(np.mean(rs, axis=0))))
    return out, np.mean(rs, axis=0)


def run_t2():
    from t2_lab import world
    grid = [dict(score=a, band=b, every=c, buffer=d, mv=e, br=f) for a, b, c, d, e, f in itertools.product(SCORES, [.5, .55, .6, .65, .7], [10, 21], [25, 40], [0, .65], ['hi10', 'hi20'])]
    with Pool(4, initializer=world) as pool:
        out = pool.map(t2_job, grid, chunksize=2)
        extra = [dict(score='sh126', band=.6, every=10, buffer=b, mv=.65, br='hi10', K=k) for k in [8, 10, 12] for b in [20, 40]]
        out_x = pool.map(t2_job, extra, chunksize=1)
        nofilter = pool.map(t2_job, [dict(score='sh126', band=0.0, every=10, buffer=40, mv=.65, br='hi10')], chunksize=1)
        chop = pool.map(t2_job, [dict(score='sh126', band=b, every=10, buffer=40, mv=.65, br='hi10') for b in [(.55, .6), (.5, .6), (.55, .65), (.5, .65), (.5, .7)]], chunksize=1)
    g = pd.DataFrame([x[0] for x in out]); R = np.array([x[1] for x in out]).T
    from common import t2_data
    all_days = t2_data()['close'].index
    days_ = all_days[all_days >= pd.Timestamp('2022-03-02')][1:]
    late = days_ >= pd.Timestamp('2024-02-01')
    g['ret_feb_jun24'] = np.prod(1 + R[late], axis=0) - 1
    g.to_csv(os.path.join(RESULTS, 'opt_t2_grid.csv'), index=False)
    sub = g[(g.score == 'sh126') & (g.band == .6) & (g.every == 10) & (g.buffer == 40) & (g.mv == .65) & (g.br == 'hi10')].index[0]
    s = g.loc[sub]
    print('Means over 4 rebalance offsets. Eval window 2 Mar 2022 to 30 Jun 2024.')
    print('=== Grid of %d configs. Submitted: Sharpe %.2f (blend of offsets %.2f), DD %.1f%%, CAGR %.1f%%, tough %.2f, tough DD %.1f%%, warm %.1f%%' % (len(g), s.sharpe, s.sharpe_blend, 100 * s.dd, 100 * s.cagr, s.tough, 100 * s.tough_dd, 100 * s.warm))
    print('Submitted percentile within the grid (100 = best):')
    for c in ['sharpe', 'dd', 'cagr', 'tough', 'tough_dd', 'warm']:
        print('  %-9s %3.0f   grid range %.2f to %.2f' % (c, 100 * (g[c] < s[c]).mean(), g[c].min(), g[c].max()))
    show = ['score', 'band', 'every', 'buffer', 'mv', 'br', 'sharpe', 'sh_min', 'dd', 'cagr', 'tough', 'tough_dd', 'warm', 'warm_dd']
    print('\nHighest eval Sharpe:'); print(g.sort_values('sharpe', ascending=False).head(8)[show].round(3).to_string(index=False))
    print('\nHighest tough Sharpe:'); print(g.sort_values('tough', ascending=False).head(5)[show].round(3).to_string(index=False))
    g['z'] = g.sharpe.rank(pct=True) + g.tough.rank(pct=True) + g.warm.rank(pct=True) + g.dd.rank(pct=True) + g.tough_dd.rank(pct=True)
    print('\nBest combined rank across eval, tough, warm and both drawdowns:'); print(g.sort_values('z', ascending=False).head(6)[show].round(3).to_string(index=False))
    print('Submitted combined-rank percentile: %.0f' % (100 * (g.z < g.z[sub]).mean()))
    print('\nDoes eval Sharpe predict tough Sharpe across configs? Spearman %.2f. And the warm window? %.2f' % (g.sharpe.corr(g.tough, method='spearman'), g.sharpe.corr(g.warm, method='spearman')))
    for dim in ['score', 'band', 'every', 'buffer', 'mv', 'br']:
        m = g.groupby(dim)[['sharpe', 'dd', 'cagr', 'tough', 'tough_dd', 'warm']].mean()
        print('\n--- mean by', dim); print(m.round(3).to_string())
    print('\n=== The weak stretch, Feb to Jun 2024 (universe +17.8%%). Submitted %.1f%%. Grid: median %.1f%%, best %.1f%%, share of configs above +10%%: %.0f%%' % (
        100 * g.ret_feb_jun24[sub], 100 * g.ret_feb_jun24.median(), 100 * g.ret_feb_jun24.max(), 100 * (g.ret_feb_jun24 > .1).mean()))
    for dim in ['band', 'score', 'every', 'buffer']:
        print(' mean by %s: %s' % (dim, (g.groupby(dim).ret_feb_jun24.mean() * 100).round(1).to_dict()))
    sf = g[(g.score == 'sh126') & (g.every == 10) & (g.buffer == 40) & (g.mv == .65) & (g.br == 'hi10')]
    print(' submitted rules, breadth threshold only:'); print(sf[['band', 'sharpe', 'dd', 'cagr', 'tough', 'warm', 'ret_feb_jun24']].round(3).to_string(index=False))
    print('\n Hysteresis on breadth, aimed at the whipsaw (exit below the first number, re-enter above the second):')
    rows = []
    for cf, r_ in chop:
        rows.append(dict(band=str(cf['band']), sharpe=cf['sharpe'], dd=cf['dd'], cagr=cf['cagr'], tough=cf['tough'], warm=cf['warm'], ret_feb_jun24=np.prod(1 + r_[late]) - 1))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print('\n=== Cross-validation on the eval window')
    cscv(R, sub, 8)
    idx = np.flatnonzero(g.band.values <= .65)
    print('  Same, excluding the 0.70 threshold:')
    cscv(R[:, idx], int(np.flatnonzero(idx == sub)[0]), 8)
    days = pd.date_range('2000-01-01', periods=0)
    n = len(R); cut = int(n * .7)
    si, so = sharpe(R[:cut]), sharpe(R[cut:])
    b = int(np.argmax(si))
    print('\nTime split, first 70%% then last 30%% of the window. Spearman %.2f' % pd.Series(si).corr(pd.Series(so), method='spearman'))
    print('  Best on the first part (%s): %.2f, then %.2f' % (g.loc[b, ['score', 'band', 'every', 'buffer', 'mv', 'br']].tolist(), si[b], so[b]))
    print('  Submitted:                   %.2f, then %.2f' % (si[sub], so[sub]))
    print('\n=== Fewer positions or a tighter sell rule, on the submitted rules')
    x = pd.DataFrame([a[0] for a in out_x])
    print(x[['K', 'buffer', 'sharpe', 'sh_min', 'dd', 'cagr', 'tough', 'tough_dd', 'warm']].round(3).to_string(index=False))
    base = R[:, sub]
    print('\nSharpe difference vs submitted, block bootstrap on daily returns:')
    for (cf, r_) in out_x:
        ci, pos = boot_diff(r_, base)
        print('  K %2d, sell rank %2d:  95%% CI %+.2f to %+.2f, P(better) %.0f%%' % (cf['K'], cf['buffer'], ci[0], ci[2], 100 * pos))
    b_ = int(np.argmax(g.sharpe.values))
    ci, pos = boot_diff(R[:, b_], base)
    print('  Highest-Sharpe grid config: 95%% CI %+.2f to %+.2f, P(better) %.0f%%' % (ci[0], ci[2], 100 * pos))
    ci, pos = boot_diff(base, nofilter[0][1])
    print('  Submitted vs the same rules with no breadth filter: 95%% CI %+.2f to %+.2f, P(better) %.0f%%' % (ci[0], ci[2], 100 * pos))
    for sc in ['ma200gap', 'mom252', 'mom63', 'mom_over_vol']:
        j = g[(g.score == sc) & (g.band == .6) & (g.every == 10) & (g.buffer == 40) & (g.mv == .65) & (g.br == 'hi10')].index[0]
        ci, pos = boot_diff(R[:, j], base)
        print('  Signal %-13s on the submitted rules: Sharpe %.2f, tough %.2f | 95%% CI %+.2f to %+.2f, P(better) %.0f%%' % (sc, g.sharpe[j], g.tough[j], ci[0], ci[2], 100 * pos))


if __name__ == '__main__':
    (run_t1 if sys.argv[1] == 't1' else run_t2)()
