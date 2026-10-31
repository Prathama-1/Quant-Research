"""Tests of the real task2.py through the official harness. Writes results/t2_final.json for the report."""
import importlib.util
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import H, RESULTS, ROOT, t2_data
from t2_lab import EVAL, WARM, strip_drift

_m = {}
_d = {}
OFFS = [0, 2, 5, 7]
SCEN = {'eval': (0, 0, EVAL), 'warm': (0, 0, WARM), 'tough': (.29, .108, EVAL), 'bear': (.49, .21, EVAL)}


def mod(const=None):
    spec = importlib.util.spec_from_file_location('task2_x', os.path.join(ROOT, 'task2.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for k, v in (const or {}).items():
        setattr(m, k, v)
    return m


def data(name='eval'):
    if not _d:
        _d['base'] = t2_data()
    if name in ('eval', 'warm'):
        return _d['base']
    if name not in _d:
        g, n, _ = SCEN[name]
        _d[name] = strip_drift(_d['base'], g, n)
    return _d[name]


def one(params=None, const=None, scen='eval', offset=0, D=None, cost=H.T2_COST):
    m = mod(const)
    p = dict(m.PARAMS); p.update(params or {}); p['offset'] = offset
    a, z = SCEN[scen][2]
    res = H.task2_backtest(m, D if D is not None else data(scen), p, a, z, cost=cost)
    s, e = H.task2_stats(res)
    s['trades_per_yr'] = s['trades'] / max(s['years'], 1e-9)
    return s, res


def job(a):
    params, const, scen, offset = a
    s, _ = one(params, const, scen, offset)
    return dict(sh=s['sharpe'], dd=s['max_dd'], cagr=s['cagr'], tr=s['trades_per_yr'], tim=s['time_in_market'], ret=s['total_return'])


def block(pool, params=None, const=None, scens=('eval', 'tough', 'warm')):
    out = {}
    args = [(params, const, sc, o) for sc in scens for o in OFFS]
    rs = pool.map(job, args, chunksize=1)
    for i, sc in enumerate(scens):
        x = pd.DataFrame(rs[i * len(OFFS):(i + 1) * len(OFFS)])
        out[sc] = dict(x.mean()); out[sc]['sh_min'] = float(x.sh.min()); out[sc]['dd_min'] = float(x.dd.min())
    return out


def ln(name, b):
    e, t, w = b['eval'], b['tough'], b['warm']
    return '%-30s Sh %5.2f (min %5.2f) DD %6.1f CAGR %5.1f tr %4.0f tim %3.0f%% | tough Sh %5.2f DD %6.1f CAGR %5.1f | warm %6.1f%% DD %6.1f' % (
        name, e['sh'], e['sh_min'], 100 * e['dd'], 100 * e['cagr'], e['tr'], 100 * e['tim'], t['sh'], 100 * t['dd'], 100 * t['cagr'], 100 * w['ret'], 100 * w['dd'])


def ghost_panel(D, n, seed, fall=.70, days=60):
    """Add n clones of real stocks that collapse: they track the real name, then lose `fall` over `days` and dry up. This mimics
    what the universe is missing: names that looked fine, then failed and left the list."""
    rng = np.random.default_rng(seed)
    idx = D['close'].index
    t0 = np.searchsorted(idx, pd.Timestamp(EVAL[0]))
    cols = list(D['close'].columns)
    picks = rng.choice(cols, n, replace=False)
    new = {k: D[k].copy() for k in ['open', 'high', 'low', 'close', 'volume']}
    extra = {k: {} for k in new}
    for i, s in enumerate(picks):
        t = int(rng.integers(t0, len(idx) - days - 1))
        f = np.ones(len(idx))
        f[t:t + days] = (1 - fall) ** (np.arange(1, days + 1) / days)
        f[t + days:] = 1 - fall
        name = 'GHOST%03d' % i
        for k in ['open', 'high', 'low', 'close']:
            v = D[k][s].values * f
            v = v.copy(); v[t + days:] = np.nan                     # delisted after the fall: no prices, so a holder cannot sell
            extra[k][name] = v
        vol = D['volume'][s].values * np.where(np.arange(len(idx)) < t, 1.0, .5).astype(float)
        vol = vol.copy(); vol[t + days:] = np.nan
        extra['volume'][name] = vol
    out = {k: pd.concat([new[k], pd.DataFrame(extra[k], index=idx)], axis=1) for k in new}
    out['nifty'] = D['nifty']
    return out


def ghost_job(a):
    n, seed, fall = a
    D = ghost_panel(data(), n, seed, fall)
    s, res = one(D=D)
    T = res['trades']
    return dict(n=n, seed=seed, fall=fall, sh=s['sharpe'], cagr=s['cagr'], dd=s['max_dd'], ghost_trades=int(T.symbol.str.startswith('GHOST').sum()) if len(T) else 0)


def uni_job(seed):
    rng = np.random.default_rng(seed)
    D0 = data()
    keep = list(rng.choice(D0['close'].columns, int(.7 * D0['close'].shape[1]), replace=False))
    D = {k: D0[k][keep] for k in ['open', 'high', 'low', 'close', 'volume']}; D['nifty'] = D0['nifty']
    s, _ = one(D=D)
    return dict(sh=s['sharpe'], cagr=s['cagr'], dd=s['max_dd'], tr=s['trades_per_yr'])


def init():
    data(); data('tough'); data('bear')


def perf(eq):
    r = eq.pct_change().dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return dict(cagr=float((eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1), sharpe=float(r.mean() / r.std() * 252 ** .5), dd=float((eq / eq.cummax() - 1).min()))


if __name__ == '__main__':
    OUT = {}
    t0 = time.time()
    s, res = one()
    s2, res2 = one()
    print('=== official run', {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items() if k != 'yearly'}, s['yearly'])
    print(' deterministic:', bool((res['eq'] == res2['eq']).all() and len(res['trades']) == len(res2['trades'])))
    OUT['base'] = s
    eq, T = res['eq'], res['trades']

    print('\n=== monthly returns %')
    m = eq.resample('ME').last().pct_change().dropna() * 100
    print(m.round(1).to_string())
    print(' positive months %d of %d, worst %.1f%%, best %.1f%%' % ((m > 0).sum(), len(m), m.min(), m.max()))

    # best year removed
    r = eq.pct_change().dropna()
    yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    best = int(yr.idxmax())
    kept = r[r.index.year != best]
    e2 = (1 + kept).cumprod()
    nb = dict(best_year=best, cagr=float(e2.iloc[-1] ** (252 / len(kept)) - 1), sharpe=float(kept.mean() / kept.std() * 252 ** .5), dd=float((e2 / e2.cummax() - 1).min()), total=float(e2.iloc[-1] - 1))
    OUT['ex_best_year'] = nb
    print('\n=== best year removed', nb)

    # trade P&L in portfolio terms: entry amount is 1/k of equity on the day before entry
    k = mod().PARAMS['top_k']
    eqs = eq.reindex(pd.to_datetime(T.entry_date)).values
    prev = eq.shift(1).reindex(pd.to_datetime(T.entry_date)).fillna(1.0).values
    pnl = prev / k * T.ret.values
    tot = pnl.sum(); srt = np.sort(pnl)[::-1]
    OUT['top10'] = dict(share_of_net=float(srt[:10].sum() / tot), share_of_gross=float(srt[:10].sum() / pnl[pnl > 0].sum()), trades=len(pnl), net=float(tot), n_symbols=int(T.symbol.nunique()))
    print('\n=== top 10 trades: %.1f%% of net P&L, %.1f%% of gross profit; %d trades in %d names' % (100 * OUT['top10']['share_of_net'], 100 * OUT['top10']['share_of_gross'], len(pnl), T.symbol.nunique()))
    x = T.assign(pnl=pnl).sort_values('pnl', ascending=False)
    print(x[['symbol', 'entry_date', 'exit_date', 'ret', 'days', 'pnl']].head(10).round({'ret': 3, 'pnl': 3}).to_string())
    print(x[['symbol', 'entry_date', 'exit_date', 'ret', 'days', 'pnl']].tail(5).round({'ret': 3, 'pnl': 3}).to_string())
    by = T.assign(pnl=pnl).groupby('symbol').pnl.sum().sort_values(ascending=False)
    OUT['top_names'] = dict(top5_share=float(by.iloc[:5].sum() / tot), top10_share=float(by.iloc[:10].sum() / tot))
    print(' by name: top 5 names %.1f%%, top 10 names %.1f%% of net' % (100 * OUT['top_names']['top5_share'], 100 * OUT['top_names']['top10_share']))
    ex = pd.Series(pnl, index=T.index).sort_values(ascending=False).index[:10]
    print(' exits by reason:', T.why.value_counts().to_dict())
    print(' trade stats: win %.1f%%, mean %.2f%%, median %.2f%%, best %.1f%%, worst %.1f%%, mean hold %.0f days' % (100 * (T.ret > 0).mean(), 100 * T.ret.mean(), 100 * T.ret.median(), 100 * T.ret.max(), 100 * T.ret.min(), T.days.mean()))

    s0, _ = one(cost=0.0)
    OUT['cost_drag'] = dict(gross_cagr=s0['cagr'], net_cagr=s['cagr'], turnover=s['turnover_per_yr'])
    print('\n=== zero-cost CAGR %.2f%%, net %.2f%%, drag %.2f%%/yr, turnover %.1fx/yr' % (100 * s0['cagr'], 100 * s['cagr'], 100 * (s0['cagr'] - s['cagr']), s['turnover_per_yr']))
    OUT['cost_mult'] = {}
    for mult in [2, 3, 5]:
        c_, _ = one(cost=mult * H.T2_COST)
        OUT['cost_mult'][mult] = dict(sharpe=c_['sharpe'], cagr=c_['cagr'], dd=c_['max_dd'])
        print(' %dx cost: Sharpe %.2f CAGR %.1f%% DD %.1f%%' % (mult, c_['sharpe'], 100 * c_['cagr'], 100 * c_['max_dd']))

    print('\n=== scenarios, official offset')
    OUT['scen'] = {}
    for sc in ['warm', 'tough', 'bear']:
        x, r_ = one(scen=sc)
        OUT['scen'][sc] = dict(sharpe=x['sharpe'], cagr=x['cagr'], dd=x['max_dd'], total=x['total_return'], tim=x['time_in_market'], tr=x['trades_per_yr'])
        print(' %-6s Sharpe %.2f CAGR %.1f%% total %.1f%% DD %.1f%% in-mkt %.0f%% trades/yr %.0f' % (sc, x['sharpe'], 100 * x['cagr'], 100 * x['total_return'], 100 * x['max_dd'], 100 * x['time_in_market'], x['trades_per_yr']))

    with Pool(6, initializer=init) as pool:
        print('\n=== means over 4 rebalance offsets')
        b = block(pool, scens=('eval', 'tough', 'warm', 'bear')); OUT['offsets'] = b; print(ln('reference', b))
        print(' bear: Sh %.2f DD %.1f%% CAGR %.1f%%' % (b['bear']['sh'], 100 * b['bear']['dd'], 100 * b['bear']['cagr']))

        print('\n=== 3x3+ sensitivity: breadth_min x lookback')
        OUT['grid'] = []
        for bm in [.5, .55, .6, .65, .7]:
            for lb in [84, 126, 189]:
                bb = block(pool, dict(breadth_min=bm, lookback=lb))
                OUT['grid'].append(dict(breadth_min=bm, lookback=lb, sh=bb['eval']['sh'], sh_min=bb['eval']['sh_min'], dd=bb['eval']['dd'], cagr=bb['eval']['cagr'], tr=bb['eval']['tr'], t_sh=bb['tough']['sh'], t_dd=bb['tough']['dd'], w_ret=bb['warm']['ret'], w_dd=bb['warm']['dd']))
                print(ln('breadth %.2f lookback %d' % (bm, lb), bb))

        print('\n=== one at a time')
        OUT['oat'] = {}
        tests = {'top_k': [8, 10, 12, 15], 'sell_rank': [20, 30, 40, 60, 90], 'max_vol': [.45, .55, .65, .8, 9.9], 'breakout_days': [1, 5, 10, 20, 40]}
        for key, vals in tests.items():
            OUT['oat'][key] = []
            for val in vals:
                bb = block(pool, {key: val})
                OUT['oat'][key].append(dict(value=val, sh=bb['eval']['sh'], dd=bb['eval']['dd'], cagr=bb['eval']['cagr'], tr=bb['eval']['tr'], t_sh=bb['tough']['sh'], t_dd=bb['tough']['dd'], w_ret=bb['warm']['ret']))
                print(ln('%s = %s' % (key, val), bb))
        ctests = {'REVIEW_EVERY': [5, 10, 21], 'MIN_TRADED_VALUE': [1e8, 1.5e8, 2.5e8, 5e8, 1e9, 2e9], 'BREADTH_MA': [20, 30, 50, 75, 100], 'VOL_DAYS': [60, 120, 250], 'JUMP': [.3, .45, .6, 9.9], 'JUMP_COOLOFF': [0, 10, 40]}
        OUT['const'] = {}
        for key, vals in ctests.items():
            OUT['const'][key] = []
            for val in vals:
                bb = block(pool, None, {key: val})
                OUT['const'][key].append(dict(value=val, sh=bb['eval']['sh'], dd=bb['eval']['dd'], cagr=bb['eval']['cagr'], tr=bb['eval']['tr'], t_sh=bb['tough']['sh'], t_dd=bb['tough']['dd'], w_ret=bb['warm']['ret']))
                print(ln('%s = %s' % (key, val), bb))

        print('\n=== no regime at all (breadth_min 0): what the filter adds')
        bb = block(pool, dict(breadth_min=0.0)); OUT['no_regime'] = bb; print(ln('breadth_min 0', bb))

        print('\n=== random 70% sub-universes (20 seeds)')
        rows = pd.DataFrame(pool.map(uni_job, range(20), chunksize=1))
        print(rows.describe().loc[['mean', 'std', 'min', '50%', 'max']].round(3).to_string())
        OUT['subuni'] = dict(sh_mean=float(rows.sh.mean()), sh_min=float(rows.sh.min()), sh_max=float(rows.sh.max()), dd_mean=float(rows.dd.mean()), dd_min=float(rows.dd.min()), cagr_mean=float(rows.cagr.mean()))

        print('\n=== survivorship: inject ghost stocks that look like real names, then fall 70% (or 90%) over 60 days and vanish')
        OUT['ghost'] = []
        for n, fall in [(15, .7), (30, .7), (60, .7), (30, .9), (60, .9)]:
            g = pd.DataFrame(pool.map(ghost_job, [(n, sd, fall) for sd in range(12)], chunksize=1))
            OUT['ghost'].append(dict(n=n, fall=fall, sh=float(g.sh.mean()), cagr=float(g.cagr.mean()), dd=float(g.dd.mean()), dd_min=float(g.dd.min()), trades=float(g.ghost_trades.mean())))
            print(' %2d ghosts, fall %.0f%%: Sharpe %.2f CAGR %.1f%% DD %.1f%% (worst %.1f%%), ghost trades per run %.1f' % (n, 100 * fall, g.sh.mean(), 100 * g.cagr.mean(), 100 * g.dd.mean(), 100 * g.dd.min(), g.ghost_trades.mean()))

    json.dump(OUT, open(os.path.join(RESULTS, 't2_final.json'), 'w'), indent=1, default=float)
    print('\nminutes %.1f' % ((time.time() - t0) / 60))
