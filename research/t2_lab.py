"""Task 2 laboratory. Runs on the official task2_backtest. Features are precomputed once with causal rolling operations and
read by date for speed; the final task2.py recomputes them from `hist` and is checked against this.

Each config is judged on:
  eval   official window, 2 Mar 2022 to 30 Jun 2024
  warm   Oct 2021 to Feb 2022, a weak stretch before the window and not used for tuning
  tough  drift-stripped data: every stock loses TOUGH_G per year so the equal-weight universe earns about 10% a year rather than
         about 47%, and Nifty about 5%. This removes most of the survivorship and bull-market tailwind. Selection and risk logic
         have to earn the return.
"""
import os
import sys
import types
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import H, RESULTS, t2_data

EVAL = ('2022-03-01', '2024-06-30')
WARM = ('2021-10-15', '2022-02-28')
TOUGH_G, TOUGH_N = 0.29, 0.108                     # log drift removed per year from stocks and from Nifty
LOG = os.path.join(RESULTS, 't2_lab.csv')
_cache = {}


def strip_drift(D, g_s, g_n):
    t0 = np.searchsorted(D['close'].index, pd.Timestamp(EVAL[0]))
    yrs = (np.arange(len(D['close'])) - t0) / 252
    f_s = np.exp(-g_s * np.maximum(yrs, 0))[:, None]
    f_n = np.exp(-g_n * np.maximum(yrs, 0))[:, None]
    out = {}
    for k in ['open', 'high', 'low', 'close']:
        out[k] = pd.DataFrame(D[k].values * f_s, index=D[k].index, columns=D[k].columns)
    out['volume'] = D['volume']
    out['nifty'] = pd.DataFrame(D['nifty'][['open', 'high', 'low', 'close']].values * f_n, index=D['nifty'].index, columns=['open', 'high', 'low', 'close'])
    return out


def features(D):
    C, V = D['close'], D['volume']
    ret = C.pct_change(fill_method=None)
    clean = ret.where((ret.abs() < .45) | ret.isna(), 0.0)
    P = (1 + clean.fillna(0)).cumprod().where(C.notna())
    TV = C * V
    N = D['nifty']['close']
    mret = N.pct_change()
    F = {}
    F['liq'] = TV.rolling(60, min_periods=40).median()
    F['age'] = C.notna().cumsum()
    for n in [21, 42, 63, 126, 189, 252]:
        F['mom%d' % n] = P / P.shift(n) - 1
    for n in [126, 252]:
        F['mom%d_21' % n] = P.shift(21) / P.shift(n) - 1
    v60 = clean.rolling(60, min_periods=40).std()
    F['vol60'] = v60
    F['vol120'] = clean.rolling(120, min_periods=60).std()
    F['sh126'] = clean.rolling(126, min_periods=80).mean() / clean.rolling(126, min_periods=80).std()
    F['sh252'] = clean.rolling(252, min_periods=120).mean() / clean.rolling(252, min_periods=120).std()
    F['hi252'] = P / P.rolling(252, min_periods=120).max() - 1
    mv = mret.rolling(120, min_periods=60).var()
    beta = pd.DataFrame({s: clean[s].rolling(120, min_periods=60).cov(mret) for s in clean.columns}).div(mv, axis=0)
    F['beta'] = beta
    res = clean.sub(beta.shift(1).mul(mret, axis=0))
    F['res126'] = res.rolling(126, min_periods=80).sum() / res.rolling(126, min_periods=80).std()
    F['res252'] = res.rolling(252, min_periods=120).sum() / res.rolling(252, min_periods=120).std()
    F['tvshock'] = np.log(TV.rolling(20, min_periods=10).mean() / TV.rolling(120, min_periods=60).mean())
    F['ma50gap'] = P / P.rolling(50, min_periods=30).mean() - 1
    F['ma200gap'] = P / P.rolling(200, min_periods=100).mean() - 1
    F['ma20gap'] = P / P.rolling(20, min_periods=15).mean() - 1
    F['ma100gap'] = P / P.rolling(100, min_periods=60).mean() - 1
    F['rank_mom'] = (F['mom63'].rank(axis=1, pct=True) + F['mom126'].rank(axis=1, pct=True) + F['mom252'].rank(axis=1, pct=True).fillna(.5)) / 3
    F['rank_res'] = F['res126'].rank(axis=1, pct=True)
    F['rank_tv'] = F['tvshock'].rank(axis=1, pct=True)
    F['mix_mr'] = F['rank_mom'] + F['rank_res']
    F['mix_mrt'] = F['rank_mom'] + F['rank_res'] + .5 * F['rank_tv']
    F['mix_rt'] = F['rank_res'] + .5 * F['rank_tv']
    F['mix_sh'] = F['sh126'].rank(axis=1, pct=True) + F['rank_res']
    F['lowvol'] = -F['vol120']
    F['mom_over_vol'] = F['mom126'] / (F['vol120'] * np.sqrt(126))
    F['mom189_21'] = P.shift(21) / P.shift(189) - 1
    F['mom63_21'] = P.shift(21) / P.shift(63) - 1
    F['skip_blend'] = sum(F[k].rank(axis=1, pct=True).fillna(.5) for k in ['mom126_21', 'mom189_21', 'mom252_21']).where(F['mom126_21'].notna())
    F['skip_res'] = F['skip_blend'] / 3 + F['rank_res']
    F['skip_res_tv'] = F['skip_blend'] / 3 + F['rank_res'] + .5 * F['rank_tv']
    F['res_vs_vol'] = F['rank_res'] + (F['mom_over_vol'].rank(axis=1, pct=True))
    F['hi20'] = (D['high'].rolling(20, min_periods=10).max())
    F['hi10'] = (D['high'].rolling(10, min_periods=5).max())
    F['hi5'] = (D['high'].rolling(5, min_periods=3).max())
    F['mom126_pos'] = F['mom126']
    F['mv_tv'] = F['mom_over_vol'].rank(axis=1, pct=True) + .5 * F['rank_tv']
    F['sh_tv'] = F['sh126'].rank(axis=1, pct=True) + .5 * F['rank_tv']
    F['mom63_over_vol'] = F['mom63'] / (F['vol60'] * np.sqrt(63))
    F['msh_blend'] = F['mom63_over_vol'].rank(axis=1, pct=True) + F['mom_over_vol'].rank(axis=1, pct=True)
    F['liq_rank'] = F['liq'].rank(axis=1, ascending=False)
    B = {}
    above50 = (P > P.rolling(50, min_periods=30).mean()).where(C.notna())
    above200 = (P > P.rolling(200, min_periods=100).mean()).where(C.notna())
    liqm = F['liq'] >= 1e8
    B['breadth50'] = above50.where(liqm).mean(axis=1)
    B['breadth200'] = above200.where(liqm).mean(axis=1)
    for n in [20, 50, 100, 200]:
        B['nifty_ma%d' % n] = N / N.rolling(n, min_periods=n // 2).mean() - 1
    ew = clean.where(liqm.shift(1)).mean(axis=1)
    ew_idx = (1 + ew.fillna(0)).cumprod()
    for n in [50, 100, 200]:
        B['ew_ma%d' % n] = ew_idx / ew_idx.rolling(n, min_periods=n // 2).mean() - 1
    return {k: v.values for k, v in F.items()}, {k: v.values for k, v in B.items()}, list(C.columns)


def world():
    if not _cache:
        D = t2_data()
        _cache['D'] = D
        _cache['X'] = features(D)
        DT = strip_drift(D, TOUGH_G, TOUGH_N)
        _cache['DT'] = DT
        _cache['XT'] = features(DT)
    return _cache


def regime_on(B, t, p):
    r = p.get('regime')
    if not r: return True
    ok = True
    for name, th in r.items():
        ok = ok and (B[name][t] >= th or np.isnan(B[name][t]))
    return ok


def make_orders(X, universe=None, rng_seed=None):
    F, B, cols = X
    idx = {s: j for j, s in enumerate(cols)}
    mask = None if universe is None else np.array([s in universe for s in cols])
    rng = None if rng_seed is None else np.random.default_rng(rng_seed)

    def orders(date, hist, portfolio, params, state):
        p = params
        t = len(hist['close']) - 1
        held = portfolio['positions']
        out = []
        peak = state.setdefault('peak', {})
        last = hist['close'].iloc[-1].values
        for s in list(peak):
            if s not in held: del peak[s]
        for s in held:
            px = last[idx[s]]
            if not np.isnan(px): peak[s] = max(peak.get(s, px), px)

        on = regime_on(B, t, p)
        if p.get('band'):                                     # hysteresis on breadth: off below lo, back on above hi
            lo, hi = p['band']
            b_ = B[p.get('band_on', 'breadth50')][t]
            st_on = state.get('risk_on', False)
            if b_ >= hi: st_on = True
            elif b_ < lo: st_on = False
            state['risk_on'] = st_on
            on = on and st_on
        k = p['K']
        offset = p.get('offset', 0)
        rebal = (t - state.setdefault('t0', t) + offset) % p['every'] == 0 if p['every'] > 0 else (len(hist['close']) >= 2 and hist['close'].index[-1].month != hist['close'].index[-2].month)
        selling = set()

        if not on and p.get('off') == 'sell':
            selling |= set(held)
        if not on and p.get('off') == 'weak':                 # risk-off: drop only what has lost its own trend
            g = F[p.get('weak_ma', 'ma50gap')][t]
            for s in held:
                if not g[idx[s]] > 0: selling.add(s)
        ma_exit = p.get('ma_exit')
        if ma_exit:
            g = F[ma_exit][t]
            for s in held:
                if g[idx[s]] < 0: selling.add(s)
        fill = p.get('fill_daily') and on and (len(held) - len(selling) < k)
        if rebal or fill:
            if p['score'] == 'random':
                s_ = rng.random(len(cols))
            elif p['score'] == 'random_fixed':                # one fixed random ranking per seed: selection-free, no forced turnover
                s_ = state.setdefault('fixed_rank', rng.random(len(cols)))
            else:
                s_ = F[p['score']][t]
            good = (F['liq'][t] >= p['liq']) & (last >= p.get('minpx', 0)) & (F['age'][t] >= p.get('min_age', 126)) & ~np.isnan(s_)
            if mask is not None: good = good & mask
            for tr in ([p['trend']] if isinstance(p.get('trend'), str) else (p.get('trend') or [])):
                good = good & (F[tr][t] > 0)
            if p.get('maxvol'):
                good = good & (F['vol120'][t] * np.sqrt(252) < p['maxvol'])
            if p.get('maxbeta'):
                good = good & (F['beta'][t] < p['maxbeta'])
            order = np.argsort(-np.where(good, s_, -np.inf))
            ranked = [cols[j] for j in order[:int(good.sum())]]
            rank = {s: i for i, s in enumerate(ranked)}
            buf = p.get('buffer', k)
            if rebal:
                for s in held:
                    if rank.get(s, 10 ** 9) >= buf: selling.add(s)
            n_after = len(held) - len(selling)
            if on:
                for s in ranked:
                    if n_after >= k: break
                    if s in held: continue
                    w = p.get('weight', 1.0 / k)
                    if p.get('ivw'):                   # inverse-vol tilt around 1/K, capped
                        v = F['vol120'][t]
                        w = float(np.clip(w * np.nanmedian(v[good]) / v[idx[s]], w * .5, min(w * p['ivw'], .15)))
                    if p.get('breakout'):              # buy only if it trades above its recent high tomorrow
                        out.append({'symbol': s, 'side': 'buy', 'type': 'stop', 'trigger': float(F[p['breakout']][t][idx[s]]), 'weight': w, 'max_positions': k})
                    else:
                        out.append({'symbol': s, 'side': 'buy', 'type': 'market', 'weight': w})
                    n_after += 1
                    if p.get('breakout') and n_after >= k + p.get('extra', 0): break
        out = [{'symbol': s, 'side': 'sell', 'type': 'market'} for s in selling] + out
        stop = p.get('stop')
        if stop:
            for s in held:
                if s not in selling and s in peak:
                    out.append({'symbol': s, 'side': 'sell', 'type': 'stop', 'trigger': peak[s] * (1 - stop)})
        return out
    return orders


def run(p, which='eval', cost=H.T2_COST, universe=None, seed=None):
    w = world()
    D, X = (w['DT'], w['XT']) if which == 'tough' else (w['D'], w['X'])
    a, z = WARM if which == 'warm' else EVAL
    m = types.SimpleNamespace(orders=make_orders(X, universe, seed))
    res = H.task2_backtest(m, D, p, a, z, cost=cost)
    s, e = H.task2_stats(res)
    s['trades_per_yr'] = s['trades'] / max(s['years'], 1e-9)
    return s, res


def score(p):
    e, res = run(p)
    w_, _ = run(p, 'warm')
    t_, _ = run(p, 'tough')
    c2, _ = run(p, cost=2 * H.T2_COST)
    T = res['trades']
    top = np.nan
    if len(T):
        # share of the summed trade P&L from the 10 best trades, sized by entry weight 1/K
        srt = T.ret.sort_values(ascending=False)
        top = srt.iloc[:10].sum() / T.ret.sum() if T.ret.sum() > 0 else np.nan
    return dict(p=str(p), cagr=e['cagr'], dd=e['max_dd'], sh=e['sharpe'], tr=e['trades_per_yr'], tim=e['time_in_market'], hold=e['avg_hold_days'], win=e['win_rate'], top10=top,
                y22=e['yearly'].get('2022', np.nan), y23=e['yearly'].get('2023', np.nan), y24=e['yearly'].get('2024', np.nan),
                c2_sh=c2['sharpe'], warm_ret=w_['total_return'], warm_dd=w_['max_dd'], warm_sh=w_['sharpe'], t_cagr=t_['cagr'], t_dd=t_['max_dd'], t_sh=t_['sharpe'])


def fmt(r):
    return '%-74s CAGR %5.1f DD %6.1f Sh %5.2f | 22 %5.1f 23 %5.1f 24 %5.1f | tr %4.0f hold %3.0f tim %3.0f%% top10 %3.0f%% | 2x %5.2f | warm %6.1f dd %5.1f | tough Sh %5.2f DD %6.1f CAGR %5.1f' % (
        r['p'][:74], 100 * r['cagr'], 100 * r['dd'], r['sh'], 100 * r['y22'], 100 * r['y23'], 100 * r['y24'], r['tr'], r['hold'], 100 * r['tim'], 100 * r['top10'], r['c2_sh'], 100 * r['warm_ret'], 100 * r['warm_dd'], r['t_sh'], 100 * r['t_dd'], 100 * r['t_cagr'])


def sweep(name, configs, pool):
    print('\n=== ' + name)
    rows = pool.map(score, configs, chunksize=1)
    for r in rows: print(fmt(r))
    sys.stdout.flush()
    pd.DataFrame(rows).assign(group=name).to_csv(LOG, mode='a', header=not os.path.exists(LOG), index=False)
    return rows


BASE = dict(score='mom126', K=15, every=0, liq=1e8, min_age=126)
