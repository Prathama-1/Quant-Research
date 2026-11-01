"""Side test, not part of the submission: pair trading on NSE stocks, walk-forward. Pairs are chosen using past data only.

Methods:
  distance   Gatev, Goetzmann and Rouwenhorst (2006): smallest squared distance between normalised log prices.
  coint      Engle-Granger: OLS hedge ratio, ADF-style t-stat on the residual, with a half-life filter.
Mechanics: a new book every 21 days, formed on the prior 126 days and traded for the next 126, so about 6 books overlap. Signals at the
close, fills at the next open. Each pair slot is long one leg and short the other, 100% gross on its own capital and no leverage.
Long-short cannot run in the Task 2 harness (no shorting), so this uses its own accounting, which assumes shorts are possible (in India
that means stock futures, so the floor of Rs 100 cr is the rough proxy for names with futures). The long-only version does run in
the official harness.
"""
import itertools
import os
import sys
import types

import numpy as np
import pandas as pd

from common import H, RESULTS, t2_data

D = t2_data()
C, O, V = D['close'], D['open'], D['volume']
dates, cols = C.index, list(C.columns)
ret = C.pct_change(fill_method=None)
clean = ret.where((ret.abs() < .45) | ret.isna(), 0.0)
LP = np.log1p(clean.fillna(0)).cumsum().where(C.notna()).values          # log price level free of jump artefacts
cv, ov = C.values, O.values
r_cc = clean.fillna(0).values
with np.errstate(invalid='ignore', divide='ignore'):
    r_oc = np.nan_to_num(cv / ov - 1)                                    # entry day: open to close
    r_co = np.nan_to_num(ov / np.r_[cv[:1], cv[:-1]] - 1)                # exit day: previous close to open
r_co = np.where(np.abs(r_co) < .45, r_co, 0.0); r_oc = np.where(np.abs(r_oc) < .45, r_oc, 0.0)
LIQ = (C * V).rolling(60, min_periods=40).median().values
NIFTY = D['nifty']['close'].pct_change().values
EV0, EV1 = int(np.searchsorted(dates, pd.Timestamp('2022-03-01'))), len(dates)
FORM, TRADE, EVERY = 126, 126, 21


def select(tf, method, n, floor):
    """Pairs chosen with data up to and including tf."""
    w = LP[tf - FORM + 1:tf + 1]
    ok = (~np.isnan(w).any(axis=0)) & (LIQ[tf] >= floor)
    idx = np.flatnonzero(ok)
    X = w[:, idx] - w[0, idx]
    if len(idx) < 20: return []
    if method == 'distance':
        G = X.T @ X
        d = np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G
        np.fill_diagonal(d, np.inf)
        iu = np.triu_indices(len(idx), 1)
        order = np.argsort(d[iu])
        cand = ((iu[0][j], iu[1][j], 1.0) for j in order[:20000])
    else:
        dX = np.diff(X, axis=0)
        cc = np.corrcoef(dX.T)
        iu = np.triu_indices(len(idx), 1)
        order = np.argsort(-cc[iu])[:3000]
        res = []
        for j in order:
            a, b = iu[0][j], iu[1][j]
            x, y = X[:, b], X[:, a]
            beta = (x @ y) / max(x @ x, 1e-12)
            if not .5 < beta < 2: continue
            e = y - beta * x
            de, le = np.diff(e), e[:-1] - e[:-1].mean()
            rho = (le @ de) / max(le @ le, 1e-12)
            se = np.sqrt(((de - rho * le) ** 2).sum() / (len(de) - 1) / max(le @ le, 1e-12))
            t = rho / max(se, 1e-12)
            hl = -np.log(2) / rho if rho < 0 else np.inf
            if t < -3.4 and 3 <= hl <= 30:              # Engle-Granger 5% is about -3.34 for two variables
                res.append((t, a, b, beta))
        res.sort()
        cand = ((a, b, beta) for _, a, b, beta in res)
    out, seen = [], {}
    for a, b, beta in cand:
        if seen.get(a, 0) >= 2 or seen.get(b, 0) >= 2: continue
        s = w[:, idx[a]] - beta * w[:, idx[b]]
        sd = s.std(ddof=1)
        if sd < .03: continue                           # a spread too quiet to pay for its costs (Do and Faff)
        seen[a] = seen.get(a, 0) + 1; seen[b] = seen.get(b, 0) + 1
        out.append((int(idx[a]), int(idx[b]), float(beta), float(s.mean()), float(sd)))
        if len(out) >= n: break
    return out


_sel = {}


def book_pairs(tf, method, n, floor):
    k = (tf, method, n, floor)
    if k not in _sel: _sel[k] = select(tf, method, n, floor)
    return _sel[k]


def run(method='distance', n=20, floor=1e8, entry=2.0, exit_=.5, stop=4.0, cost=.002, offset=0, zmode='static', collect=False):
    T = len(dates)
    day = np.zeros(T); gross = np.zeros(T); trades = 0
    starts = list(range(FORM + offset, T - 1, EVERY))
    n_books = TRADE / EVERY
    longs = [dict() for _ in range(T)] if collect else None
    for tf in starts:
        ps = book_pairs(tf, method, n, floor)
        if not ps: continue
        end = min(tf + TRADE, T - 1)
        slot = 1.0 / (n_books * n)                      # unused slots stay in cash
        for a, b, beta, mu, sd in ps:
            wa, wb = 1 / (1 + beta), beta / (1 + beta)
            sprd = LP[:, a] - beta * LP[:, b]
            if zmode == 'rolling':
                ss = pd.Series(sprd)
                zs = ((ss - ss.rolling(20, min_periods=15).mean()) / ss.rolling(20, min_periods=15).std()).values
            else:
                zs = (sprd - mu) / sd
            pos = 0; blocked = False
            for t in range(tf, end):
                z = zs[t]
                new = pos
                if not np.isnan(z):
                    if pos == 0:
                        if abs(z) > entry and abs(z) < stop and not blocked:
                            new = -1 if z > 0 else 1    # spread rich: short a, long b
                    elif (pos > 0 and z > -exit_) or (pos < 0 and z < exit_):
                        new = 0
                    elif abs(z) > stop:
                        new = 0; blocked = True         # the relationship has broken: do not re-enter in this book
                if t + 1 >= end: new = 0
                d = t + 1
                if new == pos:
                    day[d] += slot * pos * (wa * r_cc[d, a] - wb * r_cc[d, b])
                else:
                    if pos != 0: day[d] += slot * pos * (wa * r_co[d, a] - wb * r_co[d, b])
                    if new != 0: day[d] += slot * new * (wa * r_oc[d, a] - wb * r_oc[d, b])
                    day[d] -= slot * cost * abs(new - pos)
                    trades += (new != 0)
                gross[d] += slot * abs(new)
                pos = new
                if collect and pos != 0:
                    leg = a if pos > 0 else b
                    longs[t][cols[leg]] = max(longs[t].get(cols[leg], 0), abs(z))
    r = pd.Series(day[EV0:], index=dates[EV0:])
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    out = dict(sharpe=float(r.mean() / r.std() * 252 ** .5) if r.std() > 0 else 0., cagr=float(eq.iloc[-1] ** (1 / yrs) - 1), dd=float((eq / eq.cummax() - 1).min()), gross=float(gross[EV0:].mean()),
               tr_yr=trades * (len(r) / (T - FORM)) / yrs, corr=float(np.corrcoef(r.values, np.nan_to_num(NIFTY[EV0:]))[0, 1]),
               y={str(k): float(v) for k, v in r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1).items()})
    return (out, longs) if collect else out


def mean_off(**kw):
    rs = [run(offset=o, **kw) for o in [0, 7, 14]]
    return {k: float(np.mean([x[k] for x in rs])) for k in ['sharpe', 'cagr', 'dd', 'gross', 'tr_yr', 'corr']}, [x['sharpe'] for x in rs]


def show(tag, m, sh):
    print('%-50s Sharpe %5.2f (%5.2f to %5.2f)  DD %6.1f%%  CAGR %5.1f%%  avg gross %3.0f%%  trades/yr %4.0f  corr Nifty %5.2f' % (tag, m['sharpe'], min(sh), max(sh), 100 * m['dd'], 100 * m['cagr'], 100 * m['gross'], m['tr_yr'], m['corr']))


def longonly_orders(longs, regime=None):
    idx = {s: j for j, s in enumerate(cols)}

    def orders(date, hist, portfolio, params, state):
        t = len(hist['close']) - 1
        want = longs[t]
        held = portfolio['positions']
        on = True if regime is None else regime[t] >= params['breadth']
        out = [{'symbol': s, 'side': 'sell', 'type': 'market'} for s in held if s not in want or not on]
        free = 15 - sum(1 for s in held if s in want and on)
        if on:
            for s, _ in sorted(want.items(), key=lambda kv: -kv[1]):
                if free <= 0: break
                if s not in held:
                    out.append({'symbol': s, 'side': 'buy', 'type': 'market', 'weight': 1 / 15}); free -= 1
        return out
    return orders


if __name__ == '__main__':
    print('Eval window 2 Mar 2022 to 28 Jun 2024. Long-short, 100% gross per slot, unlevered, idle cash earns 0. Means over 3 book-start offsets.\n')
    print('=== 1. Base configs: cost 0.2% per side per leg, same as Task 2')
    for me in ['distance', 'coint']:
        for fl, nm in [(1e8, 'Rs 10 cr floor'), (1e9, 'Rs 100 cr floor')]:
            show('%s, %s' % (me, nm), *mean_off(method=me, floor=fl))
    print('\n=== 2. Cost per side (futures are cheaper than the 0.2% cash assumption)')
    for me in ['distance', 'coint']:
        for c_ in [0, .0005, .001, .002]:
            show('%s, Rs 100 cr floor, cost %.2f%%' % (me, 100 * c_), *mean_off(method=me, floor=1e9, cost=c_))
    print('\n=== 3. Full grid, Rs 100 cr floor, cost 0.05%: read the spread, not the best row')
    rows = []
    for me, n, en, zm in itertools.product(['distance', 'coint'], [10, 20, 40], [1.5, 2.0, 2.5], ['static', 'rolling']):
        m, sh = mean_off(method=me, floor=1e9, cost=.0005, n=n, entry=en, zmode=zm)
        rows.append(dict(method=me, n=n, entry=en, z=zm, **{k: round(v, 3) for k, v in m.items()}))
    g = pd.DataFrame(rows); g.to_csv(os.path.join(RESULTS, 'alt_pairs_grid.csv'), index=False)
    print(g.sort_values('sharpe', ascending=False).head(8).to_string(index=False))
    print('... worst:'); print(g.sort_values('sharpe').head(4).to_string(index=False))
    print('configs %d | Sharpe median %.2f, q25 %.2f, q75 %.2f, share above 0: %.0f%%, share above 1: %.0f%%' % (len(g), g.sharpe.median(), g.sharpe.quantile(.25), g.sharpe.quantile(.75), 100 * (g.sharpe > 0).mean(), 100 * (g.sharpe > 1).mean()))
    for dim in ['method', 'n', 'entry', 'z']:
        print(g.groupby(dim)[['sharpe', 'cagr', 'dd', 'gross', 'tr_yr']].mean().round(3).to_string())
    print('\n=== 4. Year by year, best-median style config: rolling z, n 20, entry 2, Rs 100 cr, cost 0.05%')
    for me in ['distance', 'coint']:
        print(me, {k: round(100 * v, 1) for k, v in run(method=me, floor=1e9, cost=.0005, zmode='rolling').items() if k == 'cagr'}, {k: round(100 * v, 1) for k, v in run(method=me, floor=1e9, cost=.0005, zmode='rolling')['y'].items()})

    print('\n=== 5. Official Task 2 harness: long-only version, buy only the cheap leg, 15 positions, 0.2% per side')
    from t2_lab import features
    F, B, _ = features(D)
    for tag, kw in [('distance, Rs 10 cr, static z', dict(method='distance', floor=1e8)), ('coint, Rs 10 cr, static z', dict(method='coint', floor=1e8)), ('distance, Rs 10 cr, rolling z', dict(method='distance', floor=1e8, zmode='rolling'))]:
        _, longs = run(collect=True, n=40, **kw)
        for regime, nm in [(None, 'no filter'), (B['breadth50'], 'breadth >= 60%')]:
            m = types.SimpleNamespace(orders=longonly_orders(longs, regime))
            res = H.task2_backtest(m, D, dict(breadth=.6), H.T2_START, H.T2_END)
            s, _ = H.task2_stats(res)
            print('%-34s %-16s Sharpe %5.2f DD %6.1f%% CAGR %5.1f%% trades/yr %4.0f in-mkt %3.0f%%' % (tag, nm, s['sharpe'], 100 * s['max_dd'], 100 * s['cagr'], s['trades'] / s['years'], 100 * s['time_in_market']))
    print('\nFor comparison: submitted Task 2 Sharpe 3.05, DD -11.2%, CAGR 48.0%. 15 random stocks, monthly: Sharpe 1.66, DD -21.0%, CAGR 33.9%.')
