"""Out-of-sample extension for Task 1: Feb 2022 to Jun 2024 15-min bars built from the REAL daily Nifty OHLC.

Each day's 25 bars are a seeded Brownian bridge from the real open to the real close, stretched to touch the real high and
low. So overnight gaps, day ranges and close-to-close moves are real, and intraday paths are synthetic. This is valid for
judging day-level logic (trend filters, overnight holds) and meaningless for anything that reads the intraday path.
It is research only: nothing in task1.py depends on it.
"""
import numpy as np
import pandas as pd

from common import H, NIFTY_FILE, t1_data

BAR_TIMES = [f'{h:02d}:{m:02d}' for h in range(9, 16) for m in range(0, 60, 15)]
BAR_TIMES = [t for t in BAR_TIMES if '09:15' <= t <= '15:15']


def _day_path(o, h, l, c, rng):
    n = len(BAR_TIMES)
    z = np.cumsum(rng.standard_normal(n))
    z = z - np.arange(1, n + 1) / n * z[-1]                     # bridge: ends at 0
    z = np.r_[0.0, z]                                           # n+1 points, 0 at both ends
    z = z / max(z.max() - z.min(), 1e-9) * max(h - l, 1e-9)     # noise on the scale of the real day's range
    line = np.linspace(o, c, n + 1)
    up, dn = max(o, c), min(o, c)
    raw = line + z
    p = raw.copy()
    # stretch the excursions so the path's extremes equal the real high and low
    a = raw - up; b = dn - raw
    if a.max() > 0:
        p = np.where(a > 0, up + a / a.max() * (h - up), p)
    if b.max() > 0:
        p = np.where(b > 0, dn - b / b.max() * (dn - l), p)
    p = np.clip(p, l, h)
    # a day whose path never left the open-close band still has to touch its real extremes
    interior = np.arange(1, n)
    if h > up and a.max() <= 0:
        k = rng.choice(interior[b[1:n] <= 0]) if (b[1:n] <= 0).any() else rng.choice(interior)
        p[k] = h
    if l < dn and b.max() <= 0:
        free = interior[(a[1:n] <= 0) & (p[1:n] != h)]
        p[rng.choice(free if len(free) else interior)] = l
    p[0], p[-1] = o, c
    return p


def synth_bars(daily, seed=11):
    rng = np.random.default_rng(seed)
    idx, rows = [], []
    for d, x in daily.iterrows():
        o, h, l, c = float(x.open), float(x.high), float(x.low), float(x.close)
        p = _day_path(o, h, l, c, rng)
        for i, t in enumerate(BAR_TIMES):
            bo, bc = p[i], p[i + 1]
            wig = abs(rng.standard_normal()) * 0.15 * abs(bc - bo)
            bh, bl = min(max(bo, bc) + wig, h), max(min(bo, bc) - wig, l)
            idx.append(pd.Timestamp(f'{d.date()} {t}')); rows.append((bo, bh, bl, bc))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=['open', 'high', 'low', 'close'])


def monthly_expiries(dates):
    """README rule: last Thursday of the month, moved to the previous trading day on holidays."""
    days = pd.DatetimeIndex(dates)
    out = set()
    for per in sorted(set(days.to_period('M'))):
        last = per.to_timestamp(how='end').normalize()
        th = last - pd.Timedelta(days=(last.dayofweek - 3) % 7)
        while th not in days and th >= per.to_timestamp():
            th -= pd.Timedelta(days=1)
        out.add(th)
    return out


def extended(seed=11):
    """Real bars to Jan 2022 followed by synthetic bars built from real daily OHLC. Returns bars, expiries, oos start."""
    bars, ex = t1_data()
    n = pd.read_csv(NIFTY_FILE, parse_dates=['date']).set_index('date').sort_index()[['open', 'high', 'low', 'close']]
    n = n[n.index > bars.index[-1].normalize()]
    s = synth_bars(n, seed)
    all_bars = pd.concat([bars, s])
    return all_bars, set(ex) | monthly_expiries(n.index), n.index[0]


if __name__ == '__main__':
    b, ex, start = extended()
    x = b[b.index.normalize() >= start]
    print('synthetic rows', len(x), 'from', x.index[0], 'to', x.index[-1], 'expiries total', len(ex))
    n = pd.read_csv(NIFTY_FILE, parse_dates=['date']).set_index('date').sort_index()
    g = x.groupby(x.index.normalize())
    chk = pd.DataFrame({'o': g.open.first(), 'h': g.high.max(), 'l': g.low.min(), 'c': g.close.last()}).join(n)
    print('max abs diff vs real daily: open %.4f high %.4f low %.4f close %.4f' % ((chk.o - chk.open).abs().max(), (chk.h - chk.high).abs().max(), (chk.l - chk.low).abs().max(), (chk.c - chk.close).abs().max()))
    print('OHLC valid:', bool(((x.high >= x[['open', 'close']].max(axis=1) - 1e-6) & (x.low <= x[['open', 'close']].min(axis=1) + 1e-6)).all()))
    res = H.task1_backtest(b, np.ones(len(b)), ex)
    s, _ = H.task1_stats(res, start=str(start.date()))
    print('buy&hold on the OOS window: CAGR %.2f%% DD %.2f%% Sharpe %.2f' % (100 * s['cagr'], 100 * s['max_dd'], s['sharpe']), {k: round(100 * v, 1) for k, v in s['yearly'].items()})
