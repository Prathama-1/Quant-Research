"""The strategy's main risk is that the overnight premium fades. Scenario: remove X bps from every overnight gap on the OOS
extension (intraday returns untouched) and compare gate settings. What does a 'premium still paying?' gate buy in a bad world
and cost in a good one? The same haircut on the real in-sample bars gives the break-even."""
import numpy as np
import pandas as pd

from common import Clock, H, t1_data, t1_eval
from synth import extended
from t1_lab2 import want_of

BASE = dict(ma=20, sofar=-.0025, rvmax=(20, .35))
GATES = {'none': None, '(60,5)': (60, 5), '(120,3)': (120, 3), '(120,5)': (120, 5), '(120,7)': (120, 7), '(250,3)': (250, 3), '(250,5)': (250, 5), '(250,7)': (250, 7)}
HAIRCUTS = [0, 3, 6, 9, 12, 15]
SEEDS = [11, 12, 13]


def haircut(bars, start, x_bps):
    """Lower every gap from `start` on by x bps: scale each day's prices by (1 - x)^(days since start)."""
    if x_bps == 0: return bars
    ck = Clock(bars)
    day = bars.index.normalize()
    start = pd.Timestamp(start)
    d0 = ck.day_id[np.searchsorted(day, start)]
    n = np.maximum(ck.day_id - d0 + 1, 0)
    f = (1 - x_bps / 1e4) ** n
    out = bars.copy()
    for c in ['open', 'high', 'low', 'close']:
        out[c] = bars[c].values * f
    return out


if __name__ == '__main__':
    worlds = []
    for s in SEEDS:
        b, e, st = extended(s)
        worlds.append((b, e, str(st.date())))
    ib, ie = t1_data()

    print('=== OOS Feb22-Jun24, mean of %d path seeds: Sharpe / MaxDD %% / CAGR %%, by gap haircut (bps per night; real OOS gap mean is 9.4)' % len(SEEDS))
    print('%-10s' % 'gate' + ''.join('%22s' % ('-%d bps' % x) for x in HAIRCUTS))
    for name, g in GATES.items():
        p = dict(BASE)
        if g: p['premium'] = g
        line = '%-10s' % name
        for x in HAIRCUTS:
            out = []
            for b, e, st in worlds:
                hb = haircut(b, st, x)
                s, _, _ = t1_eval(want_of(hb, Clock(hb), p), hb, e, start=st)
                out.append((s['sharpe'], s['max_dd'], s['cagr']))
            o = np.mean(out, axis=0)
            line += '%9.2f /%5.1f /%5.1f' % (o[0], 100 * o[1], 100 * o[2])
        print(line)

    print('\n=== real in-sample, Apr17-Jan22, same grid (real gap mean 15.6). Decays over the whole window.')
    print('%-10s' % 'gate' + ''.join('%22s' % ('-%d bps' % x) for x in HAIRCUTS))
    first_day = str(ib.index[0].date())
    for name, g in GATES.items():
        p = dict(BASE)
        if g: p['premium'] = g
        line = '%-10s' % name
        for x in HAIRCUTS:
            hb = haircut(ib, first_day, x)
            s, _, _ = t1_eval(want_of(hb, Clock(hb), p), hb, ie)
            line += '%9.2f /%5.1f /%5.1f' % (s['sharpe'], 100 * s['max_dd'], 100 * s['cagr'])
        print(line)
