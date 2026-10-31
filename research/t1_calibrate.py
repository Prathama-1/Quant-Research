"""How far can the OOS extension be trusted? Rebuild 2017-Jan 2022 from its own daily OHLC with the same bridge, run the same
day-level strategies on those synthetic bars, and compare with the same strategies on the real 15-min bars. Any gap between
the two columns is the error the synthetic method introduces. Also prints per-year results for the leading candidates."""
import numpy as np
import pandas as pd

from common import H, Clock, t1_data, t1_eval, yearly
from synth import synth_bars
from t1_lab import ALL_BARS, ALL_EX, OOS_START, FAMILIES

bars, ex = t1_data()
ck = Clock(bars)
g = bars.groupby(bars.index.normalize())
daily = pd.DataFrame({'open': g.open.first(), 'high': g.high.max(), 'low': g.low.min(), 'close': g.close.last()})
CANDS = [('night_ma', dict(n=20)), ('night_ma', dict(n=10)), ('night_ma', dict(n=50)), ('night_ma_fast_slow', dict(fast=10, slow=50)),
         ('hold_ma', dict(n=20)), ('hold_ma', dict(n=5)), ('hold_ma', dict(n=100))]

print('=== real 15-min bars vs synthetic bars built from the same days (3 seeds), in-sample window')
print('%-44s %24s | %s' % ('config', 'real: Sh / DD / CAGR', 'synthetic seeds: Sh / DD / CAGR'))
for fam, p in CANDS:
    r, _, _ = t1_eval(FAMILIES[fam](bars, p), bars, ex)
    out = []
    for seed in [1, 2, 3]:
        sb = synth_bars(daily, seed)
        # same index as real where days are full; expiries unchanged
        s, _, _ = t1_eval(FAMILIES[fam](sb, p), sb, ex)
        out.append('%5.2f /%6.1f /%5.1f' % (s['sharpe'], 100 * s['max_dd'], 100 * s['cagr']))
    print('%-44s %5.2f /%6.1f /%5.1f      | %s' % (fam + ' ' + str(p), r['sharpe'], 100 * r['max_dd'], 100 * r['cagr'], '  '.join(out)))

print('\n=== year-wise (%), real in-sample then OOS extension (seed 11)')
for fam, p in CANDS:
    s, _, _ = t1_eval(FAMILIES[fam](bars, p), bars, ex)
    w = FAMILIES[fam](ALL_BARS, p)
    o, _, _ = t1_eval(w, ALL_BARS, ALL_EX, start=OOS_START)
    print('%-44s %s OOS %s' % (fam + ' ' + str(p), yearly(s), yearly(o)))

print('\n=== OOS extension across path seeds: does the answer depend on the synthetic path?')
from synth import extended
for fam, p in CANDS:
    vals = []
    for seed in [11, 12, 13, 14, 15]:
        b, e, st = extended(seed)
        o, _, _ = t1_eval(FAMILIES[fam](b, p), b, e, start=str(st.date()))
        vals.append((o['sharpe'], o['max_dd'], o['cagr']))
    v = np.array(vals)
    print('%-44s Sharpe %s  mean %.2f | DD mean %.1f%% | CAGR mean %.1f%%' % (fam + ' ' + str(p), np.round(v[:, 0], 2), v[:, 0].mean(), 100 * v[:, 1].mean(), 100 * v[:, 2].mean()))
