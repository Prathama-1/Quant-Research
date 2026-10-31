"""Task 1 baselines scored by the harness, then a dev/validation look at what predicts each leg of the day.
Dev = 2017-2019, Val = 2020-Jan 2022. Features are only ever measured before the leg they try to predict."""
import numpy as np
import pandas as pd

from common import H, Clock, hm, row, stress_table, t1_data, t1_eval, yearly

bars, ex = t1_data()
ck = Clock(bars)
c, o = bars.close.values, bars.open.values
N = len(c)
DEV_END, VAL_START = '2019-12-31', '2020-01-01'


def report(name, want):
    s, _, _ = t1_eval(want, bars, ex)
    d, _, _ = t1_eval(want, bars, ex, end=DEV_END)
    v, _, _ = t1_eval(want, bars, ex, start=VAL_START)
    r = row(s)
    print('%-34s CAGR %6.2f DD %7.2f Sh %5.2f | dev %5.2f val %5.2f | tr/yr %5.0f tim %4.0f%% | %s' % (
        name, r['cagr'], r['dd'], r['sharpe'], d['sharpe'], v['sharpe'], r['tr_yr'], r['tim'], yearly(s)))


print('--- baselines')
report('buy & hold long', np.ones(N))
report('template MA 50/200 bars long-flat', (pd.Series(c).rolling(50).mean() > pd.Series(c).rolling(200).mean()).astype(float).values)


def overnight(entry_tod, exit_tod=None):
    """Long from the fill after entry_tod's close until the fill after exit_tod's close on the next day (None = next open)."""
    want = np.zeros(N); holding = False
    for i in range(N):
        if ck.tod[i] == entry_tod:
            holding = True
        elif holding and exit_tod is None and ck.tod[i] == hm('15:15'):
            holding = False
        elif holding and exit_tod is not None and ck.tod[i] == exit_tod:
            holding = False
        want[i] = 1.0 if holding else 0.0
    return want


print('\n--- overnight-only long, by entry decision bar (exit at next open)')
for e in ['13:00', '14:00', '14:30', '14:45', '15:00']:
    report('enter after %s, exit next open' % e, overnight(hm(e)))
print('\n--- overnight long, entry after 14:30, exit later in the morning')
for x in ['09:15', '09:30', '10:00', '11:00', '12:00']:
    report('exit after %s bar' % x, overnight(hm('14:30'), hm(x)))
print('\n--- stress: overnight long, enter after 14:30, exit next open')
print(stress_table(overnight(hm('14:30')), bars, ex).to_string())
print('\n--- stress: overnight long, enter after 15:00, exit next open')
print(stress_table(overnight(hm('15:00')), bars, ex).to_string())

# ---------------------------------------------------------------------------------------- conditional means
df = pd.DataFrame({'close': c, 'open': o, 'high': bars.high.values, 'low': bars.low.values, 'tod': ck.tod, 'day': ck.day_id}, index=bars.index)
g = df.groupby('day')
d = pd.DataFrame({'date': g.apply(lambda x: x.index[0].normalize(), include_groups=False), 'o': g.open.first(), 'h': g.high.max(), 'l': g.low.min(), 'c': g.close.last()})
dec = df[df.tod == hm('14:30')].groupby('day').close.last()
d['p1430'] = dec
d['on_next'] = d.o.shift(-1) / d.c - 1                  # the gap we would be holding
d['id_next'] = d.c.shift(-1) / d.o.shift(-1) - 1        # next day's session
d['cc'] = d.c / d.c.shift() - 1
d['gap'] = d.o / d.c.shift() - 1
d['id'] = d.c / d.o - 1
d['sofar'] = d.p1430 / d.o - 1                          # today's session so far, at the decision time
d['day_so_far'] = d.p1430 / d.c.shift() - 1
for n in [3, 5, 10, 20, 50, 100, 200]:
    d['ma%d' % n] = d.p1430 / d.c.shift().rolling(n, min_periods=n // 2).mean() - 1
    d['mom%d' % n] = d.p1430 / d.c.shift(n) - 1
d['rv10'] = d.cc.shift().rolling(10).std() * 252 ** .5
d['rv20'] = d.cc.shift().rolling(20).std() * 252 ** .5
d['rv60'] = d.cc.shift().rolling(60).std() * 252 ** .5
d['rvr'] = d.rv10 / d.rv60
d['range5'] = ((d.h - d.l) / d.c).shift().rolling(5).mean()
hh = d.h.shift().rolling(20).max(); ll = d.l.shift().rolling(20).min()
d['pos20'] = (d.p1430 - ll) / (hh - ll)
d['dow'] = pd.to_datetime(d.date).dt.dayofweek
d['yr'] = pd.to_datetime(d.date).dt.year
d['dev'] = pd.to_datetime(d.date) <= pd.Timestamp(DEV_END)


def q_table(feat, target, q=5):
    x = d[[feat, target, 'dev']].dropna()
    out = []
    for name, sub in [('dev', x[x.dev]), ('val', x[~x.dev])]:
        edges = x[x.dev][feat].quantile(np.linspace(0, 1, q + 1)).to_numpy(copy=True)          # cut points fixed on dev only
        edges[0], edges[-1] = -np.inf, np.inf
        lab = pd.cut(sub[feat], edges, labels=False, duplicates='drop')
        m = sub.groupby(lab)[target].agg(['mean', 'std', 'count'])
        out.append(('%s bps ' % name) + ' '.join('%6.1f(%4.1f)' % (a * 1e4, a / s * n ** .5) for a, s, n in zip(m['mean'], m['std'], m['count'])))
    return out


feats = ['sofar', 'day_so_far', 'gap', 'ma3', 'ma5', 'ma10', 'ma20', 'ma50', 'ma100', 'ma200', 'mom5', 'mom20', 'mom50', 'rv10', 'rv20', 'rv60', 'rvr', 'range5', 'pos20']
for tgt in ['on_next', 'id_next']:
    print('\n=== quintiles of feature vs %s: mean bps (t-stat). Cut points from dev only. Q1 low .. Q5 high' % tgt)
    for f in feats:
        a, b = q_table(f, tgt)
        print('%-11s %s' % (f, a)); print('%-11s %s' % ('', b))

print('\n=== day of week')
print(d.groupby('dow')[['on_next', 'id_next']].agg(['mean', 'count']).assign(on_bps=lambda x: x[('on_next', 'mean')] * 1e4, id_bps=lambda x: x[('id_next', 'mean')] * 1e4)[['on_bps', 'id_bps']].round(2).to_string())
print(' dow = the day the overnight hold starts; id_next is the session after that gap')
