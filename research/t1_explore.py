"""Task 1 exploration: where does Nifty's return actually come from? Overnight vs intraday, by year, by time of day,
and an out-of-sample check of the overnight effect on the daily Nifty file (runs to Jun 2024, past the 15-min file)."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
pd.set_option('display.width', 220, 'display.max_columns', 60, 'display.max_rows', 300)

b = pd.read_csv(os.path.join(DATA, 'nifty50_15min_train_2017-04-03_2022-01-31.csv'), parse_dates=['datetime']).set_index('datetime')
b['date'] = b.index.normalize(); b['tod'] = b.index.strftime('%H:%M')
d = b.groupby('date').agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'), close=('close', 'last'))
d['on'] = d.open / d.close.shift() - 1
d['id'] = d.close / d.open - 1
d['cc'] = d.close / d.close.shift() - 1


def ann(x):
    x = x.dropna()
    return pd.Series({'n': len(x), 'mean_bps': x.mean() * 1e4, 'median_bps': x.median() * 1e4, 'ann_ret_%': x.mean() * 252 * 100,
                      'ann_vol_%': x.std() * 252 ** .5 * 100, 'sharpe': x.mean() / x.std() * 252 ** .5, 't': x.mean() / x.std() * len(x) ** .5,
                      'hit': (x > 0).mean()})


print('=== 15-min file: overnight vs intraday vs close-to-close')
print(pd.DataFrame({k: ann(d[k]) for k in ['on', 'id', 'cc']}).T.round(3).to_string())
print('\n=== by year (ann. % simple mean x252)')
y = d.groupby(d.index.year)
print(pd.DataFrame({'on_ann%': y.on.mean() * 25200, 'on_med_bps': y.on.median() * 1e4, 'id_ann%': y.id.mean() * 25200, 'cc_ann%': y.cc.mean() * 25200,
                    'on_sh': y.on.mean() / y.on.std() * 252 ** .5, 'id_sh': y.id.mean() / y.id.std() * 252 ** .5, 'n': y.size()}).round(2).to_string())

print('\n=== by time of day (mean bps of close/open-1 in that bar, t-stat; excl. 3-bar sessions)')
full = b.groupby('date').tod.transform('size') == 25
x = b[full].copy(); x['r'] = x.close / x.open - 1
t = x.groupby('tod').r.agg(['mean', 'std', 'count'])
t['mean_bps'] = t['mean'] * 1e4; t['t'] = t['mean'] / t['std'] * t['count'] ** .5
for yr in sorted(set(x.index.year)):
    t[str(yr)] = x[x.index.year == yr].groupby('tod').r.mean() * 1e4
print(t.drop(columns=['mean', 'std', 'count']).round(2).to_string())

print('\n=== cross-check: daily Nifty file vs 15-min file on overlapping dates')
n = pd.read_csv(os.path.join(DATA, 'nifty50_daily_2021-04-26_2024-06-30.csv'), parse_dates=['date']).set_index('date').sort_index()
n['on'] = n.open / n.close.shift() - 1; n['id'] = n.close / n.open - 1; n['cc'] = n.close / n.close.shift() - 1
ov = d.join(n[['open', 'high', 'low', 'close', 'on']], rsuffix='_d', how='inner')
print('overlap days', len(ov))
for c in ['open', 'high', 'low', 'close']:
    dif = (ov[c + '_d'] / ov[c] - 1) * 1e4
    print('%-5s daily/15m diff bps: mean %.2f  std %.2f  max|.| %.2f  exact-match %.1f%%' % (c, dif.mean(), dif.std(), dif.abs().max(), 100 * (dif.abs() < 0.01).mean()))
print('overnight overlap, daily-file vs 15m-file: mean %.2f vs %.2f bps, median %.2f vs %.2f bps' % (ov.on_d.mean() * 1e4, ov.on.mean() * 1e4, ov.on_d.median() * 1e4, ov.on.median() * 1e4))
print('\n=== daily file: overnight vs intraday; OOS = dates after the 15-min file ends')
oos = n[n.index > d.index[-1]]
print('OOS days', len(oos), oos.index[0].date(), oos.index[-1].date())
print(pd.DataFrame({k: ann(oos[k]) for k in ['on', 'id', 'cc']}).T.round(3).to_string())
y = n.groupby(n.index.year)
print(pd.DataFrame({'on_ann%': y.on.mean() * 25200, 'on_med_bps': y.on.median() * 1e4, 'id_ann%': y.id.mean() * 25200, 'cc_ann%': y.cc.mean() * 25200,
                    'on_sh': y.on.mean() / y.on.std() * 252 ** .5, 'id_sh': y.id.mean() / y.id.std() * 252 ** .5, 'n': y.size()}).round(2).to_string())

print('\n=== serial dependence at daily scale (15-min file), corr of today vs next components')
d['on_next'] = d.on.shift(-1); d['id_next'] = d.id.shift(-1); d['cc_next'] = d.cc.shift(-1)
print(d[['on', 'id', 'cc', 'on_next', 'id_next', 'cc_next']].corr().loc[['on', 'id', 'cc'], ['on_next', 'id_next', 'cc_next']].round(3).to_string())
print('corr today gap vs today intraday: %.3f' % d.on.corr(d.id))
