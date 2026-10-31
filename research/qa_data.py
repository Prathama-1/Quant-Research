"""Data QA: checks the pack's files against what README.md claims and looks for quirks that can break a strategy."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
pd.set_option('display.width', 200, 'display.max_columns', 40, 'display.max_rows', 200)


def section(t):
    print('\n' + '=' * 8, t)


# ------------------------------------------------------------------------------------ task 1 bars
section('TASK 1 BARS')
b = pd.read_csv(os.path.join(DATA, 'nifty50_15min_train_2017-04-03_2022-01-31.csv'), parse_dates=['datetime'])
print('rows', len(b), 'range', b.datetime.min(), b.datetime.max(), 'dupes', b.datetime.duplicated().sum())
print('NaN per col', b.isna().sum().to_dict())
print('non-monotonic', (b.datetime.diff().dropna() <= pd.Timedelta(0)).sum())
bad = b[(b.high < b[['open', 'close']].max(axis=1)) | (b.low > b[['open', 'close']].min(axis=1)) | (b.high < b.low)]
print('OHLC inconsistent rows', len(bad)); print(bad.head(10))
print('n_1min distribution:\n', b.n_1min.value_counts().sort_index().to_string())
b['date'] = b.datetime.dt.normalize(); b['tod'] = b.datetime.dt.strftime('%H:%M')
print('bars per time-of-day:\n', b.tod.value_counts().sort_index().to_string())
n = b.groupby('date').size()
print('trading days', len(n), 'bars/day dist:\n', n.value_counts().sort_index().to_string())
print('short days:\n', n[n < 25].to_string())
for d in n[n < 25].index[:30]:
    x = b[b.date == d]
    print(d.date(), d.day_name(), 'bars', len(x), x.tod.iloc[0], '->', x.tod.iloc[-1], 'n1min_sum', x.n_1min.sum())
print('weekend sessions:', [str(d.date()) + ' ' + d.day_name() for d in n.index if d.dayofweek >= 5])
gaps = b.datetime.diff()
same = b.date == b.date.shift()
print('intraday gaps > 15m:\n', b[same & (gaps > pd.Timedelta(minutes=15))][['datetime', 'n_1min']].assign(gap=gaps[same & (gaps > pd.Timedelta(minutes=15))]).head(30).to_string())
big = b.date.diff().dt.days
print('calendar gaps > 5 days:\n', b[big > 5][['datetime']].assign(gap_days=big[big > 5]).to_string())
r = b.close.pct_change()
print('|bar ret| top:\n', r.abs().sort_values(ascending=False).head(12).to_frame('abs_ret').join(b[['datetime', 'open', 'high', 'low', 'close', 'n_1min']]).to_string())
oc = (b.open / b.close.shift() - 1)
print('within-day open vs prev close (bps):\n', (oc[same] * 1e4).describe().round(3).to_string())
print('largest within-day |open/prevclose-1|:\n', oc[same].abs().sort_values(ascending=False).head(8).to_frame('abs').join(b[['datetime', 'open', 'n_1min']]).to_string())
print('overnight gap (bps):\n', (oc[~same] * 1e4).describe().round(2).to_string())
print('flat bars (H==L):', (b.high == b.low).sum())
ex = pd.read_csv(os.path.join(DATA, 'nifty_monthly_expiries_2017-01_2022-01.csv'))
print('expiries', len(ex), ex.columns.tolist(), ex.iloc[0, 0], ex.iloc[-1, 0])
exd = pd.to_datetime(ex.iloc[:, 0])
print('expiries not on a trading day (within bar range):', [str(d.date()) for d in exd if d >= n.index[0] and d <= n.index[-1] and d not in n.index])

# ------------------------------------------------------------------------------------ task 2 stocks
section('TASK 2 STOCKS')
s = pd.read_csv(os.path.join(DATA, 'stocks_daily_train_2021-04-26_2024-06-30.csv'), parse_dates=['date'])
print('rows', len(s), 'symbols', s.symbol.nunique(), 'range', s.date.min(), s.date.max(), 'cols', s.columns.tolist())
print('NIFTY50 rows in stock file:', (s.symbol == 'NIFTY50').sum())
print('dupes', s.duplicated(['date', 'symbol']).sum(), 'NaN', s.isna().sum().to_dict())
sym = [x.strip() for x in open(os.path.join(DATA, 'symbols.txt')) if x.strip()]
print('symbols.txt count', len(sym), 'not in file', sorted(set(sym) - set(s.symbol))[:20], 'in file not in list', sorted(set(s.symbol) - set(sym))[:20])
print('bad OHLC', ((s.high < s[['open', 'close']].max(axis=1) - 1e-6) | (s.low > s[['open', 'close']].min(axis=1) + 1e-6)).sum())
print('non-positive prices', (s[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum(), 'zero volume', (s.volume == 0).sum())
C = s.pivot(index='date', columns='symbol', values='close').sort_index()
V = s.pivot(index='date', columns='symbol', values='volume').sort_index()
O = s.pivot(index='date', columns='symbol', values='open').sort_index()
print('trading days', len(C), 'names per day: first', C.iloc[0].notna().sum(), 'eval start', C.loc['2022-03-02'].notna().sum(), 'last', C.iloc[-1].notna().sum())
first = C.apply(lambda x: x.first_valid_index())
last = C.apply(lambda x: x.last_valid_index())
print('listed after start', (first > C.index[0]).sum(), 'of which after 2022-03-01', (first > pd.Timestamp('2022-03-01')).sum())
print('ending early (last row before final date):\n', last[last < C.index[-1]].sort_values().to_string())
holes = (C.isna() & C.ffill().notna() & C.bfill().notna()).sum()
print('names with interior missing days', (holes > 0).sum(), 'worst:\n', holes.sort_values(ascending=False).head(10).to_string())
ret = C.pct_change(fill_method=None)
st = ret.stack()
print('|1d ret| > 30%:\n', st[st.abs() > 0.30].sort_values().to_string())
tv = (C * V)
med = tv.rolling(60, min_periods=40).median()
for d in ['2021-10-01', '2022-02-28', '2023-03-01', '2024-06-28']:
    m = med.loc[d].dropna()
    print(d, 'names w/ median-TV', len(m), '>=10cr', (m >= 1e8).sum(), '>=50cr', (m >= 5e8).sum(), '>=100cr', (m >= 1e9).sum())
print('close price dist at 2022-02-28:', C.loc['2022-02-28'].describe().round(1).to_dict())
print('names with adj close < 5 at eval start:', (C.loc['2022-02-28'] < 5).sum())
# volume adjusted? a split-adjusted series should have no huge one-day jump in the median traded value
section('EW universe vs Nifty')
nf = pd.read_csv(os.path.join(DATA, 'nifty50_daily_2021-04-26_2024-06-30.csv'), parse_dates=['date']).set_index('date').sort_index()
print('nifty cols', nf.columns.tolist(), 'rows', len(nf), 'dates differ from stocks:', len(set(nf.index) ^ set(C.index)), sorted(set(nf.index) ^ set(C.index))[:10])
w = C.loc['2022-03-02':]
print('Nifty total Mar22-Jun24: %.1f%%' % (100 * (nf.close.loc[w.index[-1]] / nf.close.loc['2022-03-02':].iloc[0] - 1)))
ew = ret.loc['2022-03-02':].mean(axis=1)
print('EW universe (daily rebal, all listed) total %.1f%%, daily-vol-ann %.1f%%, sharpe %.2f' % (100 * ((1 + ew).prod() - 1), 100 * ew.std() * 252 ** .5, ew.mean() / ew.std() * 252 ** .5))
liq = med.shift(1) >= 1e8
ew2 = ret.where(liq).loc['2022-03-02':].mean(axis=1)
print('EW liquid>=10cr total %.1f%% sharpe %.2f' % (100 * ((1 + ew2).prod() - 1), ew2.mean() / ew2.std() * 252 ** .5))
hold = (C.iloc[-1] / C.loc['2022-03-02'] - 1).dropna()
print('buy-hold stock total return Mar22-Jun24: median %.1f%%, mean %.1f%%, share negative %.1f%%' % (100 * hold.median(), 100 * hold.mean(), 100 * (hold < 0).mean()))
print('Stock return distribution (survivorship tell): min %.1f%%, p5 %.1f%%, p95 %.1f%%, max %.1f%%' % (100 * hold.min(), 100 * hold.quantile(.05), 100 * hold.quantile(.95), 100 * hold.max()))
