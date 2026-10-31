"""Are large one-day moves real or adjustment artefacts? A real jump comes with a surge in traded value. An artefact does not: the
price and volume series are rescaled against each other, so price x volume stays near normal even though the price jumps.
For every one-day move above 20%, compares that day's traded value with the median of the prior 20 days, and shows how much of
the move was already there at the open. Also reports whether the submitted strategy held the stock on the day."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import harness_public as H  # noqa: E402

pd.set_option('display.width', 200, 'display.max_columns', 30, 'display.max_rows', 200)
D = H.load_stocks([os.path.join(ROOT, 'data', 'stocks_daily_train_2021-04-26_2024-06-30.csv')], [os.path.join(ROOT, 'data', 'nifty50_daily_2021-04-26_2024-06-30.csv')])
C, O, Hi, Lo, V = D['close'], D['open'], D['high'], D['low'], D['volume']
ret = C.pct_change(fill_method=None)
TV = C * V
base = TV.shift(1).rolling(20, min_periods=10).median()
vbase = V.shift(1).rolling(20, min_periods=10).median()
prev = C.shift(1)

spec = importlib.util.spec_from_file_location('task2', os.path.join(ROOT, 'task2.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
res = H.task2_backtest(m, D, dict(m.PARAMS), H.T2_START, H.T2_END)
T = res['trades']


def held(sym, day):
    x = T[T.symbol == sym]
    return bool(((pd.to_datetime(x.entry_date) <= day) & (pd.to_datetime(x.exit_date) >= day)).any())


rows = []
print('real stock-day returns: %d of a %d x %d panel' % (ret.notna().sum().sum(), ret.shape[0], ret.shape[1]))
for (d, s), r in ret.stack().items():
    if not abs(r) > .20:          # also skips NaN, which pandas 3's stack() keeps
        continue
    rows.append(dict(date=str(d.date()), symbol=s, ret=100 * r, gap_pct=100 * (O.loc[d, s] / prev.loc[d, s] - 1), day_range=100 * (Hi.loc[d, s] / Lo.loc[d, s] - 1),
                     tv_ratio=TV.loc[d, s] / base.loc[d, s], vol_ratio=V.loc[d, s] / vbase.loc[d, s], held=held(s, d)))
x = pd.DataFrame(rows).sort_values('tv_ratio')
print('moves above 20%%: %d. Sorted by traded value on the day / prior 20-day median (tv_ratio).' % len(x))
print('A real jump normally has a large tv_ratio. Near 1 suggests a rescaling artefact, especially with most of the move at the open.\n')
print(x.round(2).to_string(index=False))
print('\nmoves with tv_ratio under 2: %d' % (x.tv_ratio < 2).sum())
print('median tv_ratio over all moves above 20%%: %.1f' % x.tv_ratio.median())
print('moves held by the strategy on the day: %d' % x.held.sum())
print('\nmoves made almost entirely at the open with little extra trading, and every trade the strategy made in those names:')
for s, dd in [('ALLCARGO', '2023-04-18'), ('SCI', '2023-03-31'), ('SPLPETRO', '2022-06-07'), ('ASTERDM', '2024-04-23'), ('NMDC', '2022-10-27'), ('PAYTM', '2024-02-02')]:
    t = T[T.symbol == s]
    print(' %-9s move on %s | trades: %s' % (s, dd, [(str(a)[:10], str(b)[:10], round(float(c), 3)) for a, b, c in zip(t.entry_date, t.exit_date, t.ret)]))
print('\nmoves by date, dates with more than one:'); c = x.groupby('date').symbol.apply(list); print(c[c.apply(len) > 1].to_string())
print('moves by name, names with more than one:'); c = x.groupby('symbol').date.apply(list); print(c[c.apply(len) > 1].to_string())
x.to_csv(os.path.join(ROOT, 'research', 'results', 'qa_jumps.csv'), index=False)
