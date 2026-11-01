"""Task 2 period by period against the universe: does the edge fade late in the window? Run from anywhere."""
import importlib.util, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import harness_public as H
spec = importlib.util.spec_from_file_location('task2', os.path.join(ROOT, 'task2.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
D = H.load_stocks([os.path.join(ROOT, 'data', 'stocks_daily_train_2021-04-26_2024-06-30.csv')], [os.path.join(ROOT, 'data', 'nifty50_daily_2021-04-26_2024-06-30.csv')])
res = H.task2_backtest(m, D, dict(m.PARAMS), H.T2_START, H.T2_END)
eq, T, gross = res['eq'], res['trades'], res['gross']
C, V, N = D['close'], D['volume'], D['nifty']['close']
ret = C.pct_change(fill_method=None)
clean = ret.where((ret.abs() < .45) | ret.isna(), 0.0)
P = (1 + clean.fillna(0)).cumprod().where(C.notna())
med = (C * V).rolling(60, min_periods=40).median()
ew = clean.where(med.shift(1) >= 1e8).mean(axis=1).reindex(eq.index).fillna(0)
above = (P > P.rolling(50, min_periods=30).mean()).astype(float).where(C.notna())
br = above.where(med >= 1e8).mean(axis=1).reindex(eq.index)
on = br >= .6
PER = [('2022 Mar-Jun', '2022-03-02', '2022-06-30'), ('2022 Jul-Dec', '2022-07-01', '2022-12-31'), ('2023 Jan-Mar', '2023-01-01', '2023-03-31'), ('2023 Apr-Sep', '2023-04-01', '2023-09-30'), ('2023 Oct-Jan 24', '2023-10-01', '2024-01-31'), ('2024 Feb-Jun', '2024-02-01', '2024-06-30')]
rows = []
r = eq.pct_change().fillna(0)
for nm, a, z in PER:
    e_ = (1 + r.loc[a:z]).prod() - 1
    u_ = (1 + ew.loc[a:z]).prod() - 1
    n_ = N.reindex(eq.index).loc[a:z]; n2 = n_.iloc[-1] / n_.iloc[0] - 1
    t_ = T[(pd.to_datetime(T.entry_date) >= a) & (pd.to_datetime(T.entry_date) <= z)]
    o_ = on.loc[a:z].astype(int)
    rows.append(dict(period=nm, strategy=100 * e_, universe=100 * u_, nifty=100 * n2, invested=100 * gross.loc[a:z].mean(), regime_flips=int(o_.diff().abs().sum()), trades=len(t_), win=100 * (t_.ret > 0).mean() if len(t_) else np.nan, mean_trade=100 * t_.ret.mean() if len(t_) else np.nan))
print(pd.DataFrame(rows).round(1).to_string(index=False))
def perf(x):
    e_ = (1 + x).cumprod()
    return 'Sharpe %.2f, CAGR %.1f%%, DD %.1f%%, days %d' % (x.mean() / x.std() * 252 ** .5, 100 * (e_.iloc[-1] ** (252 / len(x)) - 1), 100 * (e_ / e_.cummax() - 1).min(), len(x))


inside = (r.index >= '2023-04-01') & (r.index <= '2024-01-31')
print('\nWhole window:            ', perf(r))
print('Apr 2023 to Jan 2024:    ', perf(r[inside]))
print('Everything else:         ', perf(r[~inside]))
print('Same split, universe:     inside', perf(ew[inside]), '| outside', perf(ew[~inside]))
print('Share of the total log return from that stretch: %.0f%%' % (100 * np.log1p(r[inside]).sum() / np.log1p(r).sum()))
late = r.index >= '2024-02-01'
print('Feb to Jun 2024:          strategy', perf(r[late]), '| universe', perf(ew[late]))
pk = eq.idxmax()
print('\nPeak equity %.3f on %s; last %.3f, %.1f%% below the peak. Max drawdown low on %s' % (eq.max(), pk.date(), eq.iloc[-1], 100 * (eq.iloc[-1] / eq.max() - 1), (eq / eq.cummax() - 1).idxmin().date()))
print('From the peak to the end: strategy %.1f%%, universe %.1f%%, Nifty %.1f%%' % (100 * (eq.iloc[-1] / eq.loc[pk] - 1), 100 * ((1 + ew.loc[pk:].iloc[1:]).prod() - 1), 100 * (N.reindex(eq.index).iloc[-1] / N.reindex(eq.index).loc[pk] - 1)))
