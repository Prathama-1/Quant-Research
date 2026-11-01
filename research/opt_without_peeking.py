"""Would the submitted settings still be chosen without the data they should not have leaned on?
Task 2: re-rank the 800-config grid with the Oct 2021 to Feb 2022 window removed from the score.
Task 1: re-rank the 800-config grid with the Feb 2022 to Jun 2024 check removed, keeping only the real 2017 to Jan 2022 data.
Reads the saved grids from opt_check.py, so it runs in seconds."""
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
pd.set_option('display.width', 200, 'display.max_columns', 30)
t2 = pd.read_csv(os.path.join(HERE, 'results', 'opt_t2_grid.csv'))
t1 = pd.read_csv(os.path.join(HERE, 'results', 'opt_t1_grid.csv'))

print('=== Task 2, combined rank on Sharpe, drawdown, tough Sharpe and tough drawdown, without the 2021 window')
t2['z'] = sum(t2[c].rank(pct=True) for c in ['sharpe', 'dd', 'tough', 'tough_dd'])
sub = t2[(t2.score == 'sh126') & (t2.band == .6) & (t2.every == 10) & (t2.buffer == 40) & (t2.mv == .65) & (t2.br == 'hi10')].index[0]
order = t2.z.rank(ascending=False)
print('Submitted: position %d of %d' % (order[sub], len(t2)))
show = ['score', 'band', 'every', 'buffer', 'mv', 'br', 'sharpe', 'dd', 'cagr', 'tough', 'tough_dd', 'warm']
print(t2.sort_values('z', ascending=False).head(8)[show].round(3).to_string(index=False))
print('\nMean combined rank by each choice, higher is better:')
for dim in ['score', 'band', 'mv', 'br', 'every', 'buffer']:
    print(' %-7s %s' % (dim, t2.groupby(dim).z.mean().round(2).to_dict()))
print('\nSubmitted rules, breadth threshold only, without the 2021 window:')
print(t2[(t2.score == 'sh126') & (t2.every == 10) & (t2.buffer == 40) & (t2.mv == .65) & (t2.br == 'hi10')][['band', 'sharpe', 'dd', 'tough', 'tough_dd', 'z']].round(3).to_string(index=False))

print('\n=== Task 1, worst of base, 1-bar delay, 2-bar delay and 2x costs on real data, without the Feb 2022 to Jun 2024 check')
t1['worst'] = t1[['sharpe', 'd1', 'd2', 'c2']].min(axis=1)
sub1 = t1[(t1.ma == 20) & (t1['floor'] == -.0025) & (t1.dec == '14:30') & (t1.cap == .35)].index[0]
print('Submitted: worst-case %.2f, position %d of %d' % (t1.worst[sub1], t1.worst.rank(ascending=False)[sub1], len(t1)))
print(t1.sort_values('worst', ascending=False).head(8).round(3).to_string(index=False))
print('\nMean by each choice:')
for dim, cols in [('floor', ['sharpe', 'd1', 'd2', 'c2', 'worst', 'cagr']), ('ma', ['sharpe', 'd1', 'd2', 'c2', 'worst', 'cagr']), ('dec', ['sharpe', 'd1', 'd2', 'c2', 'worst']), ('cap', ['sharpe', 'd1', 'd2', 'c2', 'worst'])]:
    print(t1.groupby(dim)[cols].mean().round(2).to_string()); print()
print('Submitted rules, session floor only, real data only:')
print(t1[(t1.ma == 20) & (t1.dec == '14:30') & (t1.cap == .35)][['floor', 'sharpe', 'dd', 'cagr', 'd1', 'd2', 'c2', 'worst']].round(3).to_string(index=False))
