"""Checks the submission against the assignment email and the pack's README, rule by rule. Run under an environment with pymupdf for the
report checks:  python research/compliance_check.py"""
import ast
import importlib.util
import os
import re
import sys
import time
import types
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import harness_public as H  # noqa: E402

DATA = os.path.join(ROOT, 'data')
results = []


def check(name, ok, detail=''):
    results.append(bool(ok))
    print('%-5s %-62s %s' % ('PASS' if ok else 'FAIL', name, detail))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, name + '.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


T1, T2 = load('task1'), load('task2')
BARS = os.path.join(DATA, 'nifty50_15min_train_2017-04-03_2022-01-31.csv')
EXP = os.path.join(DATA, 'nifty_monthly_expiries_2017-01_2022-01.csv')
STK = os.path.join(DATA, 'stocks_daily_train_2021-04-26_2024-06-30.csv')
NIF = os.path.join(DATA, 'nifty50_daily_2021-04-26_2024-06-30.csv')

print('=== Task 1')
bars, ex = H.load_bars(BARS), H.load_expiries(EXP)
t0 = time.time(); want = H.task1_positions(T1, bars, dict(T1.PARAMS)); el1 = time.time() - t0
res = H.task1_backtest(bars, want.values, ex)
s1, _ = H.task1_stats(res)
check('Beats the bar: Sharpe > 1.0', s1['sharpe'] > 1, '%.2f' % s1['sharpe'])
check('Beats the bar: max drawdown better than -15%', s1['max_dd'] > -.15, '%.1f%%' % (100 * s1['max_dd']))
check('Positions only -1, 0, +1 and never NaN', want.isin([-1., 0., 1.]).all() and not want.isna().any(), str(sorted(set(want))))
check('No position in the first 200 bars', (res['pos'].iloc[:H.T1_WARMUP] == 0).all())
check('Harness look-ahead check passes', not H.lookahead_check(T1, bars, dict(T1.PARAMS), want))
s2, _ = H.task1_stats(H.task1_backtest(bars, np.r_[0, want.values[:-1]], ex)); s3, _ = H.task1_stats(H.task1_backtest(bars, np.r_[0, 0, want.values[:-2]], ex)); s4, _ = H.task1_stats(H.task1_backtest(bars, want.values, ex, cost=2 * H.T1_COST))
print('      for the record: 1-bar delay %.2f, 2-bar delay %.2f, 2x costs %.2f' % (s2['sharpe'], s3['sharpe'], s4['sharpe']))

print('\n=== Task 2')
D = H.load_stocks([STK], [NIF])
log = dict(max_pos=0, max_wt=0.0, wt_at=None, max_entry=0.0, adds=0, bad=0, n=0, over=[], cash_min=1.0)


def wrapped(date, hist, portfolio, params, state):
    o = T2.orders(date, hist, portfolio, params, state)
    v = portfolio['value']; pos = portfolio['positions']
    log['n'] += 1; log['max_pos'] = max(log['max_pos'], len(pos)); log['cash_min'] = min(log['cash_min'], portfolio['cash'])
    log.setdefault('first', str(date.date()))
    log['hist_end_ok'] = log.get('hist_end_ok', True) and hist['close'].index[-1] == date
    for s, p in pos.items():
        w = p['value'] / v
        if w > log['max_wt']: log['max_wt'], log['wt_at'] = w, (s, str(date.date()))
        if w > .15: log['over'].append((str(date.date()), s, round(w, 4)))
    for x in o:
        if x.get('side') not in ('buy', 'sell') or x.get('type') not in ('market', 'stop'): log['bad'] += 1
        if x.get('side') == 'buy':
            log['max_entry'] = max(log['max_entry'], x.get('weight', 0))
            if x['symbol'] in pos: log['adds'] += 1
    return o


m = types.SimpleNamespace(orders=wrapped)
t0 = time.time(); r2 = H.task2_backtest(m, D, dict(T2.PARAMS), H.T2_START, H.T2_END); el2 = time.time() - t0
s, _ = H.task2_stats(r2)
check('Beats the bar: Sharpe > 1.9', s['sharpe'] > 1.9, '%.2f' % s['sharpe'])
check('Beats the bar: max drawdown better than -25%', s['max_dd'] > -.25, '%.1f%%' % (100 * s['max_dd']))
check('Scored window starts 2 Mar 2022 and ends 28 Jun 2024', str(r2['eq'].index[0].date()) == '2022-03-02' and str(r2['eq'].index[-1].date()) == '2024-06-28')
check('First orders() call is on the last day before the window', log['first'] == '2022-02-28', log['first'])
check('orders() never receives data past its own date', log['hist_end_ok'])
check('At most 15 positions', log['max_pos'] <= 15, 'max %d' % log['max_pos'])
check('No more than 15% of equity at entry', log['max_entry'] <= .15, 'max %.2f%%' % (100 * log['max_entry']))
check('No more than 15% of equity per name at any close (email rule)', not log['over'], 'max %.2f%% %s, %d day-positions over' % (100 * log['max_wt'], log['wt_at'], len(log['over'])))
check('No leverage: invested share never above 100%', (r2['gross'] <= 1 + 1e-9).all() and log['cash_min'] >= -1e-9, 'max %.1f%%, min cash %.4f' % (100 * r2['gross'].max(), log['cash_min']))
check('No adding to an open position', log['adds'] == 0)
check('Only valid order types, long only', log['bad'] == 0)
print('      for the record: top days over 15%%: %s' % log['over'][:6])
# look-ahead: cut the data at several dates. Results up to each cut must not change.
eq_full = H.task2_backtest(T2, D, dict(T2.PARAMS), H.T2_START, H.T2_END)['eq']
la_bad = 0
for cut in ['2022-09-30', '2023-06-30', '2024-02-29']:
    Dc = {k: v.loc[:cut] for k, v in D.items()}
    e_ = H.task2_backtest(T2, Dc, dict(T2.PARAMS), H.T2_START, None)['eq']
    la_bad += int((np.abs(e_.values - eq_full.loc[:cut].values) > 1e-12).sum())
check('No look-ahead: cutting future data changes nothing before the cut', la_bad == 0, '3 cuts, %d days differ' % la_bad)
# Corrupt all future prices: decisions before the cut must not change either
rng = np.random.default_rng(1)
cut_i = np.searchsorted(D['close'].index, pd.Timestamp('2023-06-30'), side='right')
Dx = {k: v.copy() for k, v in D.items()}
for k in ['open', 'high', 'low', 'close', 'volume']:
    a = Dx[k].to_numpy(copy=True); a[cut_i:] = a[cut_i:] * rng.uniform(.5, 1.5, a[cut_i:].shape); Dx[k] = pd.DataFrame(a, index=Dx[k].index, columns=Dx[k].columns)
e_x = H.task2_backtest(T2, Dx, dict(T2.PARAMS), H.T2_START, H.T2_END)['eq']
check('No look-ahead: corrupting future prices changes nothing before', np.abs(e_x.loc[:'2023-06-30'].values - eq_full.loc[:'2023-06-30'].values).max() < 1e-12)
check('Deterministic: two runs identical', np.array_equal(eq_full.values, H.task2_backtest(T2, D, dict(T2.PARAMS), H.T2_START, H.T2_END)['eq'].values))

print('\n=== Both tasks, code rules')
check('Runs in under 10 minutes', el1 + el2 < 600, 'Task 1 %.1fs, Task 2 %.1fs' % (el1, el2))
want2 = H.task1_positions(T1, bars, dict(T1.PARAMS))
check('Task 1 deterministic: two runs identical', want.equals(want2))
for nm, mod in [('task1', T1), ('task2', T2)]:
    src = open(os.path.join(ROOT, nm + '.py'), encoding='utf-8').read()
    tree = ast.parse(src, feature_version=(3, 9))
    imports = sorted({a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)})
    code = re.sub(r'""".*?"""', '', src, flags=re.S)
    code_nc = re.sub(r'#.*', '', code)
    check('%s: at most 6 entries in PARAMS' % nm, len(mod.PARAMS) <= 6, str(len(mod.PARAMS)))
    check('%s: parses as Python 3.9, imports only numpy and pandas' % nm, set(imports) <= {'numpy', 'pandas'}, str(imports))
    check('%s: no dates in code' % nm, not re.search(r'20\d\d|Timestamp|datetime\(', code_nc))
    check('%s: no file, network or random use' % nm, not re.search(r'open\(|read_csv|read_|to_csv|urlopen|requests|random|\.seed', code_nc))
    syms = [x.strip() for x in open(os.path.join(DATA, 'symbols.txt')) if x.strip()]
    used = [x for x in syms if re.search(r"['\"]%s['\"]" % re.escape(x), code_nc)]
    check('%s: no stock named in the logic' % nm, not used, str(used) if used else '')
    check('%s: has a module description and PARAMS comments' % nm, ast.get_docstring(tree) and '#' in src)
h = open(os.path.join(ROOT, 'harness_public.py'), 'rb').read()
orig = r'C:\Users\Prathama\Downloads\candidate_pack\harness_public.py'
check('Harness identical to the firm\'s copy', os.path.exists(orig) and open(orig, 'rb').read() == h)

print('\n=== Submission zip')
zp = os.path.join(ROOT, 'Prathama_Bharadwaj.zip')
with zipfile.ZipFile(zp) as z:
    names = [i.filename for i in z.infolist()]
    check('Exactly the required files', sorted(names) == sorted(['task1.py', 'task2.py', 'report.pdf', 'failed_ideas.md', 'survivorship_note.md']), str(names))
    check('No data folder or harness', not any(('data' in n and n.endswith('.csv')) or 'harness' in n for n in names))
    stale = [n for n in names if z.read(n) != open(os.path.join(ROOT, n), 'rb').read()]
    check('Zip copies match the current files', not stale, str(stale) if stale else '')
    check('Size is small enough to email', os.path.getsize(zp) < 20 * 1024 ** 2, '%d KB' % (os.path.getsize(zp) // 1024))

print('\n=== failed_ideas.md')
fi = open(os.path.join(ROOT, 'failed_ideas.md'), encoding='utf-8').read()
rows = [l for l in fi.splitlines() if l.startswith('| ') and not l.startswith('| Idea') and not l.startswith('| Version')]
check('At least 3 ideas', len(rows) >= 3, '%d table rows' % len(rows))
check('No claim that the warm window was unused', 'not used for tuning' not in fi and 'never used for tuning' not in fi)

print('\n=== report.pdf')
try:
    import pymupdf
    d = pymupdf.open(os.path.join(ROOT, 'report.pdf'))
    txt = ' '.join(p.get_text() for p in d)
    txt = re.sub(r'\s+', ' ', txt)
    check('At most 3 pages', len(d) <= 3, '%d pages' % len(d))
    for label, pat in [('Idea and economic reasoning', 'Why it should work'), ('Sensitivity tables', 'Sensitivity'), ('Year-wise results', 'Year'), ('Turnover and cost drag', 'Turnover and cost drag'), ('Best year removed', 'Best year'),
                       ('Share of profit from top 10 trades', 'Top 10 trades'), ('What would break it', 'What would break'), ('1-bar and 2-bar delay', '2-bar fill delay'), ('2x costs', '2x costs'), ('Parameters with tested ranges', 'Range tested'),
                       ('Survivorship discussion', 'Survivorship'), ('Disclosure of the 2021 window', 'reserves 2021')]:
        check('Report covers: ' + label, pat in txt)
    check('No false "never used for tuning" claim', 'never used for tuning' not in txt)
except ImportError:
    print('      skipped: pymupdf is not installed in this environment')
    results.append(False)
print('\n%d of %d checks passed' % (sum(results), len(results)))
