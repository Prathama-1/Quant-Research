"""Task 2: significance of the alternatives that stood out in the optimisation grid. Run from inside research/."""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from opt_check import boot_diff, t2_job
from common import t2_data

SUB = dict(score='sh126', band=.6, every=10, buffer=40, mv=.65, br='hi10')
ALTS = [('Breadth threshold 0.70', dict(SUB, band=.7)), ('Breadth threshold 0.65', dict(SUB, band=.65)), ('Skip-month signal, threshold 0.65', dict(SUB, score='mom126_21', band=.65)),
        ('Skip-month signal, threshold 0.60', dict(SUB, score='mom126_21')), ('10 positions', dict(SUB, K=10)), ('12 positions, sell rank 20', dict(SUB, K=12, buffer=20))]
if __name__ == '__main__':
    d, base = t2_job(SUB)
    days = t2_data()['close'].index
    days = days[days >= pd.Timestamp('2022-03-02')][1:]
    late, ins = days >= pd.Timestamp('2024-02-01'), (days >= pd.Timestamp('2023-04-01')) & (days <= pd.Timestamp('2024-01-31'))
    sh = lambda r: r.mean() / r.std() * 252 ** .5
    print('%-36s Sharpe %.2f DD %.1f%% CAGR %.1f%% | tough %.2f DD %.1f%% CAGR %.1f%% | warm %.1f%% | Feb-Jun 24 %.1f%% | outside Apr23-Jan24 Sharpe %.2f' % ('Submitted', d['sharpe'], 100 * d['dd'], 100 * d['cagr'], d['tough'], 100 * d['tough_dd'], 100 * d['tough_cagr'], 100 * d['warm'], 100 * (np.prod(1 + base[late]) - 1), sh(base[~ins])))
    for nm, c in ALTS:
        a, r = t2_job(c)
        ci, pos = boot_diff(r, base)
        print('%-36s Sharpe %.2f DD %.1f%% CAGR %.1f%% | tough %.2f DD %.1f%% CAGR %.1f%% | warm %.1f%% | Feb-Jun 24 %.1f%% | outside Apr23-Jan24 Sharpe %.2f | diff vs submitted 95%% CI %+.2f to %+.2f, P(better) %.0f%%' % (
            nm, a['sharpe'], 100 * a['dd'], 100 * a['cagr'], a['tough'], 100 * a['tough_dd'], 100 * a['tough_cagr'], 100 * a['warm'], 100 * (np.prod(1 + r[late]) - 1), sh(r[~ins]), ci[0], ci[2], 100 * pos))
