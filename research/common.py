"""Shared research helpers. All accounting goes through the official harness so research numbers match the score."""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'research', 'results')
BARS_FILE = os.path.join(DATA, 'nifty50_15min_train_2017-04-03_2022-01-31.csv')
EXP_FILE = os.path.join(DATA, 'nifty_monthly_expiries_2017-01_2022-01.csv')
STOCK_FILE = os.path.join(DATA, 'stocks_daily_train_2021-04-26_2024-06-30.csv')
NIFTY_FILE = os.path.join(DATA, 'nifty50_daily_2021-04-26_2024-06-30.csv')

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import harness_public as H  # noqa: E402

pd.set_option('display.width', 240, 'display.max_columns', 80, 'display.max_rows', 400)


# ------------------------------------------------------------------------------------------- task 1
def t1_data():
    return H.load_bars(BARS_FILE), H.load_expiries(EXP_FILE)


def delayed(want, k):
    """want filled k bars later than the harness's normal next-open fill."""
    want = np.asarray(want, float)
    return want if k == 0 else np.r_[np.zeros(k), want[:-k]]


def t1_eval(want, bars, ex, cost=H.T1_COST, delay=0, start=None, end=None):
    """Official backtest + stats, with an optional fill delay and cost multiple applied from outside the harness."""
    res = H.task1_backtest(bars, delayed(want, delay), ex, cost=cost)
    s, eq = H.task1_stats(res, start, end)
    s['trades_per_yr'] = s['trades'] / max(s['years'], 1e-9)
    return s, eq, res


def row(s):
    return dict(cagr=round(100 * s['cagr'], 2), dd=round(100 * s['max_dd'], 2), sharpe=round(s['sharpe'], 3), calmar=round(s['calmar'], 2),
                tr_yr=round(s['trades_per_yr'], 1), win=round(100 * s['win_rate'], 1), tim=round(100 * s['time_in_market'], 1))


def stress_table(want, bars, ex):
    """The stress cases the brief asks for, as one DataFrame."""
    rows = {'base': t1_eval(want, bars, ex)[0], 'delay_1': t1_eval(want, bars, ex, delay=1)[0], 'delay_2': t1_eval(want, bars, ex, delay=2)[0],
            '2x_cost': t1_eval(want, bars, ex, cost=2 * H.T1_COST)[0]}
    return pd.DataFrame({k: row(v) for k, v in rows.items()}).T


def yearly(s):
    return {k: round(100 * v, 1) for k, v in s['yearly'].items()}


class Clock:
    """Causal calendar arrays for a bar index. tod = minutes since midnight. first = first bar of the day."""

    def __init__(self, bars):
        idx = bars.index
        self.tod = np.asarray(idx.hour * 60 + idx.minute)
        day = idx.normalize()
        self.first = np.r_[True, np.asarray(day[1:] != day[:-1])]
        self.day_id = np.cumsum(self.first) - 1
        self.year = np.asarray(idx.year)


def hm(s):
    h, m = s.split(':')
    return int(h) * 60 + int(m)


# ------------------------------------------------------------------------------------------- task 2
def t2_data():
    return H.load_stocks([STOCK_FILE], [NIFTY_FILE])
