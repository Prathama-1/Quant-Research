"""
Task 1: Nifty 50 futures, overnight premium with a trend and risk filter.

Idea. Almost all of Nifty's return arrives while the market is shut: from 2017 to Jan 2022 the overnight gap has a Sharpe of
+3.4 and the open-to-close session -1.6. Holding overnight should be paid because the risk is hard to hedge or exit while the
cash market is shut, and intraday leverage is normally closed by the end of the day. The premium is thin and the tail is large when
the market is falling, so the strategy only takes the risk when the market is trending up, today's session is not weak, and
volatility is not extreme. It also keeps a paper record of its own trades and stops if they no longer cover costs.

Rules, all decided at the close of the `decide_at` bar and filled at the next open:
  long overnight if  price > its `ma_days`-day average             trend
                 and session so far (open to now) >= session_floor  no weak close
                 and 20-day realised vol < vol_cap                  no violent markets
                 and the last `gate_nights` paper trades averaged >= gate_bps
  otherwise flat. Exit at the next morning's open. Never short.

No look-ahead: each night uses bars up to its own decision bar, plus results of earlier nights only.
"""
import numpy as np
import pandas as pd

PARAMS = {
    'ma_days': 20,              # trend lookback, in days
    'decide_at': '14:30',       # decision bar; 14:30 leaves room to fill before the close even with a 2-bar delay
    'session_floor': -0.0025,   # skip the night if the session so far is below this
    'vol_cap': 0.35,            # skip the night if 20-day annualised realised vol is above this
    'gate_nights': 120,         # paper trades in the "is it still paying?" gate
    'gate_bps': 7.0,            # 2 sides at 2.5 bps + 1.87 bps of carry, so the gate asks whether the trade covers its costs
}

LAST_BAR = 15 * 60 + 15         # exit decision: flat at the 15:15 close fills at the next morning's open
VOL_DAYS = 20
MIN_SESSION_BARS = 10           # short special sessions (Muhurat etc.) are not traded


def positions(bars: pd.DataFrame, params: dict) -> pd.Series:
    idx = bars.index
    o, c = bars['open'].to_numpy(dtype=float), bars['close'].to_numpy(dtype=float)
    tod = np.asarray(idx.hour * 60 + idx.minute)
    day_stamp = np.asarray(idx.normalize())
    is_first = np.r_[True, day_stamp[1:] != day_stamp[:-1]]
    day = np.cumsum(is_first) - 1
    first = np.flatnonzero(is_first)                                # bar index of each day's first bar
    bars_so_far = np.arange(len(c)) - first[day] + 1

    h, m = params['decide_at'].split(':')
    decide = int(h) * 60 + int(m)
    dec = np.flatnonzero((tod == decide) & (bars_so_far >= MIN_SESSION_BARS))       # one decision bar per normal session
    want = np.zeros(len(c))
    if len(dec) == 0:
        return pd.Series(want, index=idx)
    px = pd.Series(c[dec])

    n = int(params['ma_days'])
    trend = (px > px.shift(1).rolling(n, min_periods=max(2, n // 2)).mean()).to_numpy()
    session = (c[dec] / o[first[day[dec]]] - 1) >= params['session_floor']
    vol = px.pct_change().rolling(VOL_DAYS, min_periods=5).std().fillna(0.0).to_numpy() * np.sqrt(252)
    raw = trend & session & (vol < params['vol_cap'])

    # Paper trade of every raw signal: fill at the next bar's open, out at the first open of the following day.
    # A night's result is only known on a later day, so the gate at a decision uses earlier raw nights only.
    next_first = np.searchsorted(first, dec, side='right')          # position in `first` of the day after each decision
    has_next = (next_first < len(first)) & (dec + 1 < len(c))
    paper = np.full(len(dec), np.nan)
    paper[has_next] = o[first[next_first[has_next]]] / o[dec[has_next] + 1] - 1

    k = int(params['gate_nights'])
    paying = pd.Series(paper[raw]).rolling(k, min_periods=max(2, k // 2)).mean().shift(1) * 1e4
    gate_open = np.ones(len(dec), bool)
    gate_open[raw] = (paying.fillna(np.inf) >= params['gate_bps']).to_numpy()

    on_day = np.zeros(day[-1] + 1, bool)
    on_day[day[dec]] = raw & gate_open
    want[on_day[day] & (tod >= decide) & (tod < LAST_BAR)] = 1.0
    return pd.Series(want, index=idx)
