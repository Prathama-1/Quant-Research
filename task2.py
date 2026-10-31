"""
Task 2: long-only NSE momentum, gated by market breadth.

Idea. Momentum is paid because good news diffuses slowly through the mid and small caps, but it crashes when the market turns
and weak names rebound. So the strategy only carries momentum risk while the broad market is healthy. When too few stocks are
above their own trend it goes fully to cash.

Rules, decided each day after the close and executed at the next open:
  risk-on   share of liquid stocks above their 50-day average >= breadth_min. If risk-off, sell everything and buy nothing.
  rank      liquid stocks by mean/std of daily returns over `lookback` days: return per unit of risk, which favours steady
            winners over one-off spikes. Stocks with annualised vol above max_vol are not eligible.
  sell      every REVIEW_EVERY days, sell a holding that has fallen out of the top `sell_rank` or is no longer eligible.
  buy       while risk-on and slots are free, place stop orders on the best-ranked names: buy only if tomorrow trades above the
            high of the last `breakout_days` days, so capital goes to names that are actually moving.
  size      1/top_k of portfolio value per position.

Corporate actions. One-day moves beyond +/-45% are treated as unadjusted events (SPLPETRO -49%, NMDC +86%): that day's return
is set to zero in every indicator, a holding is sold at the next open, and the stock cannot be newly bought for JUMP_COOLOFF
days. The loss on the jump itself is booked by the harness; no rule can see it coming, which is one reason no position is ever
above 1/top_k at entry.
"""
import numpy as np

PARAMS = {
    'lookback': 126,        # days in the risk-adjusted momentum score
    'top_k': 15,            # number of positions
    'sell_rank': 40,        # a holding outside this rank is sold at the next review
    'breadth_min': 0.60,    # risk-on threshold on market breadth
    'max_vol': 0.65,        # annualised 120-day volatility cap on eligibility
    'breakout_days': 10,    # entry trigger: high of the last N days
}

REVIEW_EVERY = 10           # days between rank reviews
MIN_TRADED_VALUE = 1e8      # Rs 10 crore 60-day median traded value, as in the template
BREADTH_MA = 50             # a stock counts towards breadth if above this many days' average
VOL_DAYS = 120
JUMP = 0.45                 # one-day move treated as a corporate-action artefact
JUMP_COOLOFF = 10


def _mean(x, min_n):
    n = (~np.isnan(x)).sum(axis=0)
    out = np.full(x.shape[1], np.nan)
    ok = n >= min_n
    out[ok] = np.nansum(x[:, ok], axis=0) / n[ok]
    return out


def _std(x, min_n):
    n = (~np.isnan(x)).sum(axis=0)
    out = np.full(x.shape[1], np.nan)
    ok = n >= max(min_n, 2)
    m = np.nansum(x[:, ok], axis=0) / n[ok]
    out[ok] = np.sqrt(np.nansum((x[:, ok] - m) ** 2, axis=0) / (n[ok] - 1))
    return out


def _median(x, min_n):
    n = (~np.isnan(x)).sum(axis=0)
    out = np.full(x.shape[1], np.nan)
    ok = n >= min_n
    out[ok] = np.nanmedian(x[:, ok], axis=0)
    return out


def orders(date, hist, portfolio, params, state):
    close = hist['close']
    t = len(close)
    state.setdefault('first', t)
    lookback, k = int(params['lookback']), int(params['top_k'])
    tail = max(lookback, VOL_DAYS, BREADTH_MA, 60) + 2

    names = list(close.columns)
    c = close.to_numpy(dtype=float)[-tail:]
    v = hist['volume'].to_numpy(dtype=float)[-tail:]
    h = hist['high'].to_numpy(dtype=float)
    last = c[-1]
    held = portfolio['positions']
    if len(c) < 2:
        return []

    with np.errstate(invalid='ignore', divide='ignore'):
        r = c[1:] / c[:-1] - 1
    is_jump = np.abs(r) >= JUMP                         # NaN compares False, so missing days stay NaN below
    clean = np.where(is_jump, 0.0, r)

    # Clean price level, used only for comparing a stock with its own recent average
    level = np.cumprod(1.0 + np.nan_to_num(np.vstack([np.zeros(c.shape[1]), clean])), axis=0)
    level = np.where(np.isnan(c), np.nan, level)
    ma = _mean(level[-BREADTH_MA:], BREADTH_MA * 3 // 5)

    traded = _median((c * v)[-60:], 40)
    with np.errstate(invalid='ignore'):
        liquid = traded >= MIN_TRADED_VALUE
        above = level[-1] > ma
    counted = liquid & ~np.isnan(last)
    risk_on = counted.any() and above[counted].mean() >= params['breadth_min']

    selling = set(held) if not risk_on else set()
    selling |= {s for s in held if is_jump[-1][names.index(s)]}      # an unexplained jump: we no longer know what we own
    review = (t - state['first'] + int(params.get('offset', 0))) % REVIEW_EVERY == 0     # 'offset' is only set by research runs
    free = k - (len(held) - len(selling))

    buys = []
    if review or (risk_on and free > 0):
        score_std = _std(clean[-lookback:], lookback * 2 // 3)
        with np.errstate(invalid='ignore', divide='ignore'):
            score = _mean(clean[-lookback:], lookback * 2 // 3) / score_std
            vol = _std(clean[-VOL_DAYS:], VOL_DAYS // 2) * np.sqrt(252)
            eligible = liquid & (vol < params['max_vol']) & ~np.isnan(score) & (close.notna().sum().to_numpy() >= lookback)
        ranked = np.argsort(-np.where(eligible, score, -np.inf), kind='stable')[:int(eligible.sum())]
        rank = {names[j]: i for i, j in enumerate(ranked)}

        if review:
            for s in held:
                if rank.get(s, len(names)) >= params['sell_rank']:
                    selling.add(s)
        free = k - (len(held) - len(selling))

        if risk_on and free > 0:
            n_b = int(params['breakout_days'])
            recent = h[-n_b:]
            jumped = is_jump[-JUMP_COOLOFF:].any(axis=0) if JUMP_COOLOFF else np.zeros(len(names), bool)
            high = np.where(np.isnan(recent), -np.inf, recent).max(axis=0)
            high[(~np.isnan(recent)).sum(axis=0) < n_b // 2] = np.nan
            for j in ranked:
                s = names[j]
                if free <= 0:
                    break
                if s in held or jumped[j] or np.isnan(high[j]):
                    continue
                buys.append({'symbol': s, 'side': 'buy', 'type': 'stop', 'trigger': float(high[j]), 'weight': 1.0 / k, 'max_positions': k})
                free -= 1

    return [{'symbol': s, 'side': 'sell', 'type': 'market'} for s in selling] + buys
