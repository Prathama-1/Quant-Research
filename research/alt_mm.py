"""Side test, not part of the submission: can public market-making logic work on Nifty?

The pack has no order book, so true spread capture cannot be measured. Two approximations:
 A. Avellaneda-Stoikov style quoting on 15-min bars, own accounting (the harness only fills at the next open). Each bar we rest a bid
    and an ask around the last close, shifted against inventory (reservation price). A quote fills if the bar trades through it.
    Inventory is limited to -1, 0, +1 as in Task 1, and flat by the close, as the model assumes. This assumes no queue, no latency
    and no adverse selection beyond what the bar shows, so read it as an upper bound.
    Control: the same bar shapes in shuffled order with no gaps or drift. Quoting there should earn about zero, which tests the fill model.
 B. A version that can run in the official harness: taking liquidity, fading stretches away from a short average.
"""
import os

import numpy as np
import pandas as pd

from common import H, RESULTS, Clock, hm, t1_data, t1_eval, row, stress_table

bars, ex = t1_data()
ck = Clock(bars)
LAST = np.r_[ck.first[1:], True]        # last bar of each day; offline accounting, used only to flatten


def shuffled(b, seed=1):
    """Same intraday bar shapes in random order, each opening at the previous close, with the mean bar return removed. Overnight
    gaps are dropped: left in, they would land mid-session and run over stale quotes. So this is close to a driftless market with
    realistic bar shapes, where quoting should earn about zero before costs. A clearly non-zero result means the fill model is biased."""
    rng = np.random.default_rng(seed)
    o, h, l, c = b.open.values, b.high.values, b.low.values, b.close.values
    rel = np.c_[h / o, l / o, c / o]
    rel = rel / np.mean(rel[:, 2])                      # remove drift
    rel = rel[rng.permutation(len(b))]
    out = np.zeros((len(b), 4)); p = o[0]
    for i in range(len(b)):
        out[i] = (p, rel[i, 0] * p, rel[i, 1] * p, rel[i, 2] * p)
        p = out[i, 3]
    return pd.DataFrame(out, index=b.index, columns=['open', 'high', 'low', 'close'])


def random_walk(b, seed=1, steps=15):
    """A true martingale with `steps` moves per bar, each bar's volatility scaled from the real bar's range. With coarse steps the price
    jumps past a resting quote, which fills at its limit and is instantly under water, so a loss is expected. It should shrink towards
    zero as steps get finer. If it does not, the simulator is biased."""
    rng = np.random.default_rng(seed)
    sig = ((b.high.values / b.low.values - 1) / 3.0).clip(1e-5) / np.sqrt(steps)
    out = np.zeros((len(b), 4)); p = b.open.values[0]
    for i in range(len(b)):
        path = p * np.exp(np.cumsum(sig[i] * rng.standard_normal(steps) - .5 * sig[i] ** 2))
        out[i] = (p, max(p, path.max()), min(p, path.min()), path[-1])
        p = path[-1]
    return pd.DataFrame(out, index=b.index, columns=['open', 'high', 'low', 'close'])


def mm(b, half_bps, skew=.5, through_bps=0., cost_bps=0., both=True):
    """both=True: if a bar touches both quotes, both fill (round trip, optimistic). False: only the side the bar moved against fills."""
    o, h, l, c = b.open.values, b.high.values, b.low.values, b.close.values
    cost, half, thr = cost_bps / 1e4, half_bps / 1e4, through_bps / 1e4
    pnl = np.zeros(len(c)); q = 0; fills = 0; rts = 0
    mark = o[0]
    for i in range(len(c)):
        if ck.first[i]:
            mark = o[i]                                 # flat overnight, so no gap P&L
        mid = mark if ck.first[i] else c[i - 1]
        r = mid * (1 - q * skew * half)                 # reservation price: long inventory lowers both quotes
        bid, ask = min(r * (1 - half), mid), max(r * (1 + half), mid)      # a quote through the market would trade at the market
        hit_b, hit_a = l[i] < bid * (1 - thr), h[i] > ask * (1 + thr)
        p = 0.0
        if hit_b and hit_a and both:
            if q <= 0:                                  # buy first, then sell
                p += q * (bid / mark - 1); mark = bid; p += (q + 1) * (ask / mark - 1); mark = ask
            else:                                       # long: sell first, then rebuy
                p += q * (ask / mark - 1); mark = ask; p += (q - 1) * (bid / mark - 1); mark = bid
            p -= 2 * cost; fills += 2; rts += 1
        elif hit_b or hit_a:
            side = (1 if c[i] < o[i] else -1) if (hit_b and hit_a) else (1 if hit_b else -1)
            if abs(q + side) <= 1:
                px = bid if side > 0 else ask
                p += q * (px / mark - 1); mark = px; q += side; p -= cost; fills += 1
        p += q * (c[i] / mark - 1); mark = c[i]
        if LAST[i] and q != 0:                          # flatten by crossing at the close
            p -= cost; q = 0; fills += 1
        pnl[i] = p
    d = pd.Series(pnl, index=b.index.normalize()).groupby(level=0).sum()
    s, _ = H.task1_stats(dict(daily=d, pos=pd.Series(np.zeros(len(c)), index=b.index), trades=[]))
    s['fills_yr'] = fills / s['years']
    s['bps_per_fill'] = 1e4 * pnl.sum() / max(fills, 1)
    return s


def line(tag, s):
    return '%-40s Sharpe %6.2f  DD %7.1f%%  CAGR %6.1f%%  fills/yr %6.0f  net bps per fill %6.2f' % (tag, s['sharpe'], 100 * s['max_dd'], 100 * s['cagr'], s['fills_yr'], s['bps_per_fill'])


if __name__ == '__main__':
    sb = shuffled(bars)
    print('Costs per side used, bps: 0 = none; 0.5 = illustrative low; 2.5 = harness; 2.78 = NSE futures now, averaging 5 bps STT on sells + about 0.2 exchange + 0.2 stamp on buys\n')
    print('=== A1. Quoting on real bars vs the same bars shuffled, optimistic fills, zero cost')
    for hs in [2, 5, 10, 20, 40]:
        print(line('half-spread %2d bps, real' % hs, mm(bars, hs)))
        print(line('half-spread %2d bps, SHUFFLED control' % hs, mm(sb, hs)))
    print('\n=== A1b. Simulator test on a pure random walk, half-spread 10, zero cost. Net bps per fill should shrink towards 0 as steps get finer')
    for sk in [0, .5]:
        for st_ in [15, 60, 240, 900]:
            v = [mm(random_walk(bars, sd, st_), 10, skew=sk) for sd in [1, 2]]
            print('skew %.1f, %3d steps per bar: net bps per fill %s | Sharpe %s | CAGR %s' % (sk, st_, [round(float(x['bps_per_fill']), 2) for x in v], [round(x['sharpe'], 2) for x in v], [round(100 * x['cagr'], 1) for x in v]))
    print('\n=== A2. Cost sensitivity, optimistic fills, real bars')
    for hs in [5, 10, 20, 40]:
        for cb in [0, .5, 1.3, 2.5, 2.78]:
            print(line('half-spread %2d, cost %.2f' % (hs, cb), mm(bars, hs, cost_bps=cb)))
    print('\n=== A3. Harder fills, half-spread 10 bps, cost 0 and 2.78')
    for cb in [0, 2.78]:
        print(line('both-sides fill, through 0 bps, cost %.2f' % cb, mm(bars, 10, cost_bps=cb)))
        print(line('both-sides fill, through 1 bp, cost %.2f' % cb, mm(bars, 10, through_bps=1, cost_bps=cb)))
        print(line('both-sides fill, through 3 bps, cost %.2f' % cb, mm(bars, 10, through_bps=3, cost_bps=cb)))
        print(line('adverse-side only, cost %.2f' % cb, mm(bars, 10, cost_bps=cb, both=False)))
    print('\n=== A4. Inventory skew (reservation-price strength), half-spread 10 bps, zero cost')
    for sk in [0, .25, .5, 1, 2]:
        print(line('skew %.2f' % sk, mm(bars, 10, skew=sk)))

    print('\n=== B. Official harness: fade stretches from a short EMA, intraday only, taking liquidity')
    c = bars.close
    rows = []
    for n in [4, 8, 25]:
        ema = c.ewm(span=n, adjust=False).mean()
        sd = (c / ema - 1).rolling(250, min_periods=50).std()
        z = ((c / ema - 1) / sd).to_numpy(copy=True)
        for k in [1.0, 1.5, 2.0, 3.0]:
            w = np.zeros(len(c)); pos = 0.0
            for i in range(len(c)):
                if pos == 0 and z[i] > k: pos = -1.0
                elif pos == 0 and z[i] < -k: pos = 1.0
                elif pos > 0 and z[i] > 0: pos = 0.0
                elif pos < 0 and z[i] < 0: pos = 0.0
                if ck.tod[i] >= hm('15:00'): pos = 0.0
                w[i] = pos
            st = stress_table(w, bars, ex)
            b_ = st.loc['base']; rows.append(dict(ema=n, k=k, sharpe=b_.sharpe, dd=b_.dd, cagr=b_.cagr, tr_yr=b_.tr_yr, d1=st.loc['delay_1'].sharpe, c2=st.loc['2x_cost'].sharpe))
    t = pd.DataFrame(rows); print(t.round(2).to_string(index=False))
    t.to_csv(os.path.join(RESULTS, 'alt_mm_taker.csv'), index=False)
    print('\nFor comparison, submitted Task 1: Sharpe 2.95, DD -4.7%, CAGR 22.5%')
