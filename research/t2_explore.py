"""Task 2 exploration: which signals actually pick better stocks? In a universe that rose 148%, raw portfolio returns say little.
Here each signal is judged cross-sectionally against the equal-weight liquid universe: rank IC and top-bucket EXCESS return at
several horizons, by sub-period. All features use data up to t; forward returns start at t+1's open.
The 2021 warm-up window is included as a separate, weaker market check."""
import numpy as np
import pandas as pd

from common import t2_data

D = t2_data()
C, O, H_, L, V = D['close'], D['open'], D['high'], D['low'], D['volume']
N = D['nifty']

ret = C.pct_change(fill_method=None)
clean = ret.where((ret.abs() < .45) | ret.isna(), 0.0)  # one-day moves beyond 45% are treated as corporate-action artefacts; NaN stays NaN
P = (1 + clean.fillna(0)).cumprod().where(C.notna())
TV = (C * V)
liq = TV.rolling(60, min_periods=40).median()
EW = clean.where(liq.shift(1) >= 1e8).mean(axis=1)
mkt_ret = N.close.pct_change()


def mom(n, skip=0):
    return P.shift(skip) / P.shift(n) - 1


vol60 = clean.rolling(60, min_periods=40).std()
vol120 = clean.rolling(120, min_periods=60).std()
hi252 = P.rolling(252, min_periods=120).max()
beta = pd.DataFrame({s: clean[s].rolling(120, min_periods=60).cov(mkt_ret) for s in clean.columns}).div(mkt_ret.rolling(120, min_periods=60).var(), axis=0)
resid = clean.sub(beta.shift(1).mul(mkt_ret, axis=0))
pos = (clean > 0).rolling(126, min_periods=80).mean()
neg = (clean < 0).rolling(126, min_periods=80).mean()

F = {
    'mom21': mom(21), 'mom63': mom(63), 'mom126': mom(126), 'mom252': mom(252), 'mom126_21': mom(126, 21), 'mom252_21': mom(252, 21),
    'rev5': -mom(5), 'rev21': -mom(21),
    'sharpe126': clean.rolling(126, min_periods=80).mean() / clean.rolling(126, min_periods=80).std(),
    'sharpe252': clean.rolling(252, min_periods=120).mean() / clean.rolling(252, min_periods=120).std(),
    'hi252': P / hi252 - 1,
    'residmom126': resid.rolling(126, min_periods=80).sum() / resid.rolling(126, min_periods=80).std(),
    'residmom63': resid.rolling(63, min_periods=40).sum() / resid.rolling(63, min_periods=40).std(),
    'fip126': np.sign(mom(126)) * (pos - neg),
    'lowvol60': -vol60, 'lowvol120': -vol120, 'lowbeta': -beta,
    'size_tv': np.log(liq),
    'turnover_shock': np.log(TV.rolling(20, min_periods=10).mean() / TV.rolling(120, min_periods=60).mean()),
    'above_ma200': P / P.rolling(200, min_periods=100).mean() - 1,
    'above_ma50': P / P.rolling(50, min_periods=30).mean() - 1,
    'upvol_ratio': np.log(TV.where(clean > 0).rolling(60, min_periods=10).sum() / TV.where(clean < 0).rolling(60, min_periods=10).sum()),
}
F['mom_lowvol'] = F['mom126'].rank(axis=1, pct=True) + F['lowvol120'].rank(axis=1, pct=True)
F['sharpe_hi'] = F['sharpe126'].rank(axis=1, pct=True) + F['hi252'].rank(axis=1, pct=True)

periods = {'warm 21Oct-22Feb': ('2021-10-15', '2022-02-28'), 'eval A 22Mar-22Dec': ('2022-03-01', '2022-12-31'), 'eval B 23Jan-23Dec': ('2023-01-01', '2023-12-31'), 'eval C 24Jan-24Jun': ('2024-01-01', '2024-06-30'), 'eval all': ('2022-03-01', '2024-06-30')}
HORIZONS = [5, 21, 63]
fwd = {h: (O.shift(-1 - h) / O.shift(-1) - 1) for h in HORIZONS}
fwd = {h: v.where(v.abs() < 1.5) for h, v in fwd.items()}
step = 5                                                # sample every 5 days to reduce overlap

print('Mean rank IC x100 (t in brackets) and excess return of top 15 names vs liquid EW, % per holding period. Liquid = TV >= Rs 10 cr.')
rows = []
for name, f in F.items():
    for h in HORIZONS:
        for pname, (a, z) in periods.items():
            dates = C.loc[a:z].index[::step]
            ics, tops = [], []
            for d in dates:
                ok = (liq.loc[d] >= 1e8) & f.loc[d].notna() & fwd[h].loc[d].notna()
                if ok.sum() < 50: continue
                x, y = f.loc[d][ok], fwd[h].loc[d][ok]
                ics.append(x.rank().corr(y.rank()))
                tops.append(y[x.nlargest(15).index].mean() - y.mean())
            if len(ics) < 5: continue
            ics, tops = np.array(ics), np.array(tops)
            eff = max(len(ics) * step / h, 2)           # overlap-adjusted sample size
            rows.append(dict(sig=name, h=h, per=pname, ic=100 * ics.mean(), t=ics.mean() / ics.std() * eff ** .5, top=100 * tops.mean(), top_t=tops.mean() / tops.std() * eff ** .5))
R = pd.DataFrame(rows)
R.to_csv('results/t2_ic.csv', index=False)
for h in HORIZONS:
    print('\n--- horizon %d days: IC x100 (t) | top-15 excess %% (t)' % h)
    x = R[R.h == h]
    tb = {}
    for s in F:
        tb[s] = {p: '%5.1f(%4.1f)|%5.1f(%4.1f)' % tuple(x[(x.sig == s) & (x.per == p)][['ic', 't', 'top', 'top_t']].values[0]) if len(x[(x.sig == s) & (x.per == p)]) else '' for p in periods}
    print(pd.DataFrame(tb).T[list(periods)].to_string())
