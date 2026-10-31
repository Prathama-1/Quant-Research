# Failed ideas

Everything below ran through `harness_public.py` accounting, after costs. Task 1 uses Apr 2017 to Jan 2022 and Task 2 uses 2 Mar 2022 to 30 Jun 2024, unless stated.

Some terms used in the tables:
- **d1, d2:** 1-bar and 2-bar fill delay.
- **2x:** twice the stated cost.
- **OOS:** Feb 2022 to Jun 2024, using real daily OHLC with synthetic intraday paths.
- **Warm:** Oct 2021 to Feb 2022, a weak stretch before the evaluation window that was not used for tuning.
- **Tough:** every stock loses 29% a year (log) and Nifty 10.8%, so the equal-weight universe earns about 10% a year instead of about 47%.

## Task 1: Nifty futures (bar: Sharpe > 1.0, MaxDD better than -15%)

| Idea | Numbers | Why it failed |
|---|---|---|
| MA crossover on 15-min bars, the obvious starting point | 25/100 long-flat: Sharpe 0.90, DD -13.3%. Long-short: 0.49, DD -26.3%. 50/200 long-flat: 0.54, DD -18.6% | Holding through the session pays the negative open-to-close drift (Sharpe -1.6) to collect the overnight gap (+3.4), so it earns little more than buy and hold. |
| Donchian breakout, 6 lookbacks from 25 to 1,250 bars | Best was 25/13: long-flat 0.92, long-short 0.81. At 2x: 0.53 and 0.23 | Never reached Sharpe 1.0, and the best version turns over too much to survive 2x costs. |
| Adding shorts to any trend rule | Long-short was worse than long-flat in 10 of 11 configs, e.g. Donchian 125/50: 0.47 vs 0.73 | Overnight shorts pay the positive gap (mean +15.6 bps a night). |
| Opening range breakout, intraday only, k = 2, 4, 8 bars | Long-flat: -0.11, 0.15, 1.03. At 2x: -1.14, -0.79, 0.05 | About 100 to 150 trades a year on a thin edge. The best case clears the bar and then dies at 2x costs. |
| Short the session after a gap down (threshold 0.25%, 0.5%, 1%) | -0.25, -0.23, -0.23 | Gaps do not continue: the correlation of the gap with the same session is -0.05, so the short loses after costs. |
| Unfiltered overnight long, enter after the 15:00 bar | Sharpe 1.96, but DD -26.6%. d1: -5.02, d2: -4.40. 2x: 0.81 | Fails the drawdown bar in crashes, and a one-bar delay fills the next morning, after the gap it was trying to hold. Moving the decision to 14:30 removes that failure (d1 0.51) but not the drawdown (-33.5%). |
| Same trend filter, but hold for several days | MA20: Sharpe 1.07, DD -12.5%, d1 0.84. Night-only with the same filter: 2.72, DD -5.9%, d1 1.93 | Saves costs (12 trades a year, not 162) but keeps paying the session drift. It is the natural fallback if the open print proves untradable. |
| Longer trend lookbacks for the filter (MA 50 to 200) | In-sample 2.56 to 2.69, about the same as MA20 at 2.72. OOS: 0.19 to 0.83 vs 1.02 | Equal in-sample and clearly worse once 2022 arrives: slow filters stay long through corrections. |
| "Is the premium still paying?" gate measured on all nights (250 days, 7 bps) | With 12 bps removed from each real in-sample gap: Sharpe -0.33, vs 0.63 with no gate | The strategy only trades the better nights, so an all-nights average shuts it down while it is still profitable. Replaced by a paper record of its own trades. |

## Task 2: NSE momentum (bar: Sharpe > 1.9, MaxDD better than -25%)

| Idea | Numbers | Why it failed |
|---|---|---|
| The brief's naive strategy: top 10 by 6-month return, monthly | Sharpe 1.64, CAGR 53.1%, DD -22.2%. Warm: -12.3%, DD -20.9%. Tough: 0.50, DD -37.6% | No better than chance: 15 random stocks, monthly, average 1.66 (std 0.36). It is market beta plus survivorship. |
| Trailing stops (means from a 612-config factorial) | No stop: Sharpe 1.76, DD -19.2%. 25%: 1.66, -20.1%. 15%: 1.51, -19.3%. Tough: 0.42, 0.30, 0.18 | Stops volatile winners out without cutting drawdown, because market-wide falls hit every name together. |
| Per-stock trend exit (sell below own MA 20, 50, 100, 200) | Sharpe 1.51, 1.49, 1.70, 1.67 vs 1.73 with none. CAGR 21% to 44% vs 49% | Same reason as stops: it sells winners on routine pullbacks. |
| Low volatility as the selection signal | Sharpe 1.59, CAGR 18.2%, DD -12.1%. Tough: -0.98, CAGR -10.7% | The in-sample Sharpe was all drift. With the drift removed it has no selection power. |
| 52-week-high proximity | Sharpe 1.28, CAGR 27.5%. Tough: 0.04 | Picks slow stocks sitting near their highs, not the ones still moving. |
| Residual momentum, which looked best in rank-IC tests for weak markets | Sharpe 1.84 vs 1.93 for plain return/vol. Tough: 0.49 vs 0.62 | The IC edge in weak windows did not reach the top 15, and it adds a beta estimate for no gain. |
| Keep holdings when the regime turns off, only stop buying | Sharpe 1.75 vs 1.94 for selling. Tough: 0.47 vs 0.73 | Leaves the portfolio fully exposed through the fall that the filter has just detected. |
| Hysteresis on breadth to cut whipsaw (exit below 40%, re-enter above 60%) | Mean over 48 configs: Sharpe 1.65, DD -17.5%, vs 2.74, DD -10.8% for a single 60% line | Exiting early does most of the work. Waiting for 40% gives that away to save a few trades. |
| Stricter breadth threshold, the best in-sample result (70%) | Sharpe 3.33, DD -6.6%. Tough: 0.78 (75%: -0.05), vs 1.57 at 60% | Rejected even though it is the in-sample peak: in a lower-drift market it would rarely invest at all. |
| Injecting collapsing "ghost" stocks to measure survivorship | Up to 60 ghosts, falling 60% to 90%: Sharpe 2.99 to 3.05 vs 3.05, about no change | The test itself failed. The strategy exits losers fast, so this misses the larger effect: winners that are only in the universe because they won. See the survivorship note. |

The naive strategy's Sharpe, CAGR and DD, the breadth threshold row and the ghost row come from harness runs of the actual files. The other Task 2 numbers come from research configs whose rules differ from the final `task2.py`, so read them as comparisons within each row, not as results of the final strategy.
