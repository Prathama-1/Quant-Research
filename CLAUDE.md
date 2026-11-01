# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Take-home for H TO H PARTNERS LLP (Quant Research Intern): two systematic strategies, scored by the firm's harness on
data we never see. Task 1 is Nifty futures timing on 15-min bars; Task 2 is a long-only NSE stock portfolio on daily bars.
The hidden run is the real test, so robustness beats in-sample polish.

| Task | Must beat in-sample | Reference (firm's own) | Hidden data |
|---|---|---|---|
| 1 | Sharpe > 1.0 and MaxDD better than -15%, Apr 2017 to Jan 2022 | CAGR 23.3%, DD -12.8%, Sharpe 1.22, ~88 trades/yr, ~55% in market | Feb 2022 to Oct 2026 |
| 2 | Sharpe > 1.9 and MaxDD better than -25%, 2 Mar 2022 to 30 Jun 2024 | CAGR 51.6%, DD -23.9%, Sharpe 1.93, ~100 trades/yr | Jul 2024 to Oct 2026, judged risk-adjusted |

`PACK_README.md` is the firm's README (interfaces, data notes). Read it before changing strategy code.
`research/NOTES.md` is the running lab notebook: current best configs, decisions, and every result worth keeping.

## Commands

```
python harness_public.py task1 task1.py     # official score for Task 1 (also runs the look-ahead check)
python harness_public.py task2 task2.py     # official score for Task 2
python research/compliance_check.py         # the brief's rules, one by one (the report checks need pymupdf)
```

Run these from inside `research/`, because they import `common.py` and write relative paths. Each writes to `results/`:

```
python t1_final.py     # full battery on the real task1.py, about 2 min, writes t1_final.json
python t2_final.py     # full battery on the real task2.py, about 15 min, writes t2_final.json
python t2_checks.py    # liquidity buckets, ghosts, speed, shocks, cold start, writes t2_checks.json
python t2_decomp.py    # timing vs selection, bootstrap CI, deflated Sharpe, equity CSVs, writes t_decomp.json
```

Side tests, not part of the submission: `python alt_mm.py`, `python alt_pairs.py` (market making and pairs), and
`python opt_check.py t1` or `t2` with `opt_t1_1445.py`, `opt_t2_periods.py`, `opt_t2_alts.py` (is further tuning worthwhile? It was not).
`colab.py` in the root holds every insight as a chart, split into cells by `# %%`; the user pastes these into their own notebook.

`report/build_report.py` builds `report.pdf` (must stay at 3 pages) from those JSON files and needs reportlab, matplotlib, pillow
and pymupdf, which are not in the system Python. It also saves page PNGs to `report/figures/` for visual checks.

There is no build, lint or test framework. Runtime needs only numpy and pandas. This machine has Python 3.12.6,
pandas 3.0.2, numpy 2.4.4. The README promises Python 3.9+, so strategy files must avoid 3.10+ syntax and
pandas-version-specific behaviour. Do not edit `harness_public.py`: the firm scores with an identical copy.

Machine limits that have already caused failures:
- 8 GB RAM. `t2_lab.world()` holds full feature panels per process, so use at most 3 workers with it (8 ran out of memory once).
- Windows path length. pip cannot install into a venv under the long scratchpad path; use a short path such as `%TEMP%\qrv`.
- pandas 3 returns read-only arrays from `.values`; use `to_numpy(copy=True)` before mutating.

## How the harness and strategies fit together

Strategies are plain modules exposing `PARAMS` plus `positions()` (Task 1) or `orders()` (Task 2). The harness loads them
by path. `check_params` raises if `PARAMS` has more than 6 entries, even though the email says there is no hard cap.
The harness wins: keep at most 6 entries, and list anything else varied in research in the report's sensitivity section.

Research code should import the harness and reuse its functions so that results match the official accounting:
`task1_backtest(bars, want, expiries, cost=...)`, `task1_stats(res, start, end)`, and
`task2_backtest(m, D, params, start, end, cost=...)`, `task2_stats(res, start, end)`. Stress tests are done from outside:
pass `cost=2*T1_COST` for 2x costs, and shift `want` by k bars for a k-bar fill delay.

Research splits into two layers. `t1_lab2.py` and `t2_lab.py` are fast research engines with their own signal code (Task 2
precomputes features), used to choose a design. `t1_final.py`, `t2_final.py`, `t2_checks.py` and `t2_decomp.py` test the
real `task1.py` and `task2.py`. Quote the second layer as results and treat the first only as comparisons.
The exploratory sweeps (`t1_lab.py`, `t2_run*.py`, `qa_*.py`) and their logs were removed from the tree on 2026-11-01 and are still in
git at commit `d45075f`. Their config CSVs stay in `research/results/`, because `t2_decomp.py` reads them.

Out-of-sample tools built here, since the firm's hidden data is unavailable:
- `research/synth.py`: Task 1 bars for Feb 2022 to Jun 2024 from the real daily Nifty OHLC with synthetic paths inside each day.
  Valid only for day-level logic; bar-level strategies give meaningless results on it.
- `strip_drift` in `t2_lab.py`: the "tough" (universe about +10%/yr) and "bear" (about -10%/yr) scenarios for Task 2.
- The warm-up window, Oct 2021 to Feb 2022, as a weak-market check for Task 2. Signals use it only as history, but looking at it
  informed design choices, and the brief reserves 2021 for warm-up, so the report discloses this.

Timing luck is large in Task 2: moving the rebalance day alone shifts Sharpe by about 0.3. Compare configs as means over offsets.

**Task 1 fill timing is the part that is easy to get wrong.** `want[i]` is decided at the close of bar i and filled at the
open of bar i+1, and the overnight gap accrues to the position held at the previous close. So:
- To hold the overnight gap, `want` must be set by the close of the 15:00 bar (fills at the 15:15 open).
- A decision at the close of the 15:15 bar fills at the next 09:15 open, which is after the gap. This is also how to exit
  exactly at the morning open: set `want` to 0 at the 15:15 bar's close.
- Under a k-bar delay these schedules slip, so entries should leave room before the close.

`lookahead_check` re-runs `positions()` on truncated prefixes and compares the last value, so detect "last bar of the day"
from the clock, never from the next row. Short sessions break clock logic (see below).

**The hidden run may start cold.** `run_task1` takes `bar_files` and `task1_stats` takes `start`/`end`, so the firm may pass
train + hidden bars or hidden bars alone, and only the first 200 bars (8 days) are guaranteed warm-up. Indicators must
degrade gracefully with little history (EMAs or `min_periods`), and the same applies to Task 2's `hist`.

**Task 2 execution order next day:** market sells at the open, then buys in listed order, then stop sells. Cash from a stop
sell is not available until the following day, a stop sell cannot hit a position bought that day, and orders on a symbol
with no price lapse, so sells must be re-issued every day until filled. Idle cash earns nothing. `hist` grows daily, so
always work on a tail (`iloc[-300:]`) to stay fast; the 10-minute limit includes the much longer hidden run.

## Rules from the brief

- No look-ahead, no hard-coded dates (the calendar may only enter through the data), no external data, no reading data
  files from inside `positions()`/`orders()`, deterministic.
- Task 1 positions must be in {-1, 0, +1}; Task 2 max 15 positions, max 15% per new position, no adding to a position.
- ML is allowed but needs a pinned `requirements.txt`, fixed seeds, offline CPU, and training only on provided data.

Deliverables, in one zip named `<your_name>.zip` (currently `Prathama_Bharadwaj.zip`), without `data/` or the harness:
`task1.py`, `task2.py`, any helper `.py`, `failed_ideas.md` (3+ ideas with actual numbers and a one-line reason),
`survivorship.txt`, and `report.pdf` (max 3 pages). Rebuild the zip after any change to these. The report covers: economic rationale, a sensitivity table of at least 3x3 with each parameter's tested range,
year-wise results, turnover and cost drag, results with the best year removed, share of profit from the top 10 trades
(Task 2), what would break the strategy, and for Task 1 the 1-bar delay, 2-bar delay and 2x-cost results.

## Data facts verified by the data QA (`research/qa_data.py` in commit `d45075f`)

The full log is `data/findings.txt` (local only, because `data/` is gitignored). The facts most likely to matter:

- Task 1: 29,803 bars over 1,193 days. The only short session is 2021-11-04, 3 bars stamped 14:45 to 15:15 (not the real
  Muhurat clock time) and all flat at the previous close, so the real session is missing from this file. README says
  2024-03-02, 2024-05-18 and 2025-10-21 are also short. 2020-02-01 is a Saturday session. All 34 flat bars fall on 5 dates
  and look like halts or a frozen feed.
  `load_bars` drops `n_1min`, so strategies never see it. Two large within-day jumps on 2017-07-10 are the NSE outage.
- Task 2: 555 symbols (the email says ~585, README says 555), 790 days. 2022-03-01 was a holiday, so the first evaluated day
  is 2022-03-02 and `orders()` is first called on 2022-02-28.
- Jump anomalies: SPLPETRO -49.1% on 2022-06-07 (in the README) and NMDC **+85.7%** on 2022-10-27 (not in the README).
  Cell 2.5 of `colab.py` separates real moves from artefacts by traded value. SCI +30.0% (2023-03-31) and ALLCARGO -21.0%
  (2023-04-18) look like artefacts but sit under the 45% rule in `task2.py`; no trades were affected. Real moves also exist, so
  filters must not be too tight: ZEEL -32.6% on 2024-01-23 and +40.0% on 2021-09-14, OFSS +30.1%, PAYTM -20% (limit-down lock).
  Interior missing days: FORCEMOT 76, AWL 71, E2E-BE 41, SHRIPISTON 3, DIACABS-BE 1.
- The daily Nifty file's close differs from the 15-min file's last close (std about 9 bps, mean about 0), so treat it as an
  approximation when using it for out-of-sample checks.

## Current strategies

- **Task 1, `task1.py`:** long overnight only, decided at the 14:30 close and exited at the next open, if price is above its 20-day
  average, the session so far is at least -0.25%, 20-day vol is under 35%, and the last 120 paper trades average at least 7 bps.
- **Task 2, `task2.py`:** rank by mean/std of 126-day returns, vol under 65%, top 15, buy on a stop at the 10-day high, review every
  10 days, sell below rank 40. Sell everything when under 60% of liquid stocks are above their 50-day average.

## Findings that shape the design

- **Task 1: Nifty's return is all overnight.** 2017 to Jan 2022: overnight Sharpe +3.45 (t 7.5), intraday -1.57 (t -3.4).
  The daily file gives an out-of-sample check, Feb 2022 to Jun 2024: overnight +2.63 (t 4.0), intraday -0.78. The 09:15 bar
  alone loses about 6.5 bps a day in every year, which is what a 1-bar delay on a morning exit would absorb. This premium may
  be partly a spot-auction effect that futures cannot capture, while the harness credits it in full as spot P&L + carry.
- **Task 2: the bar is mostly beta and survivorship.** The universe was picked by liquidity in Oct 2026. A daily-rebalanced
  equal-weight universe scores Sharpe 2.48 (+148%) in the evaluation window against Nifty +44.6%, before costs, and only
  11.4% of names fell. Judge selection skill against that benchmark, not against Nifty.

## Working conventions

- Quote numbers only from a fresh run of the current files. Save results under `research/results/`.
- Never run `git commit` or `git push`; the user pushes, and the final message gives them the commands. The remote
  (`Prathama-1/Quant-Research`, default branch `main`) is public, so `.gitignore` keeps the firm's data and harness out.
