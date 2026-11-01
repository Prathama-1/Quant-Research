# Lab notebook

Running log of decisions and results. Newest state first under "Status"; everything tried goes in "Log" so that
`failed_ideas.md` and the report's parameter list can be written from real numbers.

## Status

- Tidied 2026-11-01: exploratory scripts and logs named in the Log below were removed from the tree. They are in git at commit `d45075f`.

- Data QA done (`research/qa_data.py`).
- Task 1 built: `task1.py`, tested by `research/t1_final.py` (results/t1_final.txt, .json). Overnight long, filtered by MA20,
  session so far >= -0.25%, 20d vol < 35%, paper-trade gate. Official: CAGR 22.49%, DD -4.73%, Sharpe 2.946, 110 trades/yr.
  d1 2.61, d2 1.94, 2x 1.78. Ex-2020 Sharpe 2.37. OOS extension mean 1.46 (0.51 to 2.53 by path seed).
  session_floor was chosen after seeing OOS, so OOS is not a clean holdout: grid median OOS is 1.12. Expect about 1 on hidden.
- Task 2 built: `task2.py`. Official: CAGR 48.03%, DD -11.24%, Sharpe 3.05, 94 trades/yr. Mean of 4 offsets 2.97. Ex-2023 2.43.
  Tough 1.57, bear 0.20, warm +0.7%. Rs 50 cr floor 1.92, 27.6% CAGR. Top 10 trades 56.9% of net P&L. Expect 1 to 1.5 on hidden.
- Deliverables done: `failed_ideas.md`, `survivorship.txt`, `report.pdf` via `report/build_report.py`, `README.md`,
  `Prathama_Bharadwaj.zip`. The zip's copies were re-run through the harness. Nothing committed or pushed: the user does that.
- The author name on the report and the zip was inferred from the user's email, and the user was asked to confirm it.

- Side tests after the push, not in the submission: market making (`alt_mm.py`) and pair trading (`alt_pairs.py`), logs in
  results/alt_mm.txt and alt_pairs.txt. Neither beats the submitted strategies. The pack has no order book, so true spread capture is
  untestable. On a random walk the quote simulator's loss shrinks to about +/-0.2 bps per fill as steps get finer, so it is roughly fair.
  Added to failed_ideas.md and the zip at the user's request; basket and order-book upside are marked as untested.

- Optimisation check, 1 Nov 2026 (`opt_check.py`, `opt_t1_1445.py`, `opt_t2_periods.py`, `opt_t2_alts.py`, full summary in data/findings.txt
  section 9). Re-tuning does not help out of sample, so both strategies stay as submitted unless the user decides otherwise. Open
  trade-off: Task 1 at 14:45 (Sharpe 3.57, but 2-bar delay -3.37). Warning sign: Task 2 made +0.7% in Feb-Jun 2024 against +17.8% for the
  universe, and 75% of its return came from Apr 2023 to Jan 2024. The report does not state these numbers yet.
- "Never used for tuning" was false and is fixed in report.pdf and failed_ideas.md; both overlaps are disclosed there. `opt_without_peeking.py`
  re-ranks without that data. Task 2 falls to 48 of 800, and 0.60 ties 0.70 on its own rules. Task 1 is 38 of 800, and floor 0 beats -0.25% on
  worst case but earns less. Not changing anything: the gaps are within noise and re-tuning would reuse the same data.

## Discrepancies in the brief

- Email: no hard cap on parameters. README and harness: max 6 in `PARAMS` (`check_params` raises). Following the harness.
- Email: ~585 stocks. Data and README: 555.
- Email: "Use 2021 for warm-up". README: data starts 2021-04-26.

## Log

### Task 1 (numbers: results/t1_baselines.txt, t1_lab_run1.txt, t1_lab2_run.txt, t1_lab3_run.txt, t1_decay.txt)
- Failed: bar-level MA crossover. 25/100 long-flat Sharpe 0.90 DD -13.3; long-short 0.49 DD -26.3. 50/200 LF 0.54.
- Failed: Donchian. 25/13 LF 0.92, LS 0.81; d1 0.69/0.55, 2x 0.53/0.23. Shorts never helped across 6 lookbacks.
- Failed: ORB. k=2,4,8 long-flat -0.11, 0.15, 1.03; 2x cost -1.14, -0.79, 0.05. Costs eat it.
- Failed: gap-down intraday short. th 0.25/0.5/1%: -0.25, -0.23, -0.23.
- Failed as-is: unfiltered overnight long, enter after 15:00. Sharpe 1.96 but DD -26.6%, d1 -5.02, d2 -4.40, 2x 0.81.
- Multi-day hold with MA filter: Sharpe 0.58 to 1.50, far below night-only, but low turnover (3-26 trades/yr).
- Filter marginals over 192 configs (all beat the bar): MA 15/20/25 flat; session floor helps OOS 0.90 -> 1.31, d1 1.95 -> 2.31.
- Synthetic OOS calibrated against real IS: real 2.72 vs synthetic 2.09-2.61 for MA20, so it understates if anything.
- All-nights premium gate is over-conservative (IS at -12 bps: -0.33 vs 0.63 no gate). Paper-trade gate never fires IS.
- Conditional table: Thursday night has lowest premium (7.5 bps vs 14-20), not used because expiry day changes in hidden.
- Pandas 3 returns read-only arrays from `.values`: use `to_numpy(copy=True)` before mutating.

### Task 2 (numbers: results/t2_explore.txt, t2_run1.txt to t2_run6b.txt, t2_final.txt, t2_checks.txt, t2_decomp.txt)
- Random 15 stocks, monthly: Sharpe 1.66 +/- 0.36, so the bar is close to chance on this universe. Tough case: 0.08.
- Signals: risk-adjusted momentum beats raw (1.93-1.96 vs 1.67). Residual momentum looks good in IC tests, not in portfolios.
- Stops and per-name MA exits hurt, with no drawdown benefit. Regime filter: sell-all beats keep-holdings once timing luck is averaged
  (an earlier single-offset read said the opposite).
- Breadth filter attacked in t2_run6 and t2_run6b. Same effect when measured on half the names and traded on the other half (2.58
  vs 2.57). 3-day delay drops it to 1.77. Thresholds of 0.7 and above score best in-sample but fail the tough case.
- Decomposition (t2_decomp): the first attempt redrew the random ranking at each review, which forced turnover and buried it in
  costs. Fixed: one random ranking per seed. Filter alone halves DD, selection alone adds return, and they compound.
- Writing data/findings.txt exposed three more errors, all corrected there: SCI's +30% was labelled real but looks like an artefact
  (qa_jumps.py); pandas 3's stack() keeps NaN, so 438,450 "returns" was really the panel size (true count 400,362); and the
  crossover range was misquoted. No effect on results: the strategy never traded the flagged names around their moves.
- Two claims in failed_ideas.md were wrong on first draft and corrected: a quintile figure that was actually next-day, and an
  unverified line about shorts in 2020. Check every claim against the logs before writing it.
