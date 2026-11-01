# Quant-Research

My solution to a quant research take-home with two systematic strategies:

1. **Nifty futures timing**, on 15-minute bars.
2. **NSE stock momentum**, long-only and positional, on daily bars.

| | Sharpe | Max drawdown | CAGR | Trades/yr | Bar to beat |
|---|---|---|---|---|---|
| Task 1, Apr 2017 to Jan 2022 | 2.95 | -4.7% | 22.5% | 110 | Sharpe > 1.0, DD better than -15% |
| Task 2, 2 Mar 2022 to 30 Jun 2024 | 3.05 | -11.2% | 48.0% | 94 | Sharpe > 1.9, DD better than -25% |

In-sample numbers flatter both strategies. See `report.pdf` for the stress tests, the out-of-sample checks and what I expect on unseen data.

## The strategies

- **Task 1 (`task1.py`):** go long overnight only when Nifty is above its 20-day average, today's session is not weak, volatility is not extreme, and the strategy's own paper trades still cover costs. Otherwise stay flat. Nifty's return arrives almost entirely overnight, and the filters avoid the nights when that premium is poorest and riskiest.
- **Task 2 (`task2.py`):** hold up to 15 stocks with the best six-month return per unit of risk, buying them on a breakout. Go fully to cash when fewer than 60% of liquid stocks are above their 50-day average.

## What is here

| Path | Contents |
|---|---|
| `task1.py`, `task2.py` | The two strategies |
| `report.pdf` | Three-page write-up |
| `failed_ideas.md` | What did not work, with numbers |
| `survivorship.txt` | The Task 2 universe's bias and how I estimated its impact |
| `research/` | The scripts and logs behind the report's numbers. Earlier exploratory sweeps and their logs are in commit `d45075f` |
| `report/build_report.py` | Rebuilds `report.pdf` from the logged results |
| `colab.py` | Every data insight as a chart, split into notebook cells by `# %%` |

## Running it

The firm's data pack and backtest harness are not in this repo, so copy them in first. Put `harness_public.py` and the `data/` folder in the repo root. Then run:

```
python harness_public.py task1 task1.py
python harness_public.py task2 task2.py
```

The strategies need only numpy and pandas. `harness_public.py` needs pandas 2.2 or later. The results are identical to 10 decimals on pandas 3.0.2 with numpy 2.4.4 and on pandas 2.2.3 with numpy 1.26.4.

To re-run the checks behind the report, in this order from inside `research/`:

```
python t1_final.py
python t2_final.py
python t2_checks.py
python t2_decomp.py
```

`t2_final.py` takes about 15 minutes. `report/build_report.py` also needs reportlab, matplotlib, pillow and pymupdf.
