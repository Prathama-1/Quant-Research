"""Builds report.pdf (max 3 pages) from the saved result files, so every number traces to a logged run.
Needs reportlab, matplotlib, pillow and pymupdf on top of the strategy requirements:  python report/build_report.py"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, FrameBreak, Image, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'research', 'results')
OUT_PDF = os.path.join(ROOT, 'report.pdf')
FIG = os.path.join(ROOT, 'report', 'figures')
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, ROOT)
import harness_public as H  # noqa: E402

T1 = json.load(open(os.path.join(RES, 't1_final.json')))
T2 = json.load(open(os.path.join(RES, 't2_final.json')))
CK = json.load(open(os.path.join(RES, 't2_checks.json')))
DC = json.load(open(os.path.join(RES, 't_decomp.json')))

# ---------------------------------------------------------------------------- palette (validated: scripts/validate_palette.js, surface #ffffff)
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({'font.family': 'Calibri', 'font.size': 8, 'axes.edgecolor': AXIS, 'axes.linewidth': .6, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                     'xtick.major.width': .5, 'ytick.major.width': .5, 'xtick.major.size': 2, 'ytick.major.size': 0, 'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': GRID, 'grid.linewidth': .5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 300, 'savefig.dpi': 300, 'legend.frameon': False, 'legend.fontsize': 7.5, 'axes.axisbelow': True})
W_IN, H_IN = 3.62, 1.3


def finish(fig, name):
    p = os.path.join(FIG, name + '.png')
    fig.savefig(p, facecolor='white', bbox_inches='tight', pad_inches=.02)
    plt.close(fig)
    return p


def end_labels(ax, series, room=.1):
    """Direct-label each series just past its last point, and leave room on the right for the labels."""
    xs = [s.index[-1] for s in series]
    x0, x1 = min(s.index[0] for s in series), max(xs)
    ax.set_xlim(x0, x1 + (x1 - x0) * room)
    for s in series:
        ax.annotate('x%.2f' % s.iloc[-1], (s.index[-1], s.iloc[-1]), xytext=(3, 0), textcoords='offset points', va='center', ha='left', color=INK2, fontsize=7.5)


def chart_legs(bars):
    g = bars.groupby(bars.index.normalize())
    d = pd.DataFrame({'o': g.open.first(), 'c': g.close.last()})
    on = (1 + (d.o / d.c.shift() - 1).fillna(0)).cumprod()
    id_ = (1 + (d.c / d.o - 1)).cumprod()
    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    ax.plot(on.index, on.values, color=BLUE, lw=1.2, label='Overnight gap only')
    ax.plot(id_.index, id_.values, color=ORANGE, lw=1.2, label='Open-to-close session only')
    ax.set_yscale('log'); ax.set_yticks([.25, .5, 1, 2, 4, 8]); ax.get_yaxis().set_major_formatter(FuncFormatter(lambda v, _: '%g' % v)); ax.minorticks_off()
    ax.set_ylim(.2, 14)
    end_labels(ax, [on, id_])
    ax.legend(loc='upper left', ncol=2, handlelength=1.4, columnspacing=1.0, borderaxespad=0)
    ax.set_ylabel('Growth of 1, log scale')
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    return finish(fig, 'legs')


def chart_t1(bars, ex):
    e = pd.read_csv(os.path.join(RES, 't1_equity.csv'), index_col=0, parse_dates=True).equity
    r = H.task1_backtest(bars, np.ones(len(bars)), ex)
    bh = (1 + r['daily'] + H.R_CASH / 252).cumprod()
    fig, ax = plt.subplots(figsize=(W_IN, H_IN))
    ax.plot(bh.index, bh.values, color=ORANGE, lw=1.0, label='Buy and hold futures')
    ax.plot(e.index, e.values, color=BLUE, lw=1.3, label='Task 1 strategy')
    end_labels(ax, [e, bh])
    ax.legend(loc='upper left', ncol=2, handlelength=1.4, columnspacing=1.0, borderaxespad=0)
    ax.set_ylabel('Growth of 1')
    ax.set_ylim(.5, max(e.max(), bh.max()) * 1.2)
    ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    return finish(fig, 't1_equity')


def chart_t2():
    e = pd.read_csv(os.path.join(RES, 't2_equity.csv'), index_col=0, parse_dates=True).equity
    n = pd.read_csv(os.path.join(RES, 't2_nifty.csv'), index_col=0, parse_dates=True).nifty
    D = H.load_stocks([os.path.join(ROOT, 'data', 'stocks_daily_train_2021-04-26_2024-06-30.csv')], [os.path.join(ROOT, 'data', 'nifty50_daily_2021-04-26_2024-06-30.csv')])
    ret = D['close'].pct_change(fill_method=None)
    clean = ret.where((ret.abs() < .45) | ret.isna(), 0.0)
    liq = (D['close'] * D['volume']).rolling(60, min_periods=40).median().shift(1) >= 1e8
    ew = (1 + clean.where(liq).mean(axis=1).reindex(e.index).fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(W_IN, H_IN + .35))
    ax.plot(n.index, n.values, color=AQUA, lw=1.0, label='Nifty 50')
    ax.plot(ew.index, ew.values, color=ORANGE, lw=1.0, label='Equal-weight liquid universe, no costs')
    ax.plot(e.index, e.values, color=BLUE, lw=1.3, label='Task 2 strategy')
    end_labels(ax, [e, ew, n])
    ax.legend(loc='upper left', ncol=1, handlelength=1.4, borderaxespad=0, labelspacing=.25)
    ax.set_ylabel('Growth of 1')
    ax.set_ylim(.8, e.max() * 1.1)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    return finish(fig, 't2_equity')


def chart_buckets():
    b = CK['buckets']
    labs = ['Under 1', '1-5', '5-10', '10-25', '25-50', '50-100', 'Over 100']
    v = [x['mean'] * 100 for x in b]
    fig, ax = plt.subplots(figsize=(W_IN, 1.35))
    ax.bar(range(len(v)), v, width=.55, color=BLUE)
    for i, x in enumerate(v):
        ax.text(i, x + 8, '+%.0f%%' % x, ha='center', va='bottom', color=INK2, fontsize=7.5)
    ax.set_xticks(range(len(v))); ax.set_xticklabels(labs)
    ax.set_ylabel('Mean return, %'); ax.set_ylim(0, max(v) * 1.2)
    ax.set_xlabel('60-day median traded value on 28 Feb 2022, Rs crore', labelpad=1.5)
    ax.tick_params(axis='x', length=0)
    return finish(fig, 'buckets')


# ---------------------------------------------------------------------------- formatting
pdfmetrics.registerFont(TTFont('Cal', r'C:\Windows\Fonts\calibri.ttf'))
pdfmetrics.registerFont(TTFont('Cal-B', r'C:\Windows\Fonts\calibrib.ttf'))
pdfmetrics.registerFont(TTFont('Cal-I', r'C:\Windows\Fonts\calibrii.ttf'))
pdfmetrics.registerFontFamily('Cal', normal='Cal', bold='Cal-B', italic='Cal-I', boldItalic='Cal-B')
BODY = ParagraphStyle('body', fontName='Cal', fontSize=9.3, leading=11.1, spaceAfter=3.2, textColor=colors.HexColor(INK))
SMALL = ParagraphStyle('small', parent=BODY, fontSize=8.3, leading=9.8, textColor=colors.HexColor(INK2), spaceAfter=2.5)
H1 = ParagraphStyle('h1', fontName='Cal-B', fontSize=12, leading=14, spaceBefore=1, spaceAfter=3, textColor=colors.HexColor(INK))
H2 = ParagraphStyle('h2', fontName='Cal-B', fontSize=9.6, leading=11.4, spaceBefore=4, spaceAfter=1.5, textColor=colors.HexColor(INK))
TITLE = ParagraphStyle('title', fontName='Cal-B', fontSize=16, leading=18.5, spaceAfter=1, textColor=colors.HexColor(INK))
CAP = ParagraphStyle('cap', parent=SMALL, spaceBefore=3, spaceAfter=1, textColor=colors.HexColor(INK), fontName='Cal-B', fontSize=8.5)
CELL = ParagraphStyle('cell', fontName='Cal', fontSize=8.3, leading=9.5, textColor=colors.HexColor(INK))
CELLB = ParagraphStyle('cellb', parent=CELL, fontName='Cal-B')
L, R, Tm, Bm, GAP = 30, 30, 24, 26, 14
PW, PH = A4
COLW = (PW - L - R - GAP) / 2


def pct(x, d=1):
    return ('%.' + str(d) + 'f%%') % (100 * x)


def num(x, d=2):
    return ('%.' + str(d) + 'f') % x


def P(t, s=BODY):
    return Paragraph(t, s)


def table(rows, widths, head=1, bold_rows=(), pad=1.2, boxed=None):
    data = [[Paragraph(c, CELLB if (i < head or i in bold_rows) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=head, hAlign='LEFT')
    st = [('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 2), ('RIGHTPADDING', (0, 0), (-1, -1), 2),
          ('TOPPADDING', (0, 0), (-1, -1), pad), ('BOTTOMPADDING', (0, 0), (-1, -1), pad + .4),
          ('LINEBELOW', (0, head - 1), (-1, head - 1), .6, colors.HexColor(AXIS)), ('LINEBELOW', (0, -1), (-1, -1), .4, colors.HexColor(GRID)),
          ('LINEABOVE', (0, 0), (-1, 0), .6, colors.HexColor(AXIS))]
    if boxed:
        st.append(('BACKGROUND', boxed[0], boxed[1], colors.HexColor('#e9f1fb')))
    t.setStyle(TableStyle(st))
    return t


def img(path, width=COLW):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return Image(path, width=width, height=width * h / w)


def footer(c, d):
    c.saveState(); c.setFont('Cal', 7.5); c.setFillColor(colors.HexColor(MUTED))
    c.drawString(L, 13, 'Prathama Bharadwaj: quant research take-home')
    c.drawRightString(PW - R, 13, 'Page %d' % d.page)
    c.restoreState()


def build():
    bars = H.load_bars(os.path.join(ROOT, 'data', 'nifty50_15min_train_2017-04-03_2022-01-31.csv'))
    ex = H.load_expiries(os.path.join(ROOT, 'data', 'nifty_monthly_expiries_2017-01_2022-01.csv'))
    f_legs, f_t1, f_t2, f_b = chart_legs(bars), chart_t1(bars, ex), chart_t2(), chart_buckets()

    b1, st1, b2, off = T1['base'], T1['stress'], T2['base'], T2['offsets']
    S = []
    # ---------------------------------------------------------------- header
    S += [P('Nifty overnight timing and NSE momentum', TITLE),
          P('Prathama Bharadwaj. Quant Research Intern take-home, H TO H Partners LLP. All numbers are from <font name="Cal-I">harness_public.py</font>, after costs, unless stated.', SMALL)]
    S.append(table([
        ['', 'Sharpe', 'Max DD', 'CAGR', 'Trades/yr', 'Bar to beat', 'Firm reference: Sharpe, DD, CAGR', 'My expectation on hidden data'],
        ['Task 1, Apr 2017 to Jan 2022', num(b1['sharpe']), pct(b1['max_dd']), pct(b1['cagr']), '%.0f' % (b1['trades'] / b1['years']), 'Sharpe &gt; 1.0, DD better than -15%', '1.22, -12.8%, 23.3%', 'Sharpe about 1, DD within -10%'],
        ['Task 2, 2 Mar 2022 to 30 Jun 2024', num(b2['sharpe']), pct(b2['max_dd']), pct(b2['cagr']), '%.0f' % b2['trades_per_yr'], 'Sharpe &gt; 1.9, DD better than -25%', '1.93, -23.9%, 51.6%', 'Sharpe 1 to 1.5, DD within -15%'],
    ], [118, 32, 34, 32, 42, 84, 80, 113]))
    S.append(Spacer(1, 3))
    S.append(P('Both clear their bars by a wide margin, but both in-sample windows flatter them: Task 1 through an overnight premium that shrank after 2021, Task 2 through a universe where 15 random stocks already score Sharpe 1.66. '
               'The stress and out-of-sample results below are the better guide, and they are where my expectations come from.', BODY))
    S.append(FrameBreak())

    # ---------------------------------------------------------------- task 1, page 1
    S.append(P('Task 1: long overnight, in healthy markets only', H1))
    S.append(P('<b>Idea.</b> Nifty\'s return arrives while the market is shut. From 2017 to Jan 2022 the overnight gap has a Sharpe of +3.4 (t = 7.5) and the open-to-close session -1.6 (t = -3.4), with those signs in every full year. '
               'The Task 2 daily file confirms it past the 15-minute data: +2.6 and -0.8 for Feb 2022 to Jun 2024. The 09:15 bar alone loses 6.5 bps a day (t = -6.2).'))
    S.append(P('<b>Why it should work.</b> Overnight holders carry gap risk that is hard to hedge or exit while the cash market is shut, and intraday leverage is normally closed by the end of the day, so overnight capital should be scarce and get paid. '
               'Buying pressure at the open then lifts the first print, which fades (Berkman et al. 2012; Lou, Polk and Skouras 2019). The premium is thin and the tail is large when the market falls, so I only take it when the market is calm and rising.'))
    S.append(P('<b>Rules.</b> At the close of the 14:30 bar, go long if (1) price is above its 20-day average, (2) the session so far is at least -0.25%, (3) 20-day realised vol is under 35%, and (4) the last 120 paper trades averaged at least 7 bps, which is the round-trip cost plus carry. '
               'Fill at 14:45, exit at the next morning\'s open, otherwise stay flat. Never short: shorts pay the gap, and long-short lost to long-flat in 10 of 11 trend configs. Deciding at 14:30 still fills before the close with a 2-bar delay. '
               'Rule 4 never fired in-sample, so it cannot have been fitted to this data. It is there in case the premium fades.'))
    S.append(P('Each part of the day on its own, no costs', CAP)); S.append(img(f_legs))
    S.append(P('Strategy against buy and hold, after all costs', CAP)); S.append(img(f_t1))
    S.append(P('Results and stress cases', CAP))
    oos = T1['oos']
    t_rows = [['Case', 'CAGR', 'Max DD', 'Sharpe'],
              ['Official, Apr 2017 to Jan 2022', pct(b1['cagr']), pct(b1['max_dd']), num(b1['sharpe'])],
              ['1-bar fill delay', '%.1f%%' % st1['delay_1']['cagr'], '%.1f%%' % st1['delay_1']['dd'], num(st1['delay_1']['sharpe'])],
              ['2-bar fill delay', '%.1f%%' % st1['delay_2']['cagr'], '%.1f%%' % st1['delay_2']['dd'], num(st1['delay_2']['sharpe'])],
              ['2x costs', '%.1f%%' % st1['2x_cost']['cagr'], '%.1f%%' % st1['2x_cost']['dd'], num(st1['2x_cost']['sharpe'])],
              ['Best year (%d) removed' % T1['ex_best_year']['best_year'], pct(T1['ex_best_year']['cagr']), pct(T1['ex_best_year']['dd']), num(T1['ex_best_year']['sharpe'])],
              ['2017 to 2019', pct(T1['sub']['dev 2017-19']['cagr']), pct(T1['sub']['dev 2017-19']['dd']), num(T1['sub']['dev 2017-19']['sharpe'])],
              ['2020 to Jan 2022', pct(T1['sub']['val 2020-Jan22']['cagr']), pct(T1['sub']['val 2020-Jan22']['dd']), num(T1['sub']['val 2020-Jan22']['sharpe'])],
              ['Out-of-sample, Feb 22 to Jun 24', pct(oos['cagr']), pct(oos['dd']), '%s (%s to %s)' % (num(oos['sharpe']), num(oos['sharpe_min']), num(oos['sharpe_max']))]]
    S.append(table(t_rows, [COLW - 32 - 36 - 66, 32, 36, 66], bold_rows=(1,)))
    y = b1['yearly']
    S.append(Spacer(1, 3))
    S.append(table([['Year', '2017 (Apr on)', '2018', '2019', '2020', '2021', '2022 (Jan)'], ['Return'] + [pct(y[k]) for k in ['2017', '2018', '2019', '2020', '2021', '2022']]], [34, 58, 34, 34, 34, 34, COLW - 228]))
    S.append(Spacer(1, 3))
    tr, cd = T1['trades'], T1['cost_drag']
    S.append(P('<b>Turnover and cost drag.</b> %.0f round trips a year, long on %.0f%% of nights and %.0f%% of bars. Zero-cost CAGR is %s, so trading and roll costs take %.1f points a year. Win rate %.0f%%, mean trade %.1f bps, worst trade %s. Top 10 trades are %.0f%% of net profit.' % (
        b1['trades'] / b1['years'], 100 * T1['nights_held_share'], 100 * b1['time_in_market'], pct(cd['gross_cagr']), 100 * (cd['gross_cagr'] - cd['net_cagr']), 100 * tr['win'], tr['mean_bps'], pct(tr['worst'], 2), 100 * tr['top10_of_net'])))
    S.append(P('<b>Out-of-sample check.</b> The 15-minute data stops in Jan 2022, so I built later bars from the real daily OHLC with seeded synthetic paths inside each day. Gaps and ranges are real, so this only tests day-level logic. On real in-sample days it landed within about 0.6 of the true Sharpe, usually below. '
               'It is not a clean holdout, because I chose the session floor after seeing it. Across the 192 filter combinations the median is 1.12, which is why I expect about 1. '
               'Bootstrap 95%% CI on the official Sharpe: %s to %s. Deflated Sharpe ratio for %d logged configs: %s.' % (num(DC['t1_ci']['lo']), num(DC['t1_ci']['hi']), DC['t1_dsr']['n_trials'], num(DC['t1_dsr']['dsr']))))

    S.append(P('Sensitivity: in-sample Sharpe | out-of-sample Sharpe', CAP))
    g = pd.DataFrame(T1['grid'])
    fl = sorted(g['floor'].unique())
    rows = [['MA days \\ session floor'] + ['%.2f%%' % (100 * f) for f in fl]]
    for ma in sorted(g.ma.unique()):
        x = g[g.ma == ma].set_index('floor')
        rows.append([str(ma)] + ['%s | %s' % (num(x.loc[f, 'sh']), num(x.loc[f, 'oos'])) for f in fl])
    i_ma, i_fl = 1 + sorted(g.ma.unique()).index(20), 1 + fl.index(-.0025)
    S.append(table(rows, [COLW - 4 * 47] + [47] * 4, boxed=((i_fl, i_ma), (i_fl, i_ma))))
    S.append(Spacer(1, 2))
    S.append(P('All %d cells beat the bar, with max DD between %s and %s.' % (len(g), pct(g.dd.min()), pct(g.dd.max())), SMALL))
    oat = T1['oat']
    rng = lambda k, rows_=None: '%s to %s' % (num(min(r['sh'] for r in (rows_ or oat[k]))), num(max(r['sh'] for r in (rows_ or oat[k]))))
    S.append(P('Parameters and ranges tested', CAP))
    fl_row = g[g.ma == 20]
    S.append(table([['Parameter', 'Value', 'Range tested', 'Sharpe over range'],
                    ['ma_days', '20', '5 to 200', '2.28 to 2.72, no other filters'],
                    ['decide_at', '14:30', '13:30 to 15:00', '%s up to 14:30; later fails the 2-bar delay' % rng('decide_at')],
                    ['session_floor', '-0.25%', '-1% to 0', '%s to %s' % (num(fl_row.sh.min()), num(fl_row.sh.max()))],
                    ['vol_cap', '35%', '15% to off', '%s from 25%% up' % rng('vol_cap')],
                    ['gate_nights', '120', '60 to 250', rng('gate_nights')],
                    ['gate_bps', '7', 'off to 15', rng('gate_bps')],
                    ['Vol window*', '20 days', '10, 20', '2.88, 2.89, no floor or gate']],
                   [COLW - 34 - 60 - 112, 34, 60, 112]))
    S.append(Spacer(1, 1.5))
    S.append(P('*Module constant, beyond the six in PARAMS.', SMALL))
    d = T1['decay']
    S.append(P('What would break it', H2))
    cm = T1['cost_mult']
    S.append(P('<b>The premium fading.</b> Removing 0, 3, 6, 9 or 12 bps from every overnight gap gives out-of-sample Sharpe %s, %s, %s, %s, %s, and in-sample %s, %s, %s, %s, %s. The real out-of-sample gap averages 9.4 bps, so losing 6 bps makes it unprofitable. '
               '<b>Fills.</b> The harness credits the spot open in full, but futures may not reach that print. The 1-bar delay, which exits at 09:30, is my best proxy, and it holds at %s. '
               '<b>Costs.</b> Sharpe is %s at 3x and %s at 4x, so all-in round trips above about 17 bps end it. '
               '<b>A large gap with the filters on.</b> The worst trade was %s, but filters cannot see overnight news, and Nifty opened 9%% down in Mar 2020.' % (
                   num(d['0']['oos_sh']), num(d['3']['oos_sh']), num(d['6']['oos_sh']), num(d['9']['oos_sh']), num(d['12']['oos_sh']), num(d['0']['is_sh']), num(d['3']['is_sh']), num(d['6']['is_sh']), num(d['9']['is_sh']), num(d['12']['is_sh']),
                   num(st1['delay_1']['sharpe']), num(cm['3']['sharpe']), num(cm['4']['sharpe']), pct(tr['worst'], 2))))
    S.append(PageBreak())

    # ---------------------------------------------------------------- task 2, page 2
    S.append(P('Task 2: momentum, in broad markets only', H1))
    S.append(P('<b>Idea.</b> Own the 15 stocks with the best return per unit of risk, but only while most of the market is above trend. News diffuses slowly through mid and small caps, which is why momentum pays, but its edge depends on the market: '
               'the top 15 by 12-month momentum lagged the universe by 3.4% a quarter in 2022 and beat it by 13% in 2023. The filter keeps the strategy out of the first kind of market.'))
    S.append(P('<b>Rules.</b> Each close: (1) if under 60% of liquid stocks (Rs 10 cr median traded value) are above their 50-day average, sell everything and buy nothing. (2) Rank liquid stocks by mean over std of 126-day returns, excluding vol above 65%. '
               '(3) Every 10 days, sell holdings ranked below 40. (4) Fill free slots with stop orders on the best-ranked names at their 10-day high, 1/15 of equity each. No stop-losses: they cut Sharpe by up to 0.25 and did not reduce drawdown.'))
    S.append(P('<b>Corporate actions.</b> One-day moves beyond 45% (SPLPETRO -49%, and NMDC +86%, which the README does not list) are zeroed in all indicators, held names are sold at the next open, and new buys are blocked for 10 days. A -50% shock injected into a held name cost 0.22 of Sharpe.'))
    S.append(P('Growth of 1 from 2 Mar 2022', CAP)); S.append(img(f_t2))
    S.append(P('Results and stress cases', CAP))
    o, sc, ub = off['eval'], T2['scen'], T2['subuni']
    fm = {r['value']: r for r in T2['const']['MIN_TRADED_VALUE']}
    t2_rows = [['Case', 'CAGR', 'Max DD', 'Sharpe'],
               ['Official, 2 Mar 2022 to 30 Jun 2024', pct(b2['cagr']), pct(b2['max_dd']), num(b2['sharpe'])],
               ['Mean of 4 rebalance offsets', pct(o['cagr']), pct(o['dd']), num(o['sh'])],
               ['2x costs', pct(T2['cost_mult']['2']['cagr']), pct(T2['cost_mult']['2']['dd']), num(T2['cost_mult']['2']['sharpe'])],
               ['Best year (%d) removed' % T2['ex_best_year']['best_year'], pct(T2['ex_best_year']['cagr']), pct(T2['ex_best_year']['dd']), num(T2['ex_best_year']['sharpe'])],
               ['Random 70% of stocks, 20 draws', pct(ub['cagr_mean']), pct(ub['dd_mean']), '%s (min %s)' % (num(ub['sh_mean']), num(ub['sh_min']))],
               ['Liquidity floor Rs 50 cr, not Rs 10 cr', pct(fm[5e8]['cagr']), pct(fm[5e8]['dd']), num(fm[5e8]['sh'])],
               ['Tough: universe about +10% a year, not +47%', pct(off['tough']['cagr']), pct(off['tough']['dd']), num(off['tough']['sh'])],
               ['Bear: universe about -10% a year', pct(off['bear']['cagr']), pct(off['bear']['dd']), num(off['bear']['sh'])],
               ['Oct 2021 to Feb 2022, before the window*', pct(sc['warm']['total']), pct(sc['warm']['dd']), num(sc['warm']['sharpe'])]]
    S.append(table(t2_rows, [COLW - 34 - 36 - 56, 34, 36, 56], bold_rows=(1,)))
    S.append(P('*Total return, not CAGR, and not a scored result, as the brief reserves 2021 for indicator warm-up. It did inform my design choices: see Checks on page 3. The liquidity floor, tough and bear rows are means over 4 offsets.', SMALL))
    y = b2['yearly']
    S.append(table([['Year', '2022 (Mar on)', '2023', '2024 (to Jun)'], ['Return', pct(y['2022']), pct(y['2023']), pct(y['2024'])]], [40, 62, 40, COLW - 142]))
    S.append(Spacer(1, 3))
    cd2, tp = T2['cost_drag'], T2['top10']
    S.append(P('<b>Turnover and cost drag.</b> %.0f trades a year in %d names, %.1fx turnover, mean hold %.0f days, in the market %.0f%% of the time. Zero-cost CAGR is %s, so costs take %.1f points a year.' % (
        b2['trades_per_yr'], tp['n_symbols'], cd2['turnover'], b2['avg_hold_days'], 100 * b2['time_in_market'], pct(cd2['gross_cagr']), 100 * (cd2['gross_cagr'] - cd2['net_cagr']))))
    S.append(P('<b>Top 10 trades: %.0f%% of net profit</b> (%.0f%% of gross profit), and the top 5 names are %.0f%% of net. This concentration in a few multi-baggers is the weakest feature of the result. Random 70%% sub-universes still average %s, so it does not rest on any one name.' % (
        100 * tp['share_of_net'], 100 * tp['share_of_gross'], 100 * T2['top_names']['top5_share'], num(ub['sh_mean']))))
    S.append(FrameBreak())
    S.append(P('What each layer adds', CAP))
    dc = DC['decomp']; nr = T2['no_regime']['eval']
    S.append(table([['Same rules', 'Sharpe', 'CAGR', 'Max DD', 'Tough Sharpe'],
                    ['Random picks, no filter', '%s (std %s)' % (num(dc['random_fixed_noregime_sh']), num(dc['random_fixed_noregime_sh_std'])), pct(dc['random_fixed_noregime_cagr']), pct(dc['random_fixed_noregime_dd']), num(dc['random_fixed_noregime_tsh'])],
                    ['Random picks + breadth filter', '%s (std %s)' % (num(dc['random_fixed_regime_sh']), num(dc['random_fixed_regime_sh_std'])), pct(dc['random_fixed_regime_cagr']), pct(dc['random_fixed_regime_dd']), num(dc['random_fixed_regime_tsh'])],
                    ['Selection, no filter', num(nr['sh']), pct(nr['cagr']), pct(nr['dd']), num(T2['no_regime']['tough']['sh'])],
                    ['Selection + filter, submitted', num(o['sh']), pct(o['cagr']), pct(o['dd']), num(off['tough']['sh'])]],
                   [COLW - 66 - 32 - 34 - 36, 66, 32, 34, 36], bold_rows=(4,)))
    S.append(Spacer(1, 2))
    S.append(P('Random rows use one fixed random ranking per seed, 24 seeds.', SMALL))
    S.append(P('The filter alone mostly halves drawdown. Selection alone adds return. Together they are worth more, which fits momentum paying best in healthy markets. Bootstrap 95%% CI on the official Sharpe: %s to %s. Deflated Sharpe ratio for %s logged configs: %s, which says nothing about universe bias.' % (
        num(DC['t2_ci']['lo']), num(DC['t2_ci']['hi']), format(DC['t2_dsr']['n_trials'], ','), num(DC['t2_dsr']['dsr'])), BODY))
    S.append(P('Sensitivity: Sharpe | tough-case Sharpe, mean of 4 offsets', CAP))
    g2 = pd.DataFrame(T2['grid'])
    lbs = sorted(g2.lookback.unique())
    rows = [['Breadth min \\ lookback days'] + [str(x) for x in lbs]]
    for bm in sorted(g2.breadth_min.unique()):
        x = g2[g2.breadth_min == bm].set_index('lookback')
        rows.append(['%.2f' % bm] + ['%s | %s' % (num(x.loc[l, 'sh']), num(x.loc[l, 't_sh'])) for l in lbs])
    ic, ir = 1 + lbs.index(126), 1 + sorted(g2.breadth_min.unique()).index(.6)
    S.append(table(rows, [COLW - 3 * 56] + [56] * 3, boxed=((ic, ir), (ic, ir))))
    S.append(Spacer(1, 2))
    S.append(P('All cells beat the bar, with max DD between %s and %s. Neighbouring cells differ by up to about 0.7, so treat smaller gaps as noise. The 0.70 row has the best in-sample Sharpe, but I rejected it because it would rarely invest in a lower-drift market, as the tough case shows.' % (pct(g2.dd.min()), pct(g2.dd.max())), SMALL))
    ot, ct = T2['oat'], T2['const']
    rg = lambda dct, k, keep=None: '%s to %s' % (num(min(r['sh'] for r in dct[k] if keep is None or keep(r))), num(max(r['sh'] for r in dct[k] if keep is None or keep(r))))
    S.append(P('Parameters and ranges tested, mean of 4 offsets', CAP))
    jc = ct['JUMP_COOLOFF']
    S.append(table([['Parameter', 'Value', 'Range tested', 'Sharpe over range'],
                    ['lookback', '126 days', '84 to 252 days', '%s to %s, 84 to 189 days' % (num(g2[g2.breadth_min == .6].sh.min()), num(g2[g2.breadth_min == .6].sh.max()))],
                    ['breadth_min', '0.60', '0 to 0.80', '1.83 to 3.33'],
                    ['top_k', '15', '8 to 15', rg(ot, 'top_k')],
                    ['sell_rank', '40', '15 to 90', rg(ot, 'sell_rank')],
                    ['max_vol', '65%', '35% to off', '%s, from 45%% up' % rg(ot, 'max_vol')],
                    ['breakout_days', '10', '1 to 40', rg(ot, 'breakout_days')],
                    ['Review interval*', '10 days', '5 to 21 days', rg(ct, 'REVIEW_EVERY')],
                    ['Liquidity floor*', 'Rs 10 cr', 'Rs 10 to 200 cr', rg(ct, 'MIN_TRADED_VALUE')],
                    ['Breadth MA*', '50 days', '20 to 200 days', '%s, up to 100 days' % rg(ct, 'BREADTH_MA')],
                    ['Vol window*', '120 days', '60 to 250 days', rg(ct, 'VOL_DAYS')],
                    ['Jump limit*', '45%', '30% to off', rg(ct, 'JUMP')],
                    ['Jump cool-off*', '10 days', '0 to 40 days', '%s, no effect' % num(jc[0]['sh'])]],
                   [COLW - 44 - 64 - 92, 44, 64, 92]))
    S.append(Spacer(1, 1.5))
    S.append(P('*Module constant, beyond the six in PARAMS.', SMALL))
    S.append(P('What would break it', H2))
    S.append(P('<b>Survivorship</b> is the largest risk, covered on the next page. '
               '<b>Slow reaction.</b> The filter needs speed: if the signal arrives 1, 2, 3 or 10 days late, Sharpe is 2.85, 2.50, 1.77 and 1.60. '
               '<b>Choppy markets.</b> Breadth hovering near 60% means selling lows and rebuying highs at 0.4% a round trip. This already happened: from Feb to Jun 2024 the strategy made +0.7% against +17.8% for the universe, with 14 filter flips and 36% winning trades. Only 5% of the 800 settings on page 3 made over 10% in those months, so tuning does not fix it. '
               '<b>A less one-way market.</b> Apr 2023 to Jan 2024 gave 75% of the log return, when the universe alone scored Sharpe 5.65. Outside that stretch the strategy\'s Sharpe is 1.42 and its yearly return 16.8%. '
               '<b>Loss of leadership.</b> With no multi-baggers to catch, the top-10 share above suggests much of the return goes. '
               '<b>Corporate actions.</b> An event like ABFRL or STAR on a held name costs about 3% of equity at 1/15 weight, and nothing can see it coming. '
               '<b>Capacity.</b> The Rs 10 cr floor keeps small names eligible, and 0.2% a side may understate the cost of trading them in size.'))
    S.append(PageBreak())

    # ---------------------------------------------------------------- page 3
    S.append(P('Survivorship bias in the Task 2 universe', H1))
    S.append(P('The list is the set of stocks liquid in Oct 2026, so membership looks into the future: all 555 names are still trading on the last date, and failures, mergers and flopped IPOs are missing. It shows up three ways. '
               'An equal-weight portfolio of everything made +148% before costs against +44.6% for Nifty, with only 11.4% of names falling. 15 random stocks score Sharpe 1.66 (std 0.36), so the firm\'s reference of 1.93 is under one standard deviation above chance. '
               'And the less liquid a stock was in 2022, the more it later rose, because a stock with almost no trading then is only here if something large happened to it.'))
    S.append(P('Mean return to Jun 2024, by liquidity on 28 Feb 2022', CAP)); S.append(img(f_b))
    fl_ = fm
    S.append(P('<b>Effect on the strategy.</b> Raising the liquidity floor from Rs 10 cr to Rs 25, 50 and 100 cr takes Sharpe from %s to %s, %s and %s, and CAGR from %s to %s, %s and %s. That gap of about 19 points of CAGR and 1.0 of Sharpe at Rs 50 cr is my rough estimate of the flattery. It is a guide, not a bound: '
               'it includes any genuine small-cap effect, and it ignores the bias remaining above Rs 50 cr. Injecting up to 60 failing clones of real stocks barely moved the result (Sharpe %s to %s), so the direct cost of holding failures is small, because losers exit quickly. The larger effect is inflated winners.' % (
                   num(fl_[1e8]['sh']), num(fl_[2.5e8]['sh']), num(fl_[5e8]['sh']), num(fl_[1e9]['sh']), pct(fl_[1e8]['cagr']), pct(fl_[2.5e8]['cagr']), pct(fl_[5e8]['cagr']), pct(fl_[1e9]['cagr']),
                   num(min(min(x['sh'] for x in T2['ghost']), min(x['sh'] for x in CK['sudden']))), num(max(max(x['sh'] for x in T2['ghost']), max(x['sh'] for x in CK['sudden']))))))
    S.append(P('<b>Measuring it properly.</b> Rebuild a point-in-time universe from NSE bhavcopies, including later-delisted names, apply the same liquidity rule each day, and re-run. Failing that, compare the equal-weight universe with an index whose past membership is fixed, such as Nifty 500 Equal Weight. Details are in <font name="Cal-I">survivorship_note.md</font>.'))
    S.append(P('Would more tuning help?', H1))
    S.append(P('No. I tested 800 settings per task with cross-validation (CSCV, Bailey et al.): pick the best on half the data, score it on the other half, and repeat for every split.'))
    S.append(table([['Sharpe', 'Task 1, all', 'Task 1, no 14:45', 'Task 2'],
                    ['Best settings, on the half used to pick them', '3.78', '3.46', '3.72'],
                    ['Same settings, on the unseen half', '2.97', '2.35', '2.84'],
                    ['Submitted settings', '2.95', '2.95', '3.01'],
                    ['How often the best settings beat submitted', '54%', '5%', '47%']],
                   [COLW - 3 * 42, 42, 42, 42]))
    S.append(Spacer(1, 3))
    S.append(P('I chose the submitted settings after seeing this data, so their row is flattered. The fair reading is the fall of about 1 Sharpe from picking winners, and I expect mine to fall on hidden data too.'))
    S.append(P('<b>Task 1 trade-off.</b> Deciding at 14:45 gives Sharpe 3.57, worst loss -3.7% and 25.1% a year, a significant gain (95% CI +0.22 to +1.03). But a 2-bar delay then scores -3.37, because the fill misses the overnight gap. I kept 14:30 for that reason. '
               '<b>Task 2 alternatives.</b> None is significantly better (means of 4 offsets, against 2.97): a skip-month signal 3.27, 12 positions with sell rank 20 3.09, and breadth 0.70 3.33, which falls to 0.78 in the tough case.'))
    S.append(FrameBreak())
    S.append(P('Checks', H1))
    la = T1['lookahead']
    S.append(P('<b>No look-ahead.</b> The harness check passes. On Task 1, %d further truncation cuts gave no mismatches, and corrupting all prices after a cut changed nothing before it. Task 2 only ever sees <font name="Cal-I">hist</font>.' % la['cuts']))
    S.append(P('<b>Reproducible.</b> Deterministic, with no dates in the code. Results are identical to 10 decimals on pandas 3.0.2 with numpy 2.4.4 and on pandas 2.2.3 with numpy 1.26.4. Task 1 runs in under 1 second and Task 2 in about 5 seconds, including on a test window as long as the hidden one.'))
    S.append(P('<b>Robust to odd data.</b> Both handle short sessions and cold starts: with too little history they stay flat. Task 2 also handles missing prices on a held stock and injected price shocks.'))
    S.append(P('<b>What I looked at beyond the scored windows.</b> The brief reserves 2021 for indicator warm-up. The scored runs trade nothing before the window, but in research I also ran Task 2 over Oct 2021 to Feb 2022 as a weak-market check, and that and the tough case informed its design, so neither is independent evidence. '
               'Task 1\'s later-period check uses the pack\'s daily Nifty file for Feb 2022 to Jun 2024, which overlaps the hidden window, so that stretch is not unseen by me.'))
    S.append(P('<b>Selection effects.</b> %d Task 1 and %s Task 2 configs are logged. Designs were compared by averages across many configs, not by the best row. The same discipline is why I rejected the highest in-sample Task 2 threshold and why Task 2 tables average over rebalance offsets: moving the rebalance day alone shifts Sharpe by about 0.3.' % (DC['t1_dsr']['n_trials'], format(DC['t2_dsr']['n_trials'], ','))))
    S.append(P('Notes on the brief', H1))
    S.append(P('<b>Parameter limit.</b> The email allows more than 6 parameters, but <font name="Cal-I">check_params</font> rejects them. So PARAMS has 6 and the others are module constants, listed with their tests above. '
               '<b>Universe size.</b> The email says about 585 stocks, while the data and README have 555. '
               '<b>Undisclosed anomaly.</b> NMDC jumps +85.7% on 2022-10-27. The README lists only drops.'))
    S.append(P('<b>Other files.</b> <font name="Cal-I">failed_ideas.md</font> has what did not work, with numbers. Research code and all logged runs are at github.com/Prathama-1/Quant-Research, without the firm\'s data or harness.'))
    S.append(P('With more time', H1))
    S.append(P('<b>Nifty futures bars.</b> How much of the spot gap can a futures trade actually capture? This is the largest open question in Task 1, and the data would answer it directly.'))
    S.append(P('<b>A point-in-time stock universe.</b> This is the largest open question in Task 2, as set out on the left.'))
    S.append(P('<b>A longer history.</b> Task 1\'s filter has been tested on one crash and one recovery, and Task 2 on a window that is mostly a bull market. I would want both tested through a long bear market such as 2008 before trusting them with capital.'))

    doc = BaseDocTemplate(OUT_PDF, pagesize=A4, leftMargin=L, rightMargin=R, topMargin=Tm, bottomMargin=Bm, title='Nifty overnight timing and NSE momentum', author='Prathama Bharadwaj')
    head_h = 134
    top = PH - Tm
    f_head = Frame(L, top - head_h, PW - L - R, head_h, 0, 0, 0, 0, id='head')
    f_l1 = Frame(L, Bm, COLW, top - head_h - Bm, 0, 0, 0, 0, id='l1')
    f_r1 = Frame(L + COLW + GAP, Bm, COLW, top - head_h - Bm, 0, 0, 0, 0, id='r1')
    f_l = Frame(L, Bm, COLW, top - Bm, 0, 0, 0, 0, id='l')
    f_r = Frame(L + COLW + GAP, Bm, COLW, top - Bm, 0, 0, 0, 0, id='r')
    doc.addPageTemplates([PageTemplate(id='first', frames=[f_head, f_l1, f_r1], onPage=footer, autoNextPageTemplate='rest'), PageTemplate(id='rest', frames=[f_l, f_r], onPage=footer)])
    doc.build(S)


if __name__ == '__main__':
    build()
    import pymupdf
    d = pymupdf.open(OUT_PDF)
    print('pages', len(d), 'size KB', os.path.getsize(OUT_PDF) // 1024)
    for i, pg in enumerate(d):
        pg.get_pixmap(dpi=110).save(os.path.join(FIG, 'page%d.png' % (i + 1)))
