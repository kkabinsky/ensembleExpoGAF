# The financial evaluation, recomputed

Two programs. The first recomputes the backtest, the second redraws the figures
in the layout the manuscript uses.

```
python backtest_new.py
python plot_backtest_new.py
```

They read the saved per-window alarms under
`../Iran_new_run/results*/<asset>/<detector>/exponential/test_scores.csv` and
the price workbooks under `../Iran_new_run/datasets*`. Nothing is refitted, so
both run in seconds and give the same output on every machine.

## It reproduces the published table before it changes anything

`backtest_new.py` prints a check against all eighteen cells of Table 3 under
the published rule and the published thresholds. All eighteen cells reproduce
in all three columns, which is what shows the program is reading the same price
path and the same saved alarms. Only after that does it apply the new rules, so
a reader can see the starting point is the published one.

## What is different

**The rule.** Under the published rule the position moves to cash on any day
carrying an alarm, which pays the same whether the alarm arrives before the
fall or after it. Since the framework claims early warning, `first_alarm` holds
the asset until the first alarm and then stays in cash to the end of the test
window, so only the date of the first alarm matters. `persist` sits between the
two: out after two alarms in a row, back after five clear days.

**The alarm rates.** A detector that fires on a third of all days is in cash on
a third of all days, and in a falling market that alone earns a positive excess
return. Alongside the published thresholds, each detector's threshold is also
set on the crash-free 2019 control so that all of them alarm at the same rate
away from a crash, at 5, 10 and 20 per cent.

**One ensemble is reported.** ENS is EnsembleExpoGAF, the ensemble the
framework proposes. The unweighted 5-of-9 majority alarm is rebuilt from the
saved alarms because the check against Table 3 needs it, but it is left out of
every table and figure. `--keep-majority` puts it back.


## The timing test, and why the headline numbers must not be read alone

Under the first-alarm rule the ranking is almost entirely explained by how long
each detector sits in cash: the correlation between excess return and time in
the market is -0.77. A detector that alarms on the first day is out of the
market for the whole window, and across these fifteen cells that alone earns a
positive excess return without saying anything about when the fall arrives. The
published rule has the same defect in the opposite direction.

`backtest_new.py` therefore also compares every detector against random exit
dates that spend exactly as long in the market, moving only the dates. A value
near 0.5 means the alarm dates carry nothing the calendar did not already give.
On this sample every detector sits between 0.44 and 0.68, and the core encoding
sits at 0.50. The excess returns in the tables above should be read next to
that column, never on their own.

## Conventions kept from the published backtest

The decision for a window is taken at its last observation, `window_end`, and
earns the return from that day to the next. Daily returns are clipped at plus
or minus 60 per cent; this is not cosmetic, because WTI settled at minus 37.63
dollars on 20 April 2020 and a simple return is undefined there.

## Output

| File | What it holds |
|---|---|
| `output/backtest_new_cells.csv` | every episode, asset, detector, threshold mode and rule: final return, Buy-and-Hold, the difference, alarm rate, days in market, first alarm date, and days between the first alarm and the onset |
| `output/bar_values_new.csv` | the value behind every bar in the figures |
| `output/table_financial_new.tex` | Table 3 in its published layout under the new rule |
| `output/iran_2025_2026_first_alarm_6panels_new.pdf` | the replacement for Figure 3 |
| `output/backtest_bar_<episode>_new.pdf` | one figure per episode, in the layout of the supplementary figures |

## Options

```
python backtest_new.py --headline defensive --headline-mode far0.10
python plot_backtest_new.py --strategy defensive --mode far0.10
```

`--headline` and `--strategy` choose the rule; `--headline-mode` and `--mode`
choose the thresholds, either `as reported` or `far0.05`, `far0.10`, `far0.20`.
Every combination is written to `backtest_new_cells.csv` regardless, so the
options only decide what goes into the table and the figures.
