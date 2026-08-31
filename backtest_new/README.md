# The financial evaluation

Two programs. The first computes the backtest, the second draws the figures.

```
python backtest_new.py
python plot_backtest_new.py
```

They read the saved per-window alarms under
`../Iran_new_run/results*/<asset>/<detector>/exponential/test_scores.csv` and
the price workbooks under `../Iran_new_run/datasets*`. Nothing is refitted, so
both run in seconds and give the same output on every machine.

## Alarm rules

Three rules are computed for every cell.

**`defensive`** moves the position to cash on any day carrying an alarm and
returns to the market when the alarm clears.

**`first_alarm`** holds the asset until the first alarm and then stays in cash
to the end of the test window, so only the date of the first alarm matters.

**`persist`** sits between the two: out after two alarms in a row, back after
five clear days.

## Thresholds

Two threshold modes are computed. `as reported` uses the 0.95 quantile of the
train-normal score distribution. The `far` modes set each detector's threshold
on the crash-free 2019 control so that every detector alarms at the same rate
away from a crash, at 5, 10 and 20 per cent.

The second mode matters because a detector that fires on a third of all days is
in cash on a third of all days, and in a falling market that alone earns a
positive excess return.

## Detectors

`ENS` is EnsembleExpoGAF, the ensemble decision layer. An unweighted 5-of-9
majority alarm is also computed from the saved alarms but is excluded from the
tables and figures; `--keep-majority` includes it.

## The exposure control

Under the first-alarm rule the ranking is almost entirely explained by how long
each detector sits in cash: the correlation between excess return and time in
the market is $-0.77$. A detector that alarms on the first day is out of the
market for the whole window, and across these fifteen cells that alone earns a
positive excess return without saying anything about when the fall arrives.

`backtest_new.py` therefore compares every detector against random exit dates
that spend exactly as long in the market, moving only the dates. A value near
0.5 means the alarm dates carry nothing the calendar did not already give. On
this sample every detector sits between 0.44 and 0.68, and the core encoding
sits at 0.50. The excess returns should be read next to that column, never on
their own.

## Conventions

The decision for a window is taken at its last observation, `window_end`, and
earns the return from that day to the next. Daily returns are clipped at plus
or minus 60 per cent; this is not cosmetic, because WTI settled at minus 37.63
dollars on 20 April 2020 and a simple return is undefined there.

## Output

| File | What it holds |
|---|---|
| `output/backtest_new_cells.csv` | every episode, asset, detector, threshold mode and rule: final return, Buy-and-Hold, the difference, alarm rate, days in market, first alarm date, and days between the first alarm and the onset |
| `output/bar_values_new.csv` | the value behind every bar in the figures |
| `output/table_financial_new.tex` | the financial table in LaTeX |
| `output/iran_2025_2026_first_alarm_6panels_new.pdf` | the dated alarm audit, six panels |
| `output/backtest_bar_<episode>_new.pdf` | one figure per episode |

## Options

```
python backtest_new.py --headline defensive --headline-mode far0.10
python plot_backtest_new.py --strategy defensive --mode far0.10
```

`--headline` and `--strategy` choose the rule; `--headline-mode` and `--mode`
choose the thresholds, either `as reported` or `far0.05`, `far0.10`, `far0.20`.
Every combination is written to `backtest_new_cells.csv` regardless, so the
options only decide what goes into the table and the figures.
